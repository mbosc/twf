

from copy import deepcopy
import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace

from datasets.seq_cub200 import MyCUB200, SequentialCUB200
from utils.conf import base_path, get_device
from datasets.seq_cifar100 import MyCIFAR100, SequentialCIFAR100
from datasets.seq_tinyimagenet import MyTinyImagenet, SequentialTinyImagenet32, SequentialTinyImagenet32R
from torchvision import transforms
from tqdm import tqdm
import os
from onedrivedownloader import download as dn

def get_ckpt_remote_url(args):
    if args.pre_dataset == "cifar100":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21108&authkey=AFsCv4BR-bmTUII" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs18_cifar100.pth"
        
    elif args.pre_dataset == "tinyimgR":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21106&authkey=AKTxp5LFQJ9z9Ok" width="98" height="120" frameborder="0" scrolling="no"></iframe>', "erace_pret_on_tinyr.pth"
            
    elif args.pre_dataset == "imagenet":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21107&authkey=ADHhbeg9cUoqJ0M" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs50_imagenet_full.pth"
    

    else:
        raise ValueError("Unknown auxiliary dataset")


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def get_aux_dataset(self):
        if self.args.pre_dataset == 'cifar100':
            aux_dset = MyCIFAR100(base_path(
            ) + 'CIFAR100', train=True, download=True, transform=SequentialCIFAR100.TRANSFORM)
            aux_test_dset = MyCIFAR100(base_path(
            ) + 'CIFAR100', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), SequentialCIFAR100.get_normalization_transform()]))

        elif self.args.pre_dataset == 'tinyimg':
            aux_dset = MyTinyImagenet(base_path(
            ) + 'TINYIMG', train=True, download=True, transform=SequentialTinyImagenet32.TRANSFORM)
            aux_test_dset = MyTinyImagenet(base_path(
            ) + 'TINYIMG', train=False, download=True, transform=SequentialTinyImagenet32.TEST_TRANSFORM)

        elif self.args.pre_dataset == 'tinyimgR':
            aux_dset = MyTinyImagenet(base_path(
            ) + 'TINYIMG', train=True, download=True, transform=SequentialTinyImagenet32R.TRANSFORM)
            aux_test_dset = MyTinyImagenet(base_path(
            ) + 'TINYIMG', train=False, download=True, transform=SequentialTinyImagenet32R.TEST_TRANSFORM)
        
        else:
            raise NotImplementedError(
                f"Dataset `{self.args.pre_dataset}` not implemented")

        return aux_dset, aux_test_dset

    def mini_eval(self):
        model = self
        tg = model.training
        test_dl = torch.utils.data.DataLoader(
            self.aux_test_dset, batch_size=self.args.batch_size, shuffle=False, num_workers=0)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target, _ in test_dl:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                correct += (torch.argmax(output, dim=1) == target).sum().item()
                total += len(data)
        model.train(tg)
        return correct / total
        
    def load_aux_dataset(self):
        self.aux_dset, self.aux_test_dset = self.get_aux_dataset()

        self.num_aux_classes = self.aux_dset.N_CLASSES
        self.aux_transform = transforms.Compose(
            [transforms.ToPILImage(), self.aux_dset.transform])

        if self.args.pre_dataset != 'imagenet':
            self.aux_dl = torch.utils.data.DataLoader(
                self.aux_dset, batch_size=self.args.minibatch_size, shuffle=True, num_workers=0, drop_last=True)
            self.aux_iter = iter(self.aux_dl)

        else:
            self.aux_dl = None
            self.aux_iter=iter([[torch.randn((1,3,224,224)).to(self.device)]])

    def reset_classifier(self):
        self.net.classifier = torch.nn.Linear(
                self.net.classifier.in_features, self.net.num_classes).to(self.device)
        self.opt = SGD(self.net.parameters(), lr=self.args.lr,
                           weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)

    def load_initial_checkpoint(self, ignore_classifier=False):
        self.aux_dset, self.aux_test_dset = None, None
    
        url, ckpt_name = get_ckpt_remote_url(self.args)
        if self.args.pre_dataset == 'imagenet':
            if not os.path.exists(self.args.load_cp):
                print("Downloading checkpoint file...")
                dn(url, self.args.load_cp)
                print(f"Downloaded in: {self.args.load_cp}")
            self.load_cp(self.args.load_cp, moco=True, ignore_classifier=ignore_classifier)
        else:
            if url is None and self.args.load_cp is None:
                self.aux_dset, self.aux_test_dset = self.get_aux_dataset()
                self.net.classifier = torch.nn.Linear(
                    self.net.classifier.in_features, self.num_aux_classes).to(self.device)

                self.opt = SGD(self.net.parameters(
                ), lr=self.args.lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
                sched = None
                if self.args.pre_dataset.startswith('cub'):
                    sched = torch.optim.lr_scheduler.MultiStepLR(
                        self.opt, milestones=[80, 150, 250], gamma=0.5)
                elif 'tinyimg' in self.args.pre_dataset.lower():
                    sched = torch.optim.lr_scheduler.MultiStepLR(
                        self.opt, milestones=[20, 30, 40, 45], gamma=0.5)

                for e in range(self.args.pre_epochs):
                    for i, (x, y, _) in tqdm(enumerate(self.aux_dl), desc='Pre-training epoch {}'.format(e), leave=False, total=len(self.aux_dl)):
                        y = y.long()
                        self.net.train()
                        self.opt.zero_grad()
                        x = x.to(self.device)
                        y = y.to(self.device)
                        aux_out = self.net(x)
                        aux_loss = self.loss(aux_out, y)
                        aux_loss.backward()
                        self.opt.step()
                        
                    if sched is not None:
                        sched.step()
                    if e % 5 == 4:
                        print(
                            e, f"{self.mini_eval()*100:.2f}%")
                from datetime import datetime
                # savwe the model
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                modelpath = self.NAME + '_' + now + '.pth'
                torch.save(self.net.state_dict(), modelpath)
                print(modelpath)
            else:                    
                if self.args.load_cp is None or not os.path.isfile(self.args.load_cp):
                    self.args.load_cp = self.args.load_cp if self.args.load_cp is not  None else './checkpoints/'

                    print("Downloading checkpoint file...")
                    dn(url, self.args.load_cp)
                    print(f"Downloaded in: {self.args.load_cp}")
                self.load_cp(self.args.load_cp, moco=True, ignore_classifier=ignore_classifier)
                print("Loaded!")

        if self.aux_test_dset is not None:
            pre_acc = self.mini_eval()
            print(f"Pretrain accuracy: {pre_acc:.2f}")

        if self.args.stop_after_prep:
            exit()




    def load_cp(self, cp_path, new_classes=None, moco=False, ignore_classifier=False) -> None:
        """
        Load pretrain checkpoint, optionally ignores and rebuilds final classifier.

        :param cp_path: path to checkpoint
        :param new_classes: ignore and rebuild classifier with size `new_classes`
        :param moco: if True, allow load checkpoint for Moco pretraining
        """
        s = torch.load(cp_path, map_location=self.device)
        if 'state_dict' in s:  # loading moco checkpoint
            if not moco:
                raise Exception(
                    'ERROR: Trying to load a Moco checkpoint without setting moco=True')
            s = {k.replace('encoder_q.', ''): i for k,
                 i in s['state_dict'].items() if 'encoder_q' in k}

        if not ignore_classifier:
            if new_classes is not None:
                self.net.classifier = torch.nn.Linear(
                    self.net.classifier.in_features, self.num_aux_classes).to(self.device)
                for k in list(s):
                    if 'classifier' in k:
                        s.pop(k)
            else:
                cl_weights = [s[k] for k in list(s.keys()) if 'classifier' in k]
                if len(cl_weights) > 0:
                    cl_size = cl_weights[-1].shape[0]
                    self.net.classifier = torch.nn.Linear(
                        self.net.classifier.in_features, cl_size).to(self.device)
        else:
            for k in list(s):
                if 'classifier' in k:
                    s.pop(k)
                    
        for k in list(s):
            if 'net' in k:
                s[k[4:]] = s.pop(k)
        for k in list(s):
            if 'wrappee.' in k:
                s[k.replace('wrappee.', '')] = s.pop(k)
        for k in list(s):
            if '_features' in k:
                s.pop(k)

        try:
            self.net.load_state_dict(s)
        except:
            _, unm = self.net.load_state_dict(s, strict=False)

            if new_classes is not None or ignore_classifier:
                assert all(['classifier' in k for k in unm]
                           ), f"Some of the keys not loaded where not classifier keys: {unm}"
            else:
                assert unm is None, f"Missing keys: {unm}"

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr,
                       weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
        self.device = get_device()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x, **kwargs)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

