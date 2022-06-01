

from datetime import datetime
from utils.training import evaluate
from torch.optim import SGD

from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets.utils.validation import ValidationDataset
from utils.status import progress_bar
import torch
import numpy as np
import math
from torchvision import transforms


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_aux_dataset_args(parser)
    return parser


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []
        self.current_task = 0

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.load_initial_checkpoint()
            self.reset_classifier()

    def end_task(self, dataset):
        if dataset.SETTING != 'domain-il':
            assert len(dataset.train_loader.dataset.data)
            self.old_data.append(dataset.train_loader.dataset.data)
            self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
            self.current_task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS: return

            # reinit network
            # self.net = dataset.get_backbone()
            # self.net.to(self.device)
            self.net.train()
            self.opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)

            # prepare dataloader
            all_data, all_labels = None, None
            for i in range(len(self.old_data)):
                if all_data is None:
                    all_data = self.old_data[i]
                    all_labels = self.old_labels[i]
                else:
                    all_data = np.concatenate([all_data, self.old_data[i]])
                    all_labels = np.concatenate([all_labels, self.old_labels[i]])

            transform = dataset.TRANSFORM if dataset.TRANSFORM is not None else transforms.ToTensor()
            # temp_dataset = ValidationDataset(all_data, all_labels, transform=transform)
            dataset.train_loader.dataset.data = all_data
            dataset.train_loader.dataset.targets = all_labels
            loader = torch.utils.data.DataLoader(dataset.train_loader.dataset, batch_size=self.args.batch_size, shuffle=True)

            if self.args.scheduler is not None:
                self.opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
                if self.args.scheduler == 'simple':
                    assert self.args.scheduler_rate is not None
                    # if args.n_epochs == 50:
                    step = None
                    if self.args.opt_steps is not None:
                        step = self.args.opt_steps
                        print('steps', step)
                    elif self.args.dataset == 'seq-cifar100':
                        step = [35, 45]
                    

                    scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, step, gamma=self.args.scheduler_rate, verbose=True)
                
            
            # from datetime import datetime
            # # now = datetime.now()
            
            print(f'\nmean acc bf training:',np.mean(evaluate(self, dataset)[0]), '\n')
            self.opt.zero_grad()
            for e in range(self.args.n_epochs):
                for i, batch in enumerate(loader):
                    inputs, labels, _ = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    # if i % 3 == 0:
                    self.opt.step()
                    self.opt.zero_grad()
                    progress_bar(i, len(loader), e, 'J', loss.item())

                    # with open(f'logs/{now}.txt', 'a') as f:
                    #     f.write(f'{loss.item()}\n')

                self.opt.step()
                self.opt.zero_grad()
                if self.args.scheduler is not None:
                    scheduler.step()
                if e < 5 or e % 5 == 0:
                    print(f'\nmean acc at e {e}:',np.mean(evaluate(self, dataset)[0]), '\n')
            # print(f"\nTotal training time {datetime.now() - now}\n")
                

        
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        return 0,0,0,0,0
