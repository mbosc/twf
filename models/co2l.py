

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.buffer import Buffer
from tqdm import tqdm
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils.simclrloss import SupConLoss
import math
from torchvision import transforms
from copy import deepcopy

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)
    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--current_temp', type=float, default=0.2)
    parser.add_argument('--past_temp', type=float, default=0.01)
    parser.add_argument('--distill_power', type=float, default=1.0)

    parser.add_argument('--linear_lr', type=float, default=1)
    parser.add_argument('--linear_lr_decay', type=float, default=0.2)
    parser.add_argument('--linear_lr_decay_steps', type=int, nargs='+', default=[60,75,90])
    parser.add_argument('--co2l_task_epoch', type=int, default=100)

    return parser

class TwinTransform():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return torch.stack([self.transform(x), self.transform(x)], dim=0)

class Co2l(ContinualModel):
    NAME = 'co2l'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        self.num_classes = get_dataset(args).N_TASKS * self.cpt
        self.supcon_loss = SupConLoss(temperature=args.temp, device=self.device)
        self.twin_transform = TwinTransform(self.transform)
        self.current_task = 0

        self.args.valset_split = 0

        # pimp my classifier
        self.inf, self.outf = self.net.classifier.in_features, 128     
        self.net.classifier = nn.Sequential(nn.Linear(self.inf, self.inf), nn.ReLU(), nn.Linear(self.inf, self.outf)).to(self.device)
        self.net.classifier.in_features = self.outf
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr,
                       weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)

    def fit_linear_classifier(self):
        self.net.eval()
        self.classifier = nn.Linear(self.inf, self.num_classes).to(self.device)
        class_opt = torch.optim.SGD(self.classifier.parameters(), lr=self.args.linear_lr, momentum=0.9)
        sched = torch.optim.lr_scheduler.MultiStepLR(class_opt, milestones=self.args.linear_lr_decay_steps, gamma=self.args.linear_lr_decay)

        for epoch in tqdm(range(1), desc='Fitting linear classifier'):
            all_inputs, all_labels = self.buffer.get_data(
                len(self.buffer.examples), transform=self.twin_transform.transform)

            while len(all_inputs):
                data, target = all_inputs[:self.args.batch_size], all_labels[:self.args.batch_size]
                all_inputs, all_labels = all_inputs[self.args.batch_size:], all_labels[self.args.batch_size:]
                class_opt.zero_grad()
                with torch.no_grad():
                    feats = self.net(data, returnt='features')
                output = self.classifier(feats)
                loss = F.cross_entropy(output, target)
                loss.backward()
                class_opt.step()
            sched.step()

    def forward(self, x):
        if not hasattr(self, "classifier"):
            self.classifier = nn.Linear(self.inf, self.outf).to(self.device)
        return self.classifier(self.net(x, returnt='features'))

    def baguette_replay(self, dataset):
        
        if self.current_task > 0:
            buff_val_mask = torch.rand(len(self.buffer)) < self.args.valset_split
            train_val_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
            train_val_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

            self.val_loader = deepcopy(dataset.train_loader)
            
            # REDUCE AND MERGE TRAINING SET
            dataset.train_loader.dataset.targets = np.concatenate([
                    dataset.train_loader.dataset.targets[~train_val_mask],
                    self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
                 ])
            if type(dataset.train_loader.dataset.data) == torch.Tensor:
                dataset.train_loader.dataset.data = torch.cat([
                    dataset.train_loader.dataset.data[~train_val_mask],
                    (self.buffer.examples * 255).type(torch.uint8).cpu()[:len(self.buffer)][~buff_val_mask]
                    ])
            else:
                
                dataset.train_loader.dataset.data = np.concatenate([
                    dataset.train_loader.dataset.data[~train_val_mask],
                    (self.buffer.examples * 255).type(torch.uint8).cpu()[:len(self.buffer)][~buff_val_mask].permute(0, 2, 3, 1).numpy()
                    ])
                

            # REDUCE AND MERGE VALIDATION SET
            self.val_loader.dataset.targets = np.concatenate([
                    self.val_loader.dataset.targets[train_val_mask],
                    self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
                 ])
            if type(self.val_loader.dataset.data) == torch.Tensor:
                self.val_loader.dataset.data = torch.cat([
                    self.val_loader.dataset.data[train_val_mask],
                    (self.buffer.examples * 255).type(torch.uint8).cpu()[:len(self.buffer)][buff_val_mask]
                    ])
            else:
                
                
                self.val_loader.dataset.data = np.concatenate([
                    self.val_loader.dataset.data[train_val_mask],
                    (self.buffer.examples).cpu()[:len(self.buffer)][buff_val_mask].numpy()
                    ])

    def end_task(self, dataset):
        # # restore normal transform
        # dataset.train_loader.dataset.transform = dataset.train_loader.dataset.transform.transform
        # class-stratify buffer
        if not hasattr(self, "bup_dataset"):
            self.bup_dataset = deepcopy(dataset)
        self.buffer.class_stratified_add_data(self.bup_dataset, self.cpt, self.net, desired_attrs=['examples', 'labels'])

        self.fit_linear_classifier()

        # deepcopy the model
        self.prev_model = deepcopy(self.net)
        self.prev_model.eval()

        self.current_task += 1

    def begin_task(self, dataset):
        if self.current_task == 0 or ("start_from" in self.args and self.args.start_from is not None and self.task == self.args.start_from):
            if 'cub' in self.args.dataset:
                self.vbsteps = 8
            self.args.cur_epochs = self.args.n_epochs
            self.load_aux_dataset()
            self.load_initial_checkpoint(ignore_classifier=True)
        else:
            self.args.cur_epochs = self.args.co2l_task_epoch

        # inject double aug in dataset
        self.bup_dataset = deepcopy(dataset)

        if self.current_task > 0:
            self.baguette_replay(dataset)
        
        train_transform = dataset.train_loader.dataset.transform

        dataset.train_loader.dataset.transform = TwinTransform(train_transform)

        self.total_batches = len(dataset.train_loader)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr,
                       weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)

    def warmup(self, epoch, opt):
        if not hasattr(self, 'batch_id'):
            self.batch_id = 0
            self.last_epoch_wu = epoch
        if self.last_epoch_wu > epoch:
            self.batch_id = 0

        
        warmup_from = 0.01
        warm_epochs = 10
        lr_decay_rate = 0.1
        
        eta_min = self.args.lr * (lr_decay_rate ** 3)
        warmup_to = eta_min + (self.args.lr - eta_min) * (
                    1 + math.cos(math.pi * warm_epochs / self.args.cur_epochs)) / 2
        
        if epoch <= warm_epochs:
            p = (self.batch_id + (epoch - 1) * self.total_batches) / \
                (warm_epochs * self.total_batches)
            lr = warmup_from + p * (warmup_to - warmup_from)

            for param_group in opt.param_groups:
                param_group['lr'] = lr


        self.batch_id+=1
        self.last_epoch_wu = epoch
        
        
    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.args.lr
        lr_decay_rate = 0.1
        
        eta_min = lr * (lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / self.args.cur_epochs)) / 2
    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        if hasattr(self, 'vbsteps'):
            for step in range(self.vbsteps):
                s_s, s_e = (len(inputs) // self.vbsteps) * step, (len(inputs) // self.vbsteps) * (step + 1)
                loss = self._observe(inputs[s_s: s_e], labels[s_s: s_e], not_aug_inputs[s_s: s_e], epoch=epoch)
        else:
            loss = self._observe(inputs, labels, not_aug_inputs, epoch=epoch)
        self.opt.step()
        return loss


    def _observe(self, inputs, labels, not_aug_inputs, epoch=None):

        str_batch_size = inputs.shape[0]
        inputs = torch.cat((inputs[:, 0], inputs[:, 1]), dim=0)


        all_batch_size = str_batch_size

        outputs = F.normalize(self.net(inputs), dim=1) # inputs[0]

        # IRD (current)
        if self.current_task > 0:
            features1_prev_task = outputs

            features1_sim = torch.div(torch.matmul(
                features1_prev_task, features1_prev_task.T), self.args.current_temp)
            logits_mask = torch.scatter(
                torch.ones_like(features1_sim),
                1,
                torch.arange(features1_sim.size(0)).view(-1,
                                                         1).cuda(non_blocking=True),
                0
            )
            logits_max1, _ = torch.max(
                features1_sim * logits_mask, dim=1, keepdim=True)
            features1_sim = features1_sim - logits_max1.detach()
            row_size = features1_sim.size(0)
            logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

        # Asym SupCon
        f1, f2 = torch.split(outputs, [all_batch_size, all_batch_size], dim=0)
        outputs =  torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.supcon_loss(outputs, labels, target_labels=list(
            range(self.current_task*self.cpt, (self.current_task+1)*self.cpt)))

        # IRD (past)
        if self.current_task > 0:
            with torch.no_grad():
                features2_prev_task = F.normalize(self.prev_model(inputs), dim=1)

                features2_sim = torch.div(torch.matmul(
                    features2_prev_task, features2_prev_task.T), self.args.past_temp)
                logits_max2, _ = torch.max(
                    features2_sim*logits_mask, dim=1, keepdim=True)
                features2_sim = features2_sim - logits_max2.detach()
                logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                    features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

            loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
            loss += self.args.distill_power * loss_distill

        loss.backward()
        

        
        return loss.item(), 0, 0, 0, 0
