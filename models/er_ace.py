

import torch
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='ER-ACE with future not fixed (as made by authors)')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)
    return parser


class ErACE(ContinualModel):
    NAME = 'er_ace'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErACE, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0

    def begin_task(self, dataset):
        if self.task == 0:
            self.load_initial_checkpoint()
            self.reset_classifier()

    def end_task(self, dataset):
        self.task += 1
        
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        labels = labels.long()
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.)

        if self.task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            loss_re = self.loss(self.net(buf_inputs), buf_labels)

        loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return loss.item(), 0, 0, 0, 0