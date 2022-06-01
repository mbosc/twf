import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        self.num_classes = get_dataset(args).N_TASKS * self.cpt
        self.eye = torch.tril(torch.ones((self.num_classes, self.num_classes))).bool().to(self.device)
        self.eye = torch.tril(self.eye.T, self.cpt - 1)

        self.current_task = 0

    def end_task(self, dataset):
        self.current_task += 1

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.load_initial_checkpoint()
            self.reset_classifier()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        real_batch_size = inputs.shape[0]
        mask = self.eye[(labels // self.cpt) * self.cpt]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        losses = self.loss(outputs, labels, reduction='none')
        loss_stream = losses[:real_batch_size].mean().item()
        loss_buffer = losses[real_batch_size:].mean().item()
        loss = losses.mean()

        with torch.no_grad():
          mask = self.eye[(labels // self.cpt) * self.cpt]
          lossesM = self.loss(outputs[mask].reshape(inputs.shape[0], -1), labels % self.cpt, reduction='none')
          loss_streamM = lossesM[:real_batch_size].mean().item()
          loss_bufferM = lossesM[real_batch_size:].mean().item()

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item(), loss_streamM, loss_bufferM, loss_stream, loss_buffer
