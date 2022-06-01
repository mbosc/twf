

from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_aux_dataset_args(parser)
    return parser


class Sgd(ContinualModel):
    NAME = 'sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Sgd, self).__init__(backbone, loss, args, transform)
        self.current_task = 0

    def end_task(self, dataset):
        self.current_task += 1

    def begin_task(self, dataset):
        if self.current_task == 0: 
            self.load_initial_checkpoint()
            self.reset_classifier()


    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        labels = labels.long()
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item(),0,0,0,0