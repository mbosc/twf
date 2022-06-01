

from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import datasets
import numpy as np
import socket
import torch
import os
class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass


def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset, class_order=None) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """



    

    if class_order is not None:
        train_dataset.targets = class_order[np.array(train_dataset.targets)]
        test_dataset.targets = class_order[np.array(test_dataset.targets)]

    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
        np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    if not hasattr(train_dataset, 'lazy_load') or not train_dataset.lazy_load:
        train_dataset.data = train_dataset.data[train_mask]
    if not hasattr(test_dataset, 'lazy_load') or not test_dataset.lazy_load:
        test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(train_dataset,
                            batch_size=setting.args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader
