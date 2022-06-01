

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import Callable, List, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, kernel_size=3, groups: int = 1, dilation: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride if kernel_size == 3 else 2,
                     padding=dilation if kernel_size == 3 else 3, bias=False, dilation=dilation, groups=groups)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.return_prerelu = False
        self.conv1 = conv3x3(in_planes=in_planes, out_planes=planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        if self.return_prerelu:
            self.prerelu = out.clone()

        out = relu(out)
        return out

class Bottleneck(nn.Module):
    "Resnet v1.5 bottleneck"

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.return_prerelu = False
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(in_planes=width, out_planes=width, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.stride = stride
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        
        if self.return_prerelu:
            self.prerelu = out.clone()

        out = relu(out)

        return out

class ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def expand_classifier(self, n_classes):
        self.classifier = nn.Linear(self.classifier.in_features, n_classes)

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, first_kernel_size=3, hookme=False) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.return_prerelu = False
        self.block = block
        self.num_blocks = num_blocks
        self.hookme = hookme
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(in_planes=3, out_planes=nf * 1, kernel_size=first_kernel_size)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        if first_kernel_size != 3:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.classifier = nn.Linear(nf * 8 * block.expansion, num_classes)

        self._features = lambda: exec(
            'raise NotImplementedError("Deprecated: use forward with returnt=\'features\'")')

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def set_return_prerelu(self, enable=True):
        self.return_prerelu = enable
        for c in self.modules():
            if isinstance(c, self.block):
                c.return_prerelu = enable

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out_0 = self.bn1(self.conv1(x))
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)
        out_1 = self.layer1(out_0)  # 64, 32, 32
        out_2 = self.layer2(out_1)  # 128, 16, 16
        out_3 = self.layer3(out_2)  # 256, 8, 8
        out_4 = self.layer4(out_3)  # 512, 4, 4

        if self.hookme:
            out_4.register_hook(self.activations_hook)

        feature = avg_pool2d(out_4, out_4.shape[2])  # 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        if returnt == 'features':
            return feature

        out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'full':
            return out, [
                out_0 if not self.return_prerelu else out_0_t,
                out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out_4 if not self.return_prerelu else self.layer4[-1].prerelu,
                out
            ]
        else:
            return (out, feature)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        return self.forward(x, returnt='features')

    def get_params(self, discard_classifier=False) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for kk, pp in list(self.named_parameters()):
            if not discard_classifier or not 'classifier' in kk:
                params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor, discard_classifier=False) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params(discard_classifier).size()
        progress = 0
        for pp in list(self.parameters() if not discard_classifier else self._features.parameters()):
            cand_params = new_params[progress: progress +
                                     torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def set_grads(self, new_grads: torch.Tensor, discard_classifier=False) -> torch.Tensor:
        """
        Sets all the gradients concatenated in a single tensor.
        :param new_grads: concatenated values to be set (??)
        """
        assert new_grads.size() == self.get_grads(discard_classifier).size()
        progress = 0
        for pp in list(self.parameters() if not discard_classifier else self._features.parameters()):
            cand_grads = new_grads[progress: progress +
                                   torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.grad = cand_grads

    def get_grads(self, discard_classifier=False) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for kk, pp in list(self.named_parameters()):
            if not discard_classifier or not 'classifier' in kk:
                grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    def set_grad_filter(self, filter_s: str, enable: bool) -> None:
        negative_mode = filter_s[0] == '~'
        if negative_mode:
            filter_s = filter_s[1:]
            for _, p in filter(lambda x: filter_s not in x[0], self.named_parameters()):
                p.requires_grad = enable
        else:
            for _, p in filter(lambda x: filter_s in x[0], self.named_parameters()):
                p.requires_grad = enable


def resnet18(nclasses: int, nf: int = 64, first_k=3, hookme=False) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, first_kernel_size=first_k, hookme=hookme)


def resnet34(nclasses: int, nf: int = 64) -> ResNet:
    """
    Instantiates a ResNet34 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf)