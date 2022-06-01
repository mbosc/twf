from cv2 import transform
import torch
import torch.nn.functional as F
import numpy as np

from torchvision.transforms import functional as TF
from torchvision import transforms


def normalize(x, mean, std):
    assert len(x.shape) == 4
    return (x - torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)) \
        / torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)


def random_flip(x):
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < 0.5
    x[mask] = x[mask].flip(3)
    return x


def random_grayscale(x, prob=0.2):
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < prob
    x[mask] = (x[mask] * torch.tensor([[0.299, 0.587, 0.114]]).unsqueeze(
        2).unsqueeze(2).to(x.device)).sum(1, keepdim=True).repeat_interleave(3, 1)
    return x


def random_crop(x, padding):
    assert len(x.shape) == 4
    crop_x = torch.randint(-padding, padding, size=(x.shape[0],))
    crop_y = torch.randint(-padding, padding, size=(x.shape[0],))

    crop_x_start, crop_y_start = crop_x + padding, crop_y + padding
    crop_x_end, crop_y_end = crop_x_start + \
        x.shape[-1], crop_y_start + x.shape[-2]

    oboe = F.pad(x, (padding, padding, padding, padding))
    mask_x = torch.arange(
        x.shape[-1] + padding * 2).repeat(x.shape[0], x.shape[-1] + padding * 2, 1)
    mask_y = mask_x.transpose(1, 2)
    mask_x = ((mask_x >= crop_x_start.unsqueeze(1).unsqueeze(2))
              & (mask_x < crop_x_end.unsqueeze(1).unsqueeze(2)))
    mask_y = ((mask_y >= crop_y_start.unsqueeze(1).unsqueeze(2))
              & (mask_y < crop_y_end.unsqueeze(1).unsqueeze(2)))
    return oboe[mask_x.unsqueeze(1).repeat(1, x.shape[1], 1, 1) * mask_y.unsqueeze(1).repeat(1, x.shape[1], 1, 1)].reshape(x.shape[0], 3, x.shape[2], x.shape[3])





class CustomRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    @torch.no_grad()
    def __call__(self, img, other_img=None):
        if np.random.rand() < self.p:
            return TF.hflip(img), [TF.hflip(x.unsqueeze(0)).squeeze(0) for x in other_img]
        return img, other_img


class CustomRandomCrop(object):
    def __init__(self, size, padding=0, resize=False, min_resize_index=None):
        self.size = size
        self.padding = padding
        self.resize = resize
        self.min_resize_index = min_resize_index
        self.transform = transforms.RandomCrop(size, padding)

    @torch.no_grad()
    def __call__(self, img, other_img=None):
        img = TF.pad(img, self.padding)
        i, j, h, w = self.transform.get_params(img, self.size)

        maps = []
        for idx, map in enumerate(other_img):
            m=map.unsqueeze(0)
            orig_size = m.shape[-2:]
            if self.resize:
                if self.min_resize_index is None or idx <= self.min_resize_index:
                    m = TF.resize(m, (int(orig_size[0]*2), int(orig_size[1]*2)), interpolation=transforms.InterpolationMode.NEAREST)

            rate = (self.size[0]//m.shape[-1])
            _i, _j, _h, _w = i//rate, j//rate, h//rate, w//rate
            m = TF.pad(m, self.padding//rate)
            m = TF.crop(m, _i, _j, _h, _w)

            if self.resize:
                if self.min_resize_index is None or idx <= self.min_resize_index:
                    m = TF.resize(m, orig_size, interpolation=transforms.InterpolationMode.NEAREST)

            maps.append(m.squeeze(0))
        return TF.crop(img, i, j, h, w), maps


class DoubleTransform(object):
    def __init__(self, tf):
        self.transform = tf

    @torch.no_grad()
    def __call__(self, img, other_img):
        return self.transform(img), other_img


class DoubleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __iter__(self):
        return iter(self.transforms)

    def __getitem__(self, i):
        return self.transforms[i]

    def __len__(self):
        return len(self.transforms)

    @torch.no_grad()
    def __call__(self, img, other_img):
        other_img = [o.clone() for o in other_img]
        img = img.clone() if isinstance(img, torch.Tensor) else img.copy()
        for t in self.transforms:
            img, other_img = t(img, other_img)
        return img, other_img
