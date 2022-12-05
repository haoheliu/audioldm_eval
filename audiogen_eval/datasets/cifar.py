import sys
from contextlib import redirect_stdout

import torch
import torchvision
from PIL import Image
from torchvision.datasets import CIFAR10
from train import instantiate_from_config


class TransformPILtoRGBTensor:
    def __call__(self, img):
        assert type(img) is Image.Image, "Input is not a PIL.Image"
        assert isinstance(img, Image.Image), "Input is not a PIL.Image"
        width, height = img.size
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(
            height, width, 3
        )
        img = img.permute(2, 0, 1)
        return img


class Cifar10_RGB(CIFAR10):
    def __init__(self, transforms_cfg, cifar_cfg):
        with redirect_stdout(sys.stderr):
            super().__init__(**cifar_cfg)
        self.transform = torchvision.transforms.Compose(
            [instantiate_from_config(c) for c in transforms_cfg]
        )

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


# class ImagesPathDataset(Dataset):
#     def __init__(self, files, transforms=None):
#         self.files = files
#         self.transforms = TransformPILtoRGBTensor() if transforms is None else transforms

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, i):
#         path = self.files[i]
#         img = Image.open(path).convert('RGB')
#         img = self.transforms(img)
#         return img
