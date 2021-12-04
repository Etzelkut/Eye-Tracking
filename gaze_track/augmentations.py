from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from basem.basic_dependency import *

import torchvision.transforms.functional as TF
from torchvision import transforms


class Rescale(nn.Module):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, sample, output_size):
        image, landmarks = sample['img'], sample['landmarks']

        h, w = image.shape[-2:]
        new_h, new_w = output_size

        image = TF.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_h / h, new_w / w]
        
        sample['img'], sample['landmarks'] = image, landmarks

        return sample



class RandomCrop(nn.Module):

  def __init__(self, output_size):
    super().__init__()
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
        self.output_size = (output_size, output_size)
    else:
        assert len(output_size) == 2
        self.output_size = output_size

  @torch.no_grad()
  def forward(self, sample):
    image, landmarks = sample['img'], sample['landmarks']

    h, w = image.shape[-2:]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    image = image[:, top: top + new_h,
                  left: left + new_w]

    landmarks = landmarks - [top, left]

    sample['img'], sample['landmarks'] = image, landmarks
    return sample


class CenterCrop(nn.Module):

  def __init__(self, output_size):
    super().__init__()
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
        self.output_size = (output_size, output_size)
    else:
        assert len(output_size) == 2
        self.output_size = output_size

  @torch.no_grad()  # disable gradients for effiency
  def forward(self, sample):
    image, landmarks = sample['img'], sample['landmarks']

    h, w = image.shape[-2:]
    new_h, new_w = self.output_size

    top =  int((h - new_h)/2)
    left =  int((w - new_w)/2)

    image = image[:, top: top + new_h, left: left + new_w]

    landmarks = landmarks - [top, left]

    sample['img'], sample['landmarks'] = image, landmarks
    return sample


class Preprocess(nn.Module):
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x_out = TF.to_tensor(x) # CxHxW
        return x_out.float() / 255.0


class DataAugmentationImage(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, max_epochs):
      super().__init__()
      self.max_epochs = max_epochs
      self.jitter = transforms.ColorJitter(0.25, 0.25, 0.25, 0.25)
      self.blur = transforms.GaussianBlur(kernel_size=17, sigma=(0.2, 2))
      self.sharp = transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.25)
      self.sharpblur = transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.1)
      
      self.std = 1.0
      self.mean = 0

      self.normalize = transforms.Normalize([0.5], [0.5])

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x, epoch):
      p = random.random()
      if p < (epoch/self.max_epochs):
        x = self.jitter(x)
        x = self.blur(x)
        x = self.sharp(x)
        x = self.sharpblur(x)
        p = random.random()
        if p < 0.5:
          x = x + torch.randn(x.size(), device = x.get_device(), dtype=x.dtype) * self.std + self.mean
      
      x = self.normalize(x)
      return x