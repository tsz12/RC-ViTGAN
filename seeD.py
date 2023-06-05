from argparse import ArgumentParser
from pathlib import Path
import shutil

import imageio
def silence_imageio_warning(*args, **kwargs):
    pass
imageio.core.util._precision_warn = silence_imageio_warning

import gin
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
from torch.utils.data import DataLoader

#from evaluate.gan import FIDScore, FixedSampleGeneration, ImageGrid
from datasets import get_dataset
from augment import get_augment
from utils import cycle

from training.gan import setup
from utils import Logger
from utils import count_parameters
from utils import accumulate
from utils import set_grad

# import for gin binding
import penalty

import wandb
import time

from torchvision import datasets, transforms

#writer就相当于一个日志，保存你要做图的所有信息。第二句就是在你的项目目录下建立一个文件夹log，存放画图用的文件。刚开始的时候是空的
from torchsummary import summary
from vit_generator import vit_my,vit_small,vit_my_8
from mydiscriminator import ResidualDiscriminatorP
#from copy_vit_generator import vit_my


discriminator = ResidualDiscriminatorP(size=128, small32=False,
                                               mlp_linear=True, d_hidden=512)
discriminator.to('cpu')

model_stats = summary(discriminator, [(6, 128,128)],verbose=0)
summary_str = str(model_stats)
print(summary_str)