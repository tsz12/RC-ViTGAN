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
#from vit_generator import vit_my,vit_small,vit_my_8
from vit_generator_skip import vit_my,vit_small,vit_my_8
from models.gan.stylegan2.generator_wovit import Generator
from mydiscriminator import ResidualDiscriminatorP
import torch.nn as nn
#from copy_vit_generator import vit_my

generator = vit_my_8(patch_size=16,noise=True)
#generator=Generator(size=128,style_dim=384)
# discriminator_pair = ResidualDiscriminatorP(size=128, small32=False,
#                                                mlp_linear=True, d_hidden=512)
# discriminator_single = ResidualDiscriminatorP(size=128, small32=False,
#                                                mlp_linear=True, d_hidden=512,input_channel=3)
#generator=nn.DataParallel(generator)
#print([x for x in generator.named_children()])
# generator=nn.DataParallel(generator)
# print(generator.module.sample_latent)
generator.to('cpu')
#discriminator_pair.to('cpu')

model_stats = summary(generator, [(3, 224,224),(384,),(128,128)],verbose=0)
#model_stats = summary(generator, [(384,)],verbose=0)
summary_str = str(model_stats)
print(summary_str)

# model_stats = summary(discriminator_pair, [(6, 128,128)],verbose=0)
# summary_str = str(model_stats)
# print(summary_str)

# model_stats = summary(discriminator_single, [(3, 128,128)],verbose=0)
# summary_str = str(model_stats)
# print(summary_str)

# from tensorboardX import SummaryWriter
# tb_writer = SummaryWriter(log_dir="tb")
# generator = vit_small(patch_size=8).cuda()
# x=torch.rand(4,3,128,128).cuda()
# y=torch.rand(4,384).cuda()
# with SummaryWriter(comment='vit_generator') as w:
#     w.add_graph(generator,[x,y])
# # x=torch.randn(1,3,128,128)
# # y=torch.randn(1,384)
# # tb_writer.add_graph(generator, [x,y],verbose=True)
# # tb_writer.close()
# import torch
# from torchviz import make_dot
# x=torch.rand(1,3,128,128)
# y=torch.rand(1,384)
# generator = vit_small(patch_size=8)
# z=generator(x,y)
# g = make_dot(z)
# g.view()