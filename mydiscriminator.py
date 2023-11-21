# discriminator model for adversarial training
import collections
import pathlib
import random
import pickle
from typing import Dict, Tuple, Sequence

import cv2
from skimage.color import rgb2lab, lab2rgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
#----------------pix2pix用的Discriminator-------------------------
class Pix2PixDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Pix2PixDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
    
# The code is based on code publicly available at
#   https://github.com/rosinality/stylegan2-pytorch
# written by Seonghyeon Kim.

#----------------stylegan2用的Discriminator-------------------------
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.gan.stylegan2.layers import ConvLayer, EqualLinear
from models.gan.stylegan2.layers import Downsample

from models.gan.base import BaseDiscriminator


class FromRGB(ConvLayer):
    def __init__(self, in_channel,out_channel):
        super(FromRGB, self).__init__(in_channel, out_channel, 1, activate=True)


def _minibatch_stddev_layer(input, stddev_group=4, stddev_feat=1):
    batch, channel, height, width = input.shape
    if batch%2==0:
        batch=int(batch/2)
    group = min(batch, stddev_group)
    #print(f"_minibatch_stddev_layer里input的大小为{input.shape}")#([2, 512, 4, 4])
    stddev = input.view(
        group, -1, stddev_feat, channel // stddev_feat, height, width
    )#4,-1
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True)
    stddev = stddev.mean(2)
    stddev = stddev.repeat(group, 1, height, width)

    return torch.cat([input, stddev], 1)


class SkipBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.from_rgb = FromRGB(in_channel=6,out_channel=in_channel)
        self.conv1 = ConvLayer(in_channel, in_channel, 3, activate=True)
        self.conv2 = ConvLayer(in_channel, out_channel, 3,
                               blur_kernel=blur_kernel, downsample=True, activate=True)
        self.downsample = Downsample(blur_kernel)

    def forward(self, input, features=None):
        output = self.from_rgb(input)

        if features is not None:
            features = output + features
        else:
            features = output

        features = self.conv1(features)
        features = self.conv2(features)
        input = self.downsample(input)

        return input, features


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        #print(f"当前ResBlock的in_channel为{in_channel},当前ResBlock的out_channel为{out_channel}")
        self.conv1 = ConvLayer(in_channel, in_channel, 3, activate=True)
        self.conv2 = ConvLayer(in_channel, out_channel, 3,
                               blur_kernel=blur_kernel, downsample=True, activate=True)
        self.skip = ConvLayer(in_channel, out_channel, 1,
                              blur_kernel=blur_kernel, downsample=True, activate=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class ResidualDiscriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], small32=False):
        super().__init__()

        if small32:
            channels = {
                4: 512,
                8: 512,
                16: 256,
                32: 128,
            }
        else:
            channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: int(256 * channel_multiplier),
                128: int(128 * channel_multiplier),
                256: int(64 * channel_multiplier),
                512: int(32 * channel_multiplier),
                1024: int(16 * channel_multiplier),
            }
        #size=128
        layers = [FromRGB(in_channel=6,out_channel=channels[size])]
        log_size = int(math.log(size, 2))#7
        in_channel = channels[size]

        for i in range(log_size, 2, -1):#从7到3的整数
            out_channel = channels[2 ** (i - 1)]#(512,512,512,512,512)
            layers.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.layers = nn.Sequential(*layers)

        self.last_conv = ConvLayer(in_channel + 1, channels[4], 3)#(513,512)
        self.last_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        input = input * 2. - 1.
        out = self.layers(input)

        out = _minibatch_stddev_layer(out)
        out = self.last_conv(out)
        out = out.view(out.size(0), -1)
        out = self.last_linear(out)

        return out


class SkipDiscriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], small32=False):
        super().__init__()

        if small32:
            channels = {
                4: 512,
                8: 512,
                16: 256,
                32: 128,
            }
        else:
            channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: int(256 * channel_multiplier),
                128: int(128 * channel_multiplier),
                256: int(64 * channel_multiplier),
                512: int(32 * channel_multiplier),
                1024: int(16 * channel_multiplier),
            }
        self.stddev_group = 4
        self.stddev_feat = 1

        self.layers = nn.ModuleList()
        log_size = int(math.log(size, 2))
        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.layers.append(SkipBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.last_rgb = FromRGB(channels[4])
        self.last_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.last_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        input = input * 2. - 1.
        features = None

        for layer in self.layers:
            input, features = layer(input, features)

        output = self.last_rgb(input)
        features = output + features

        features = _minibatch_stddev_layer(features)
        features = self.last_conv(features)
        features = features.view(features.size(0), -1)

        d = self.last_linear(features)
        return d

class ResidualDiscriminatorP(BaseDiscriminator):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], small32=False,input_channel=6, **kwargs):
        if small32:
            channels = {
                4: 512,
                8: 512,
                16: 256,
                32: 128,
            }
        else:
            channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: int(256 * channel_multiplier),
                128: int(128 * channel_multiplier),
                256: int(64 * channel_multiplier),
                512: int(32 * channel_multiplier),
                1024: int(16 * channel_multiplier),
            }
        self.n_features = channels[4] * 4 * 4
        #self.n_features = channels[4] * 4 * 4 * 2
        super().__init__(self.n_features, n_classes=1, **kwargs)

        self.mid=self.conv2d = nn.Conv2d(in_channels=6,out_channels=3,kernel_size=1)

        #size=128
        layers = [FromRGB(in_channel=input_channel,out_channel=channels[size])]
        log_size = int(math.log(size, 2))#7
        in_channel = channels[size]#256

        for i in range(log_size, 2, -1):#7-3
            out_channel = channels[2 ** (i - 1)]#512,512,512,512,512
            layers.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.layers = nn.Sequential(*layers)
        self.last_conv = ConvLayer(in_channel + 1, channels[4], 3)

    def penultimate(self, input):
        #input=self.mid(input)
        #print(f"input的shape为{input.shape}")
        input = input * 2. - 1.
        #print(f"input的shape为{input.shape}")
        out = self.layers(input)
        #print(f"out的shape为{out.shape}")
        out = _minibatch_stddev_layer(out)
        #print(f"out的shape为{out.shape}")
        out = self.last_conv(out)
        #print(f"out的shape为{out.shape}")
        out = out.view(out.size(0), -1)
        #print(f"out的shape为{out.shape}")

        return out
