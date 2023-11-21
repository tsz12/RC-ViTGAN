# The code is based on code publicly available at
#   https://github.com/rosinality/stylegan2-pytorch
# written by Seonghyeon Kim.

import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from stylegan2.op import upfirdn2d, fused_leaky_relu, conv2d_gradfix
from stylegan2.op import FusedLeakyReLU

class ConLinear(nn.Module):
    def __init__(self, ch_in, ch_out, is_first=False, bias=True):
        super(ConLinear, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=bias)
        if is_first:
            nn.init.uniform_(self.conv.weight, -np.sqrt(9 / ch_in), np.sqrt(9 / ch_in))
        else:
            nn.init.uniform_(self.conv.weight, -np.sqrt(3 / ch_in), np.sqrt(3 / ch_in))

    def forward(self, x):
        return self.conv(x)


class SinActivation(nn.Module):
    def __init__(self,):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class LFF(nn.Module):
    def __init__(self, hidden_size, ):
        super(LFF, self).__init__()
        self.ffm = ConLinear(2, hidden_size, is_first=True)
        self.activation = SinActivation()

    def forward(self, x):
        x = self.ffm(x)
        x = self.activation(x)
        return x
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias_init=0, lr_mul=1, activation=None, use_bias=True,modulated=False
    ):
        super().__init__()
        self.modulated=modulated
        #print(f"modulated为{modulated}")
        if not modulated:
            self.weight=nn.Parameter(torch.zeros(out_dim,in_dim), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(out_dim), requires_grad=False)
        else:
            #print(1)
            self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(out_dim))
            else:
                self.bias = nn.Parameter(torch.zeros(out_dim), requires_grad=False)
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.bias_init = bias_init

        if activation == 'fused_lrelu':
            self.activation = activation
        else:
            self.activation = None
            


    def forward(self, input):
        # print(f"input为{input.shape}")
        # print(f"self.weight为{self.weight.shape}")
        # print(f"self.scale为{self.scale}")
        if not self.modulated:
            out = F.linear(input, self.weight,self.bias)
        else:
            bias = self.bias * self.lr_mul + self.bias_init
            # print(f"bias为{bias.shape}")
            if self.activation:
                out = F.linear(input, self.weight * self.scale)
                out = fused_leaky_relu(out, bias)
            else:
                out = F.linear(input, self.weight * self.scale, bias=bias)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ConvLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, blur_kernel=[1, 3, 3, 1],
                 downsample=False, activate=True):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(EqualConv2d(in_channel, out_channel, kernel_size,
                                  padding=self.padding, stride=stride, bias=False))

        if activate:
            layers.append(FusedLeakyReLU(out_channel))

        super().__init__(*layers)

