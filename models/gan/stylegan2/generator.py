# The code is based on code publicly available at
#   https://github.com/rosinality/stylegan2-pytorch
# written by Seonghyeon Kim.

import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from models.gan.stylegan2.op import FusedLeakyReLU, conv2d_gradfix
from models.gan.stylegan2.layers import PixelNorm, Upsample, Blur
from models.gan.stylegan2.layers import EqualLinear


class ModulatedConv2d(nn.Module):#在Style里使用，应该就是这里出错的，把这里看懂再去那里看看
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 demodulate=True, upsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )#随机初始化卷积核，并将其参数化
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        #print(f"input的大小为{input.shape}")
        #print(f"style的大小为{style.shape}")
        #print(f"调制后的style大小为{self.modulation(style).shape}")
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)#!!style调制层出错
        weight = self.scale * self.weight * style#用style对权重（卷积核的参数）进行调制

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        #input = input.view(1, batch * in_channel, height, width)
        input = input.reshape(1, batch * in_channel, height, width)

        if self.upsample:
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            #out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            out = conv2d_gradfix.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)

            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        else:
            #out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            out = conv2d_gradfix.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.const.repeat(batch, 1, 1, 1)
        return out


class StyleLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True):#blur_kernel是用于图像增强的参数
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim,
                                    upsample=upsample, blur_kernel=blur_kernel,
                                    demodulate=demodulate)
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out


class  ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(self, size,
                 style_dim=512, n_mlp=8, channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, small32=False):
        super().__init__()
        self.size = size#生成图片的大小
        self.style_dim = style_dim#隐层码维度
        #n_mlp:从z到w的Mapping Network网络层数。
        #示例：decoder = nn.DataParallel(Generator(1024, 512, 8))
        #self.style是由PixelNorm和8个EqualLinear层组成的MLP（也就是将噪声z映射为隐层码w的网络）。
        layers = [PixelNorm()]#出自ProgressiveGAN，为了避免幅度失控，在每个卷积层后将每个像素的特征向量归一到单位长度
        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim,
                                      lr_mul=lr_mlp,
                                      activation='fused_lrelu'))
        #EqualLinear本质还是nn.functional.linear,对weight和bias做了一些缩放,这同样出自ProgressiveGAN，weight从标准正态分布随机采样，而将何凯明初始化放到之后动态地进行，这对RMSProp、Adam等优化方式有帮助，保证所有的weight都是一样的学习速度。
        #分别定义每一层，用Sequential组成一个模块
        #self.style是Mapping network
        self.style = nn.Sequential(*layers)

        #self.channels是各分辨率对应卷积层的输出维度列表。
        if small32:
            self.channels = {
                4: 512,
                8: 512,
                16: 256,
                32: 128,
            }
        else:
            self.channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: int(256 * channel_multiplier),
                128: int(128 * channel_multiplier),#此时的self.channels应该是256
                256: int(64 * channel_multiplier),
                512: int(32 * channel_multiplier),
                1024: int(16 * channel_multiplier),
            }
        #！把self.channels[4]里的4改成256试试看
        #self.input是主支的常量输入
        self.input = ConstantInput(self.channels[4])#return一个正态分布采样、参数化的tensor，维度为（batchsize，self.channels[4]，4，4）

        #self.conv1和self.to_rgb1分别是第一个卷积层和第一个to_rgb层，也就是对常量输入进行卷积和to_rgb操作
        #StyleGAN2的GeneratorBlock是由两个Style Block+一个To RGB Block组成
        self.conv1 = StyleLayer(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        #StyleLayer=ModulatedConv+NoiseInjection ！！我觉得可以把NoiseInjection去掉
        #ModulatedConv对于卷积核权重，先从标准正态分布采样、参数化，再在forward过程中通过缩放进行调整。而后按照前述原理，将隐层码映射为style，再对卷积核进行调制解调。
        #NoiseInjection向特征图加噪，image + self.weight * noise。其中self.weight初始为[0]，可学习

        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)
        #将输入feature map的channel变为3（即rgb三通道图），kernel_size=1x1，不变空间维度
        #如果要跳连，则先上采样再residual
        #(batch, 3, h, w)

        #self.num_layers表示主支除了上面的对常量输入的第一个卷积层外，还有多少层。
        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        #用这些组件来组合generator.py
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()


        in_channel = self.channels[4]
        for i in range(3, self.log_size + 1):
            #每一个generator block都是由两个styleconv block（一个变通道且上采样，另一个并不）和一个toRGB block组成
            out_channel = self.channels[2 ** i]
            # 取列表中设好的对应分辨率的channel数

            self.layers.append(
                StyleLayer(in_channel, out_channel, 3, style_dim,
                           upsample=True, blur_kernel=blur_kernel)
            )
            self.layers.append(
                StyleLayer(out_channel, out_channel, 3, style_dim,
                           blur_kernel=blur_kernel)
            )
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel

        #self.n_latent=18意味着共18层卷积层需要18个latent code w去分别调制每一层的卷积核。
        self.n_latent = self.log_size * 2 - 2

    @property
    def device(self):
        return self.input.const.device

    def make_noise(self):
        noises = []
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            noises.append(torch.randn(*shape, device=self.device))
        return noises
    #产生self.n_latent-1个Noise map

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.device
        )
        #随机采样self.n_latent*channel数的噪声

        latent = self.style(latent_in).mean(0, keepdim=True)
        # 先将噪声映射为隐层码w，再取平均得到1*style_dim的结果
        return latent


    def get_latent(self, input):
        return self.style(input)
    # 将噪声映射为隐层码w

    def sample_latent(self, num_samples):
        return torch.randn(num_samples, self.style_dim, device=self.device)

    def forward(self, input,
                return_latents=False,
                style_mix=0.9,
                input_is_latent=False,
                #=True代表input是隐层码w，=False则代表输入的input还需经过self.style（PixelNorm+8个EqualLinear）映射为隐层码w。
                noise=None):

        latent = self.style(input) if not input_is_latent else input

        if noise is None:
            noise = [None] * self.num_layers
            #noise是一个长度为self.num_layers（=self.n_latent-1）的空值列表
        if latent.ndim < 3:#ndim表示数组的维度2*3*3的数组即为3维
            latents = latent.unsqueeze(1).repeat(1, self.n_latent, 1)
            #若latent的维度小于3，则将其复制升维为(batch,self.n_latent,style_dim)
        else:
            latents = latent
        #经过以上操作，得到了size为(batch,self.n_latent,style_dim)的latent

            '''
        style mixing就是在生成主支网络中选择一个交叉点
        交叉点前的低分辨率合成使用w1控制,交叉点之后的高分辨率合成使用w2,
        这样最终得到的图像则融合了图A和图B的特征。根据交叉点位置的不同,可以得到不同的融合结果。
        下面是对latents进行处理使其完成上述工作。
            '''
        if self.training and (style_mix > 0):
            batch_size = input.size(0)
            latent_mix = self.style(self.sample_latent(batch_size))
            latent_mix = latent_mix.unsqueeze(1)

            nomix_mask = torch.rand(batch_size) >= style_mix
            mix_layer = torch.randint(self.n_latent, (batch_size,))
            mix_layer = mix_layer.masked_fill(nomix_mask, self.n_latent)
            mix_layer = mix_layer.unsqueeze(1)

            layer_idx = torch.arange(self.n_latent)[None]
            mask = (layer_idx < mix_layer).float().unsqueeze(-1)
            mask = mask.to(latents.device)

            latents = latents * mask + latent_mix * (1 - mask)
        #此时得到的latents是一个batch,self.n_latent,style_dim)的张量

        #获取了latent的第一个维度，也就是batch_size,去得到一个维度为（batchsize，self.channels[4]，4，4）的主枝常量输入out
        out = self.input(latents)
        #用self.conv1对out做卷积，输入latents[:, 0]，也就是self.n_latent中的第一个[batch,style_dim]的latent
        #经一个EqualLinear变为s，调制在卷积核权重上。做完卷积后，out维度还是(batch, 512, 4, 4)。
        out = self.conv1(out, latents[:, 0], noise=noise[0])
        #用18个中的第2个w对to_rgb中的卷积核权重进行调制，经卷积将out变为一个rgb通道图，维度(batch, 3, 4, 4).
        skip = self.to_rgb1(out, latents[:, 1])

        idx = 1
        """
        每次取出一层,取8次,分别从self.convs中取第1、3、5、7、9、11、13、15个卷积层为conv1
        从self.convs中取第2、4、6、8、10、12、14、16个卷积层为conv2,
        从noise取第2、4、6、8、10、12、14、16个噪声图为noise1,
        从noise取第3、5、7、9、11、13、15、17个噪声图为noise2,
        self.to_rgbs的全部8层为to_rgb。
        """
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.layers[::2], self.layers[1::2],
                noise[1::2], noise[2::2], self.to_rgbs
        ):
        #将第2~18个w分别用于调制这些层的卷积核权重，conv1、conv2先卷积再加噪。
            out = conv1(out, latents[:, idx], noise=noise1)
            out = conv2(out, latents[:, idx+1], noise=noise2)
        #to_rgb层中将上一层to_rgb的输出上采样加在本层输出上，最终生成的图片就是最后一层to_rgb的输出结果。
            skip = to_rgb(out, latents[:, idx+2], skip)
            idx += 2

        image = skip
        image = 0.5 * image + 0.5

        if not self.training:
            #限制image中的数值下限为0，上限为1
            image = image.clamp(0, 1)

        if return_latents:
            return image, latents
        else:
            return image