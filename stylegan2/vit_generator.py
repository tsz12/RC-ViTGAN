# The code is based on code publicly available at
#   https://github.com/rosinality/stylegan2-pytorch
# written by Seonghyeon Kim.

import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from models.gan.stylegan2.op import FusedLeakyReLU, conv2d_gradfix
from models.gan.stylegan2.layers import PixelNorm, Upsample, Blur, EqualConv2d
from models.gan.stylegan2.layers import EqualLinear, LFF#分别为1*1的卷积和LFF正则化
from models.gan.stylegan2.vit_common import SpectralNorm, Attention, FeedForward
from models.gan.stylegan2.vit_cips import CIPSGenerator
from models.gan.stylegan2.op import FusedLeakyReLU
from models.gan.stylegan2.generator import StyleLayer, ToRGB#用到了StyleGAN2的mapping network和toRGB block
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

#转换到坐标形式
def convert_to_coord_format(b, h, w, device='cpu', integer_values=False):
    if integer_values:
        x_channel = torch.arange(w, dtype=torch.float, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.arange(h, dtype=torch.float, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    else:
        x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    #print(f"convert_to_coord_format为:{torch.cat((x_channel, y_channel), dim=1).size()}")
    #torch.Size([b, 2, w, h])
    #坐标x和y分别用h*w的张量表示？
    return torch.cat((x_channel, y_channel), dim=1)

#AdaIN----->vitgan特有
class SelfModulatedLayerNorm(nn.Module):
    def __init__(self, dim, spectral_norm=False):
        super().__init__()
        self.param_free_norm = nn.LayerNorm(dim, eps=0.001, elementwise_affine=False)
        if spectral_norm:
            self.mlp_gamma = SpectralNorm(EqualLinear(dim, dim, activation='linear'))
            self.mlp_beta = SpectralNorm(EqualLinear(dim, dim, activation='linear'))
        else:
            self.mlp_gamma = EqualLinear(dim, dim, activation='linear')
            self.mlp_beta = EqualLinear(dim, dim, activation='linear')

    def forward(self, inputs):
        #对positional embedding进行调制
        x, cond_input = inputs#cond_input为torch.Size([64, 384]),x为torch.Size([64, 64, 384])
        #这里虽然叫cond_input其实就是lantent
        #print(f"x和cond_input相等吗?{x==cond_input}")不等
        #print(f"cond_input为{cond_input.shape}")
        #print(f"x为{x.shape}")
        bs = x.shape[0]
        cond_input = cond_input.reshape((bs, -1))#cond_input为torch.Size([64, 384])
        #print(f"cond_input为{cond_input.shape}")
        #cond_input原大小是[b, 2, w, h]，将其按bs展开成二维张量，一个batch对应一个长度为384的向量

        '''
        self.mlp_gamma = SpectralNorm(EqualLinear(dim, dim, activation='linear'))
        EqualLinear返回的是一个module,前向过程为forward(self, input)
        self.mlp_beta = SpectralNorm(EqualLinear(dim, dim, activation='linear'))
        if SpectralNorm为false的话就没有外面的SpectralNorm
        '''
        #cond_inout为Epos，也为h0
        gamma = self.mlp_gamma(cond_input)#self.mlp_gamma,self.mlp_beta都是参数可学习的线性层
        gamma = gamma.reshape((bs, 1, -1))
        beta = self.mlp_beta(cond_input)
        beta = beta.reshape((bs, 1, -1))#-1表示维度值用其它维度决定

        out = self.param_free_norm(x)#LayerNorm
        out = out * (1.0 + gamma) + beta

        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim = 384, heads = 6, dim_head = 64, \
        mlp_dim = 1536, l2_attn = True, spectral_norm = True, dropout = 0.):

        super().__init__()
        self.layernorm1 = SelfModulatedLayerNorm(dim, spectral_norm = False)
        self.attn = Attention(dim, heads = heads, dim_head = dim_head, \
                    l2_attn = l2_attn, spectral_norm = spectral_norm, dropout = dropout)#Multi-Head Attention
        self.layernorm2 = SelfModulatedLayerNorm(dim, spectral_norm = False)
        self.ff = FeedForward(dim, mlp_dim, spectral_norm = spectral_norm, dropout = dropout)#MLP

        
    def forward(self, inputs):
        x, latent = inputs#注意看inputs是啥，这里应该就是positional embedding和latent
        x = self.layernorm1([x, latent])
        x = self.attn(x) + x
        x = self.layernorm2([x, latent])
        x = self.ff(x) + x
        return x

#搞清vitgan的generator结构
class Generator(nn.Module):
    #def __init__(self, size=32, token_width=8, num_layers=4,
    def __init__(self, size=128, token_width=8, num_layers=4,#这里是我自己改过的
        style_dim=384, n_mlp=8, channel_multiplier=2, small32=False,
        blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, use_nerf_proj=False,
        patch_size=16,channels=3):#这里的patch_size其实是根据token_width和图片的尺寸决定的。128/8=16
        super().__init__()
        self.size = size#这里的size是get_Architecture时指定，就是dataset.py返回的imagesize，此时是128
        self.style_dim = style_dim#w的维度384
        self.token_width = token_width#token的维度8，这个是说明图片分块的时候咋分
        self.use_nerf_proj = use_nerf_proj#控制是否使用卷积来加速收敛

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim,
                                      lr_mul=lr_mlp,
                                      activation='fused_lrelu'))
                                      

        self.style = nn.Sequential(*layers)#mapping network

        #1为batchsize
        self.coords = convert_to_coord_format(1, token_width, token_width, integer_values=False, device='cpu')#self.device)
        #lff正则化层
        self.lff = LFF(style_dim)

        #【改】将图片切分成patchs
        #--------------------------------------
        self.patch_size=patch_size#表示一个patch的大小
        self.channels=channels
        assert self.size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'  # 保证一定能够完整切块
        num_patches = (self.size// self.patch_size) ** 2  # 获取图像切块的个数，64
        patch_dim =self.channels * self.patch_size ** 2  # 线性变换时的输入大小，即每一个图像patch宽、高、通道的乘积
        #assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'  # 池化方法必须为cls或者mean

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size), 
            # 将批量为b通道为c高为h*p1宽为w*p2的图像转化为批量为b个数为h*w维度为p1*p2*c的图像块
            # 即，把b张c通道的图像分割成b*（h*w）张大小为P1*p2*c的图像块
            # 例如：patch_size为16  (8, 3, 48, 48)->(8, 9, 768)
            nn.Linear(patch_dim, self.style_dim),  
            # 对分割好的图像块进行线性处理（全连接），输入维度为每一个小块的所有像素个数，输出为dim（函数传入的参数）
        )
        #--------------------------------------
        self.feat_dim = {
            0: 384,
            1: 384,
            2: 384,
            3: 384,
            4: 384,#int(256 * channel_multiplier),
            5: 384,#int(128 * channel_multiplier),
            6: 384,#int(64 * channel_multiplier),
            7: 384,#int(32 * channel_multiplier),
            8: 384,#int(16 * channel_multiplier),
        }

        self.cnn_channels = {
            8: 384,
            16: 384,
            32: 384,
            64: 384,#int(192 * channel_multiplier),
            128: 384,#int(96 * channel_multiplier),
            256: 384,#int(48 * channel_multiplier),
            512: 384,#int(24 * channel_multiplier),
            1024: 384,#int(12 * channel_multiplier),
        }

        #self.input = ConstantInput(self.channels[4])

        self.log_size = int(math.log(size, 2))#7
        self.num_layers = (self.log_size - 2) * 2 + 1#11

        self.layers = nn.ModuleList()
        self.convs = nn.ModuleList()
        #self.to_rgbs = nn.ModuleList()
        #self.noises = nn.Module()

        #in_channel = self.channels[4]
        for i in range(num_layers):
            this_dim = self.feat_dim[i]
            self.layers.append(TransformerBlock(dim = this_dim, heads = 6, dim_head = this_dim // 6, \
                                mlp_dim = this_dim*4, l2_attn = True, spectral_norm = True, dropout = 0.))

        self.layernorm = SelfModulatedLayerNorm(self.feat_dim[num_layers-1], spectral_norm=False)
        

        in_channel = self.cnn_channels[8]#什么时候取8什么时候取256?别挣扎了，反正都是384
        #in_channel = self.cnn_channels[256]

        if self.use_nerf_proj == False:
            #这里相当于加入了多个权重调制层再加RGB
            #根据y来构造图片
            for i in range(4, self.log_size + 1):#range(a,b)从a到b-1
                out_channel = self.cnn_channels[2 ** i]
                #print(f"{i}StyleLayer的out_channel为{out_channel}")
                self.convs.append(
                    StyleLayer(in_channel, out_channel, 3, style_dim,
                            upsample=True, blur_kernel=blur_kernel)
                )
                self.convs.append(
                    StyleLayer(out_channel, out_channel, 3, style_dim,
                            blur_kernel=blur_kernel)
                )
            #print(f"{i}ToRGB的out_channel为{out_channel}")
            self.to_rgb = ToRGB(out_channel, style_dim, upsample=False)
        else:
            self.cips = CIPSGenerator(size=size//token_width, style_dim=style_dim, n_mlp=4)
        
    @property
    def device(self):
        return self.lff.ffm.conv.weight.device

    def make_noise(self):
        noises = []
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            noises.append(torch.randn(*shape, device=self.device))
        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def sample_latent(self, num_samples):
        return torch.randn(num_samples, self.style_dim, device=self.device)

    def forward(self, input,
                return_latents=False,
                style_mix=0.9,
                input_is_latent=False,
                noise=None,imgs=None):
        #print(f"input的batch_size为{input.shape[0]}")
        latent = self.style(input) if not input_is_latent else input
        #print(f"latent的大小为:{latent.shape}")
        bs = latent.shape[0]
        #print(f"bs的大小为:{bs}")
        #print(f"imgs为None吗{imgs==None}")
        #----------------------------------------------------------------------------
        if imgs != None:
            #print(f"imgs为None吗,imgs的大小为？{imgs==None}{imgs.shape}")
            patch_embedding=self.to_patch_embedding(imgs).to(self.device)
            #print(f"patch_embedding的大小为:{patch_embedding.shape}")
        #----------------------------------------------------------------------------
        
        coords = self.coords.repeat(bs, 1, 1, 1).to(self.device)#("cuda")#.repeat()表示在对应维度上复制几次
        pe = self.lff(coords)
        #print(f"pe的大小为:{pe.shape}")
        #x = torch.permute(pe, (0, 2, 3, 1)).reshape((bs, -1, self.style_dim))
        #permute用于交换维度的索引
        x=pe.permute((0, 2, 3, 1)).reshape((bs, -1, self.style_dim))#把二维的position embedding token拉平
        #print(f"加上patch_embedding前x的大小为:{x.shape}")
        #在输入上加入原图片的patch embedding
        #--------------------------------------
        if imgs != None:
             x+=patch_embedding
        #--------------------------------------

        #print(f"加上patch_embedding后x的大小为:{x.shape}")
        for layer in self.layers:
            x = layer([x, latent])
            #print(f"x的大小为:{x.shape}")
        x = self.layernorm([x, latent])
        #这里之后得到的应该是y=[y1,y2,...,yl]
        #print(f"x的大小为:{x.shape}")
        if self.use_nerf_proj == False:
            #根据y来构造再着色图片
            x = x.reshape((bs, 8, 8, x.shape[-1]))#x.shape[-1]表示最后一维的维数，其实x.shape就是一个列表
            x = x.permute((0, 3, 1, 2))#b,8,8,2 
            #print(f"x的大小为:{x.shape}")
            for conv_layer in self.convs:
                x = conv_layer(x, latent)#对特征图进行上采样得到最终结果,这里其实就是没加噪声的
                #print (f"x的大小为{x.shape}")
            x = self.to_rgb(x, latent)
            #print (f"to_rgb后x的大小为{x.shape}")
        else:
            x = x.reshape((-1, x.shape[-1]))
            #print (x.shape)
            x = self.cips(x)
            mul = self.size // self.token_width
            #torch.Size([128, 3, 4, 4])

            x = x.reshape((bs, self.token_width, self.token_width, 3, mul, mul))
            x = x.permute((0, 3, 1, 4, 2, 5))
            x = x.reshape([bs, 3, self.size, self.size])

        return x
