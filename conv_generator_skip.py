# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import trunc_normal_

from stylegan2.op import FusedLeakyReLU, conv2d_gradfix
from layers import PixelNorm, Upsample, Blur, EqualConv2d
from layers import EqualLinear, LFF
from stylegan2.vit_common import SpectralNorm, FeedForward
from stylegan2.vit_cips import CIPSGenerator
from stylegan2.op import FusedLeakyReLU
from stylegan2.generator import StyleLayer, ToRGB
#from einops import rearrange, repeat
#from einops.layers.torch import Rearrange
from torchvision import datasets, transforms

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,spectral_norm = True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        # if spectral_norm:
        #     self.fc1=SpectralNorm(self.fc1)
        #     self.fc2=SpectralNorm(self.fc2)


    def forward(self, x):                     
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,l2_attn = True,spectral_norm = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        #self.temperature = nn.Parameter(torch.FloatTensor([1.0]))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        #self.l2_attn=l2_attn
        #self.spectral_norm =spectral_norm

        # if spectral_norm:
        #     self.qkv = SpectralNorm(self.qkv)
        #     self.proj = SpectralNorm(self.proj)    

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # if self.l2_attn:
        #     AB = torch.matmul(q, k.transpose(-1, -2))
        #     AA = torch.mean(q * q, -1, keepdim=True)
        #     BB = AA    # Since query and key are tied.
        #     BB = BB.transpose(-1, -2)
        #     dots = AA - 2 * AB + BB
        #     dots = dots * self.scale
        # else:
        #     dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale        
        # dots = dots * self.temperature
        # attn=dots.softmax(dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale#x@y=x.matmul(y)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class SelfModulatedLayerNorm(nn.Module):
    def __init__(self, dim, modulated=False):
        super().__init__()
        self.modulated=modulated
        #self.param_free_norm = nn.LayerNorm(dim, eps=0.001, elementwise_affine=False)
        self.param_free_norm = nn.LayerNorm(dim, eps=0.001, elementwise_affine=True)
        self.mlp_gamma = EqualLinear(dim, dim, activation='linear',modulated=modulated)
        self.mlp_beta = EqualLinear(dim, dim, activation='linear',modulated=modulated)  

    def forward(self, inputs):
        x, cond_input = inputs
        bs = x.shape[0]
        cond_input = cond_input.reshape((bs, -1))#cond_input为torch.Size([21, 384])
        gamma = self.mlp_gamma(cond_input)
        gamma = gamma.reshape((bs, 1, -1))
        beta = self.mlp_beta(cond_input)
        beta = beta.reshape((bs, 1, -1))
        #print(f"self.param_free_norm.weight为{self.param_free_norm.weight}")
        out = self.param_free_norm(x)
        return out
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,level=4,en_size=196,modulated=False):
        super().__init__()
        self.norm1 = SelfModulatedLayerNorm(dim,modulated=modulated)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = SelfModulatedLayerNorm(dim,modulated=modulated)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        #print(f"self.skip的in_features为{en_size+1},out_features为{pow(2,2*level)}")
        #self.skip=nn.Linear(in_features=en_size+1,out_features=pow(2,2*level))
        self.skip=nn.Linear(in_features=en_size+1,out_features=256)
    def forward(self, inputs,return_attention=False):
        x,latent=inputs
        #print(f"x的shape为{x.shape}")
        #print(f"latent的shape为{latent.shape}")
        x=self.norm1([x,latent])
        #print(f"x为{x.shape}")
        #print(f"self.attn(x)为{type(self.attn(x))}")
        y,attn=self.attn(x)
        y=y+x
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x=self.norm2([x,latent])
        x=self.drop_path(self.mlp(x))+x
        #print(f"x为{x.shape}")
        x = x.permute((0, 2, 1))

        
        skip=self.skip(x)

        x = x.permute((0, 2, 1))
        skip=skip.permute((0, 2, 1))
        # y, attn = self.attn(self.norm1(x))
        # if return_attention:
        #     return attn
        # x = x + self.drop_path(y)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x,skip

class SinActivation(nn.Module):
    def __init__(self,):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)
    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    #def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        # print(f"img_size为{img_size}")
        # print(f"patch_size为{patch_size}")
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.activation = SinActivation()
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x=self.activation(x)
        return x    
    
# class Mid(nn.Module):
#     def __init__(self, input_size=[224], output_size=[128],patch_size=16, in_chans=3, embed_dim=384):
#         super().__init__()
# def vit_my_8(patch_size=16, noise=True,**kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),en_img_size=[224],de_img_size=[128],modulated=True, noise_injection=noise,**kwargs)
#     return model
class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, en_img_size=[224],de_img_size=[128], patch_size=16, in_chans=3, num_classes=0, embed_dim=384, depth=4,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 n_mlp=1, lr_mlp=0.01,blur_kernel=[1, 3, 3, 1],channel_multiplier=2,
                 modulated=False,g_decoder=True,noise_injection=False,**kwargs):
        super().__init__()
        self.en_img_size=en_img_size
        self.de_img_size=de_img_size
        self.patch_size=patch_size
        self.modulated=modulated
        self.g_decoder=g_decoder
        self.noise_injection=noise_injection
        # self.transTodeSize=transforms.Resize([de_img_size[0],de_img_size[0]])
        # self.transToGray=transforms.Grayscale(1)
        # self.transToRGBPIL=transforms.ToPILImage(mode='LAB')
        # self.transToHSVPIL=transforms.ToPILImage(mode='HSV')
        self.e_num_patch_w=self.en_img_size[0]//self.patch_size
        self.e_num_patch_h=self.en_img_size[0]//self.patch_size
        self.d_num_patch_w=self.de_img_size[0]//self.patch_size
        self.d_num_patch_h=self.de_img_size[0]//self.patch_size
        self.en_size=self.e_num_patch_h*self.e_num_patch_w
        self.de_size=self.d_num_patch_h*self.d_num_patch_w
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(embed_dim, embed_dim,
                                      lr_mul=lr_mlp,
                                      activation='fused_lrelu'))                                     
        self.style = nn.Sequential(*layers)#mapping network
        #self.activation = SinActivation()
        self.num_features = self.embed_dim = embed_dim#

        self.patch_embed = PatchEmbed(
            img_size=en_img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        de_img_size=self.de_img_size[0]
        self.log_size = int(math.log(de_img_size, 2))
        for i in range(4, self.log_size+1):#range(a,b)从a到b-1(4-7)
            self.blocks.append(Block(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,level=7,en_size=self.en_size,modulated=modulated))
        for i in range(self.log_size-1,3,-1):#6-4
            self.blocks.append(Block(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,level=i,en_size=self.en_size,modulated=modulated))           
        # self.blocks = nn.ModuleList([
        #     Block(
        #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,modulated=modulated)
        #     for i in range(depth)])
        #self.norm = norm_layer(embed_dim)
        self.convs = nn.ModuleList()
        # self.skips = nn.ModuleList()


        
        self.cnn_channels = {
            8: 384,
            16: 384,
            32: 384,
            # 64: int(192 * channel_multiplier),
            # 128: int(96 * channel_multiplier),
            # 256: int(48 * channel_multiplier),
            # 512: int(24 * channel_multiplier),
            # 1024: int(12 * channel_multiplier),
            64: 384,#int(192 * channel_multiplier),
            128: 384,#int(96 * channel_multiplier),
            256: 384,#int(48 * channel_multiplier),
            512: 384,#int(24 * channel_multiplier),
            1024: 384,#int(12 * channel_multiplier),
        }
        #in_channel = self.cnn_channels[8]
        for i in range(4, self.log_size+1):#range(a,b)从a到b-1(4-7)
            #out_channel = self.cnn_channels[2 ** i]#?
            #print(f"{i}StyleLayer的out_channel为{out_channel}")
            if i==4:
                in_channel = embed_dim
            elif i==5:
                in_channel=embed_dim*2
            else:
                #in_channel = embed_dim*2
                in_channel = int(embed_dim+embed_dim/pow(4,i-5))
            out_channel = embed_dim
            #print(f"in_channel为{in_channel}")
            self.convs.append(
                StyleLayer(in_channel, out_channel, 3, embed_dim,
                            upsample=True, blur_kernel=blur_kernel,demodulate=True,noise_injection=self.noise_injection)
                )
            
            self.convs.append(
                    StyleLayer(out_channel, out_channel, 3, embed_dim,
                        blur_kernel=blur_kernel,demodulate=True,noise_injection=noise_injection)
                )
        for i in range(0,depth-(self.log_size-3)):#0-3
            out_channel = embed_dim
            #in_channel = embed_dim*2
            in_channel = embed_dim+6
            self.convs.append(
                StyleLayer(in_channel, out_channel, 3, embed_dim,
                        blur_kernel=blur_kernel,demodulate=True,noise_injection=self.noise_injection)
                )
            self.convs.append(
                    StyleLayer(out_channel, out_channel, 3, embed_dim,
                        blur_kernel=blur_kernel,demodulate=True,noise_injection=noise_injection)
                )            
        #print(f"{i}ToRGB的out_channel为{out_channel}")
        self.to_rgb = ToRGB(out_channel+1, embed_dim, upsample=False)
        #self.to_rgb = ToRGB(out_channel, embed_dim, upsample=False)
  
        self.norm = SelfModulatedLayerNorm(embed_dim,modulated=modulated)


        self.mid=nn.Linear(in_features=self.en_size+1,out_features=self.de_size)
        #self.mid=nn.Conv2d(self.e_num_patch_w*self.e_num_patch_h,self.d_num_patch_w*self.d_num_patch_h) if self.e_num_patch_w != self.de_img_size else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        #trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
    
    @property
    def device(self):
        return self.lff.ffm.conv.weight.device

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        #npatch = x.shape[1]#64
        N = self.pos_embed.shape[1]-1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        #patch_pos_embed = self.pos_embed[:, 0:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        # a1=patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2)
        # a2=(w0 / math.sqrt(N), h0 / math.sqrt(N))
        # a3='bicubic'

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        #patch_pos_embed = nn.functional.interpolate(a1,scale_factor=a2,mode=a3)
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        #return  patch_pos_embed
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x
        #return self.pos_drop(x)

    @property
    def device(self):
        return self.patch_embed.proj.weight.device

    def sample_latent(self, num_samples):
        return torch.randn(num_samples, self.embed_dim, device=self.device)
        #return torch.ones(num_samples, self.embed_dim, device=self.device)
        #return torch.zeros(num_samples, self.embed_dim, device=self.device)

    def forward(self, x,input,illu=None,
                input_is_latent=False):
        if not self.modulated:
            latent=torch.zeros(x.shape[0],self.embed_dim).cuda()
        else:
            latent = self.style(input) if not input_is_latent else input
        bs=latent.shape[0]

        x = self.prepare_tokens(x)

        clist=[]

        for i,blk in enumerate(self.blocks):
            x ,skip= blk([x,latent])
            clist.append(skip)
        x = self.norm([x, latent])

        
      

        if not self.g_decoder:
            return x[:, 0]
        else: 
            #x=x[:,1:]
            #print(f"x.shape为{x.shape};x.shape[-1]为{x.shape[-1]}")
            #x.shape为torch.Size([21, 196, 384]);x.shape[-1]为384
            # num_patch_w=128//self.patch_size
            # num_patch_h=128//self.patch_size 
            #x=self.mid(x)
            x = x.permute((0, 2, 1))
            x=self.mid(x)
            x = x.permute((0, 2, 1))


            #x = x.reshape((bs, self.patch_size, self.patch_size, x.shape[-1]))
            #indices = torch.tensor(range(1, num_patch_w*num_patch_h+1))
            indices = torch.tensor(range(0, self.d_num_patch_w*self.d_num_patch_h))
            indices = indices.to(torch.device(self.device))
            #print(f"indices为{indices}")
            #x=torch.index_select(x, 1, indices)
            x=torch.index_select(x, 1, indices)
            x = x.reshape((bs, self.d_num_patch_w, self.d_num_patch_h, x.shape[-1]))
            x = x.permute((0, 3, 1, 2))
            for i,conv_layer in enumerate(self.convs):
                if i==0 or i%2!=0:
                #if i%2!=0:
                    x = conv_layer(x, latent)
                else: 
                    i=int(i/2)
                    #print(f"i为{i}")
                    c=clist[self.log_size-i]
                    
                    if i<=3:
                        #c=c.reshape(bs,pow(2,(3+i)),pow(2,(3+i)),self.embed_dim)
                        c=c.reshape(bs,pow(2,(3+i)),pow(2,(3+i)),-1)
                    else:
                        #c=c.reshape(bs,128,128,self.embed_dim)
                        c=c.reshape(bs,128,128,-1)
                    #c=c.reshape(bs,14,14,self.embed_dim)
                    c=c.permute(0,3,1,2)
                    x=torch.cat((x,c),dim=1)
                    #x=0.5*x+0.5*c
                    x = conv_layer(x, latent)
                    #print(f"{self.device}:x的大小为{x.shape}")
            #x=torch.cat((x,x_gray),dim=1)#bs,4,128,128
            #x=x+x_gray
            illu=illu.unsqueeze(0)
            illu=illu.permute(1,0,2,3)
            if illu is not None:
                x=torch.cat((x,illu),dim=1)
            x = self.to_rgb(x, latent)
            return x

    def get_last_selfattention(self, x):#！！！
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),en_img_size=[224], de_img_size=[128],**kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), en_img_size=[224],de_img_size=[128],**kwargs)
    return model

def vit_my(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),en_img_size=[224],de_img_size=[128],modulated=True, **kwargs)
    return model

def vit_my_8(patch_size=16, noise=True,**kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),en_img_size=[224],de_img_size=[128],modulated=True, noise_injection=noise,**kwargs)
    return model

def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),en_img_size=[224],de_img_size=[128], **kwargs)
    return model

def vit_encoder(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),en_img_size=[224],de_img_size=[128],g_decoder=False, **kwargs)
    return model

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)#!
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
