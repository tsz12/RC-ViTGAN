import torch
import torch.nn.functional as F
from penalty import compute_penalty
from torch import autograd
from torchvision import datasets, transforms
from color_harmonization import ch_loss

de_zero_transform= transforms.Compose([
            transforms.ToPILImage (),
            transforms.Resize([128,128]),
            transforms.ToTensor()
        ])

def loss_D_fn(P, D, options, images, gen_images):
    assert images.size(0) == gen_images.size(0)
    gen_images = gen_images.detach()
    N = images.size(0)
    images.requires_grad = True
    #来源于WGAN的梯度惩罚，拼接图片的数据，将拼接张量输入判别器，拿到输入判别器图片数据的梯度（注意不是网络weights的梯度）
    #对梯度计算norm，看这个Norm离单位距离1多远，距离越远惩罚越小
    #在真假样本之间随意插值来惩罚，使得真假样本的过渡区域满足一阶拉普拉斯约束
    all_images = torch.cat([images, gen_images], dim=0)
    d_all = D(P.augment_fn(all_images))#用d_all来存储discriminator的输出值
    d_real, d_gen = d_all[:N], d_all[N:]#分别取第一维度的前N维后第一维度的后N维

    if options['loss'] == 'nonsat':
        d_loss = F.softplus(d_gen).mean() + F.softplus(-d_real).mean()#critic loss
    elif options['loss'] == 'wgan':
        d_loss = d_gen.mean() - d_real.mean()
    elif options['loss'] == 'hinge':
        d_loss = F.relu(1. + d_gen, inplace=True).mean() + F.relu(1. - d_real, inplace=True).mean()
    else:
        raise NotImplementedError()

    grad_real, = autograd.grad(outputs=d_real.sum(), inputs=images,#images_aug,
                               create_graph=True, retain_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    penalty = compute_penalty(P.penalty, P=P, D=D, all_images=all_images,
                              images=images, gen_images=gen_images,
                              d_real=d_real, d_gen=d_gen,
                              lbd=options['lbd'], lbd2=options['lbd2'])
    penalty += grad_penalty * (0.5 * P.lbd_r1)

    return d_loss, {
        "penalty": penalty,
        "d_real": d_real.mean(),
        "d_gen": d_gen.mean(),
    }

def loss_D_match_fn(P, D, options, l_img_input,u_img_input):

    u_img_input = u_img_input.detach()
    NL = l_img_input.size(0)
    NU = u_img_input.size(0)
    l_img_input.requires_grad = True
    #来源于WGAN的梯度惩罚，拼接图片的数据，将拼接张量输入判别器，拿到输入判别器图片数据的梯度（注意不是网络weights的梯度）
    #对梯度计算norm，看这个Norm离单位距离1多远，距离越远惩罚越小
    #在真假样本之间随意插值来惩罚，使得真假样本的过渡区域满足一阶拉普拉斯约束
    all_images = torch.cat([l_img_input, u_img_input], dim=0)
    #print(f"P.augment_fn(all_images)的大小为{P.augment_fn(all_images).shape}")
    d_all = D(P.augment_fn(all_images))#用d_all来存储discriminator的输出值
    #d_all = D(all_images)
    d_real, d_gen = d_all[:NL], d_all[NL:]#分别取第一维度的前N维后第一维度的后N维

    if options['loss'] == 'nonsat':
        d_loss = F.softplus(d_gen).mean() + F.softplus(-d_real).mean()#critic loss这个损失函数的目的是拉开真假样本输出值的差异
    elif options['loss'] == 'wgan':
        d_loss = d_gen.mean() - d_real.mean()
    elif options['loss'] == 'hinge':
        d_loss = F.relu(1. + d_gen, inplace=True).mean() + F.relu(1. - d_real, inplace=True).mean()
    else:
        raise NotImplementedError()

    grad_real, = autograd.grad(outputs=d_real.sum(), inputs=l_img_input,#images_aug,
                               create_graph=True, retain_graph=True)#autograd.grad()用于对输入变量求导而不是对权重求导,backward()用于对权重求导
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()#.mean()对所有位置的数求均值，该函数返回的是一个实数
    #直接惩罚的梯度，没有限制它要小于1，也没有限制它要等于1，直接惩罚的梯度本身
    #bcr
    penalty = compute_penalty(P.penalty, P=P, D=D, all_images=all_images,
                              images=l_img_input, gen_images=u_img_input,
                              d_real=d_real, d_gen=d_gen,
                              lbd=options['lbd'], lbd2=options['lbd2'])#options.lbd=1,options.lbd2=1
    #print(f"平衡一致性损失为{penalty}")
    #penalty=bcr+grad_penalty*0.05
    penalty += grad_penalty * (0.5 * P.lbd_r1)

    return d_loss, {
        "penalty": penalty,
        "d_real": d_real.mean(),
        "d_gen": d_gen.mean(),
    }


def loss_G_match_fn(P, D, options, u_img_input):
    d_gen = D(P.augment_fn(u_img_input))
    #d_gen = D(u_img_input)
    if options['loss'] == 'nonsat':
        g_loss = F.softplus(-d_gen).mean()#softplus是relu的平滑
    else:
        g_loss = -d_gen.mean()
    return g_loss

def loss_G_fn(P, D, options, images, gen_images):
    d_gen = D(P.augment_fn(gen_images))
    if options['loss'] == 'nonsat':
        g_loss = F.softplus(-d_gen).mean()
    else:
        g_loss = -d_gen.mean()

    return g_loss

#增加一个预训练的损失函数

def loss_G_pre_fn(P, D, options, images, gen_images):
    g_loss=F.mse_loss(gen_images,images)#改过了！！！
    return g_loss


def loss_train1_AB_fn(P,options, target_images, gen_images):
    target_images_LAB=transforms.ToPILImage(mode='LAB')(target_images)
    gen_images_LAB=transforms.ToPILImage(mode='LAB')(gen_images)
    target_images_LAB_tensor=transforms.ToTensor()(target_images_LAB)
    gen_images_LAB_tensor=transforms.ToTensor()(gen_images_LAB)

    g_loss=F.smooth_l1_loss(target_images_LAB_tensor[:,1:3,:,:], gen_images_LAB_tensor[:,1:3,:,:])
    return g_loss

def loss_train1_fn(P,options, target_images, gen_images):
    g_loss=F.smooth_l1_loss(target_images, gen_images)
    return g_loss

from skimage.color import rgb2lab, lab2rgb
import numpy as np
def get_AB(img):
    """
    Get the luminance of an image. Shape: (h, w)
    """
    #print(f"img大小为{img.shape}")
    img = img.permute(0,2,3, 1)  # (h, w, channel) 
    img = img.cpu().detach()
    img = img.numpy()
    img = img.astype(np.float) / 255.0
    img_LAB = rgb2lab(img)
    img_AB = img_LAB[:,:,:,1:3]  # luminance  # (h, w)
    return torch.from_numpy(img_AB)

def loss_train2_fn(P,options, target_images, gen_images):
    #target_images_AB=get_AB(target_images)
    #gen_images_AB=get_AB(gen_images)
    #g_loss=F.mse_loss(target_images, gen_images)
    g_loss=F.smooth_l1_loss(target_images, gen_images)#这个函数默认是求均值的
    #g_loss=F.smooth_l1_loss(target_images_AB, gen_images_AB)*1000
    #g_loss=g_loss.requires_grad_(True)
    return g_loss

def loss_D_my_fn(P, D, options, ltrain_gray_images, ltrain_color_images,
                 ltarget_gray_images, ltarget_color_images,
                 utrain_gray_images, utrain_color_images,
                 ltrain_images, ltarget_images,
                 utrain_images,
                 lgen_images,
                 ugen_images):
    
    assert ltarget_images.size(0) == lgen_images.size(0)
    NL = ltarget_images.size(0)
    d_l_loss,l_aux=loss_D_fn(P, D, options, ltarget_images, lgen_images)

  
    assert utrain_images.size(0) == ugen_images.size(0)
    NU = utrain_images.size(0)
    d_u_loss,u_aux=loss_D_fn(P, D, options, utrain_images, ugen_images)

    assert ltarget_images.size(0) == lgen_images.size(0)
    d_r_loss,r_aux=loss_D_fn(P, D, options, utrain_images, ugen_images)    

    wl=NL/(NU+2*NL)
    wu=NU/(NU+2*NL)
    wr=NL/(NU+2*NL)

    d_loss=wl*d_l_loss+wu*d_u_loss+wr*d_r_loss
    penalty=wl*l_aux['penalty']+wu*u_aux['penalty']+wr*r_aux['penalty']
    d_real_mean=wl*l_aux['d_real']+wu*u_aux['d_real']+wr*r_aux['d_real']
    d_gen_mean=wl*l_aux['d_gen']+wu*u_aux['d_gen']+wr*r_aux['d_gen']
    return d_loss, {
        "penalty": penalty,
        "d_real": d_real_mean,
        "d_gen": d_gen_mean,
    }


def loss_G_my_fn(P, D, options, ltrain_par_images, ltrain_color_images,
                 ltarget_gray_images, ltarget_color_images,
                 utrain_gray_images, utrain_color_images,
                 ltrain_images, ltarget_images,
                 utrain_images,
                 lgen_images,
                 ugen_images):
    NL = ltrain_images.size(0)
    NU = utrain_images.size(0)
    wl=NL/(NU+NL)
    wu=NU/(NU+NL)
    w=10
    g_l_critic_loss=loss_G_fn(P, D, options, ltrain_images, lgen_images)
    g_u_critic_loss=loss_G_fn(P, D, options, utrain_images, ugen_images)
    g_l_mse_loss=F.smooth_l1_loss(lgen_images,ltarget_images)
    g_critic_loss=wl*g_l_critic_loss+wu*g_u_critic_loss

    return g_critic_loss+w*g_l_mse_loss

#dp_loss, dp_aux = train_fn["train3_D_match"](P, discriminator_pair, opt,l_img_input,u_img_input)
#ds_loss, ds_aux = train_fn["train3_D_match"](P, discriminator_single, opt,real_images,ugen_images)
def loss_D_my_match_fn(P, D, options,l_img_input,u_img_input):
    NL = l_img_input.size(0)
    NU = u_img_input.size(0)
    d_loss,aux=loss_D_match_fn(P, D, options, l_img_input, u_img_input)

    return d_loss, {
        "penalty": aux['penalty'],
        "d_real": aux['d_real'],
        "d_gen": aux['d_gen'],
    }     

#gp_loss ,gp_aux= train_fn["train3_G_match"](P, discriminator_pair, opt,lgen_images,ltarget_images,u_img_input)
#gs_loss ,gs_aux= train_fn["train3_G_match"](P, discriminator_single, opt,ugen_images,target_images,ugen_images)
def loss_G_my_match_fn(P, D,options,lgen_images,ltarget_images,u_img_input):
     g_l_mse_loss=F.smooth_l1_loss(lgen_images,ltarget_images)
     g_critic_loss=loss_G_match_fn(P, D,options,u_img_input)
    #  utrain_color_images=u_img_input[:,0:3,:,:]
    #  ugen_images=u_img_input[:,3:6,:,:]
    #  trans=transforms.Grayscale(3)
    #  utrain_color_images_gray=trans(utrain_color_images)
    #  ugen_images_gray=trans(ugen_images)
     #g_u_mse_loss=F.smooth_l1_loss(utrain_color_images_gray,ugen_images_gray)
     g_ch_loss=ch_loss(lgen_images,img_size=128)
     #g_loss=g_critic_loss+100*g_l_mse_loss#+g_ch_loss
     g_loss=g_critic_loss+100*g_l_mse_loss+g_ch_loss#+50*g_u_mse_loss
     return g_loss,{
         "critic_loss":g_critic_loss,
         "l_mse_loss":g_l_mse_loss,
         "g_ch_loss":g_ch_loss
         #"lexture_loss":g_u_mse_loss
     }
def loss_G_my_match_wochloss_fn(P, D,options,lgen_images,ltarget_images,u_img_input):
     g_l_mse_loss=F.smooth_l1_loss(lgen_images,ltarget_images)
     g_critic_loss=loss_G_match_fn(P, D,options,u_img_input)
     g_ch_loss=0
     g_loss=g_critic_loss+100*g_l_mse_loss+g_ch_loss#+50*g_u_mse_loss
     return g_loss,{
         "critic_loss":g_critic_loss,
         "l_mse_loss":g_l_mse_loss,
         "g_ch_loss":g_ch_loss
         #"lexture_loss":g_u_mse_loss
     }
def loss_G_my_match_wotargetloss_fn(P, D,options,lgen_images,ltarget_images,u_img_input):
     g_l_mse_loss=0
     g_critic_loss=loss_G_match_fn(P, D,options,u_img_input)
     g_ch_loss=ch_loss(lgen_images,img_size=128)
     g_loss=g_critic_loss+100*g_l_mse_loss+g_ch_loss#+50*g_u_mse_loss
     return g_loss,{
         "critic_loss":g_critic_loss,
         "l_mse_loss":g_l_mse_loss,
         "g_ch_loss":g_ch_loss
         #"lexture_loss":g_u_mse_loss
     }

