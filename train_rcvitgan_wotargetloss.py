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
from models.gan import get_architecture
from utils import cycle,cycle3,cycle4

from training.gan import setup
from utils import Logger
from utils import count_parameters
from utils import accumulate
from utils import set_grad
#from data_parallel1 import BalancedDataParallel

# import for gin binding
import penalty

import wandb
import time

# import for evaluation
#from evaluate.gan import FIDScore, FixedSampleGeneration, ImageGrid

#from torchvision import datasets, transforms
from mydiscriminator import ResidualDiscriminatorP#,Pix2PixDiscriminator
from fid_score import my_fid_score

from tensorboardX import SummaryWriter
writer=SummaryWriter('out/log_dino_noSLN_smoothL1_wotargetloss')#可视化数据放在这个文件夹

from collections import OrderedDict
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
try:
    from third_party.fid.inception import InceptionV3
except ImportError:
    from inception import InceptionV3
#用于计算metrics的工具
def eval_step(engine, batch):
    return batch
default_evaluator = Engine(eval_step)
default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(4, 2)),
    ('fc', nn.Linear(2, 1))
]))
# # wrapper class as feature_extractor
# class WrapperInceptionV3(nn.Module):

#     def __init__(self, fid_incv3):
#         super().__init__()
#         self.fid_incv3 = fid_incv3

#     @torch.no_grad()
#     def forward(self, x):
#         y = self.fid_incv3(x)
#         y = y[0]
#         y = y[:, :, 0, 0]
#         return y
    
# block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[192]
# model = InceptionV3([block_idx]).cuda()
# # wrapper model to pytorch_fid model
# wrapper_model = WrapperInceptionV3(model)
# wrapper_model.eval()

from color_harmonization import ch_loss
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
def parse_args():
    """Training script for StyleGAN2."""

    parser = ArgumentParser(description='Training script: StyleGAN2 with DataParallel.')
    parser.add_argument('gin_config', type=str, help='Path to the gin configuration file')
    parser.add_argument('architecture', type=str, help='Architecture')

    parser.add_argument('--mode', default='std', type=str, help='Training mode (default: std)')
    parser.add_argument('--penalty', default='none', type=str, help='Penalty (default: none)')
    parser.add_argument('--aug', default='none', type=str, help='Augmentation (default: hfrt)')
    parser.add_argument('--use_warmup', action='store_true', help='Use warmup strategy on LR')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')

    parser.add_argument('--temp', default=0.1, type=float,
                        help='Temperature hyperparameter for contrastive losses')
    parser.add_argument('--lbd_a', default=1.0, type=float,
                        help='Relative strength of the fake loss of ContraD')

    # Options for StyleGAN2 training
    parser.add_argument('--no_lazy', action='store_true',
                        help='Do not use lazy regularization')
    parser.add_argument("--d_reg_every", type=int, default=16,
                        help='Interval of applying R1 when lazy regularization is used')
    parser.add_argument("--lbd_r1", type=float, default=10, help='R1 regularization')
    parser.add_argument('--style_mix', default=0.9, type=float, help='Style mixing regularization')
    parser.add_argument('--halflife_k', default=20, type=int,
                        help='Half-life of exponential moving average in thousands of images')
    parser.add_argument('--ema_start_k', default=None, type=int,
                        help='When to start the exponential moving average of G (default: halflife_k)')
    parser.add_argument('--halflife_lr', default=0, type=int, help='Apply LR decay when > 0')

    parser.add_argument('--use_nerf_proj', action='store_true', help='是否对LR采用预热策略Use warmup strategy on LR')

    # Options for logging specification
    parser.add_argument('--no_fid', action='store_true',
                        help='Do not track FIDs during training')
    parser.add_argument('--no_gif', action='store_true',
                        help='Do not save GIF of sample generations from a fixed latent periodically during training')
    parser.add_argument('--n_eval_avg', default=3, type=int,
                        help='How many times to average FID and IS')
    parser.add_argument('--print_every', help='', default=1000, type=int)#默认值改过了
    parser.add_argument('--evaluate_every', help='', default=2000, type=int)#默认值改过了！
    parser.add_argument('--save_every', help='', default=10000, type=int)#默认值改过了！
    parser.add_argument('--comment', help='Comment', default='', type=str)

    # Options for resuming / fine-tuning
    # resume和finetune的区别在哪？
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to logdir to resume the training')
    parser.add_argument('--finetune', default=None, type=str,
                        help='Path to logdir that contains a pre-trained checkpoint of D')

    return parser.parse_args()


def _update_warmup(optimizer, cur_step, warmup, lr):#这属于线性warm up
    if warmup > 0:
        ratio = min(1., (cur_step + 1) / (warmup + 1e-8))
        lr_w = ratio * lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_w

#lr每次decay为之前的一半好像比较合理？所以halflife_lr=batchsize*1000好像比较合理
#改成每隔一万步变成了原来的一半
def _update_lr(optimizer, cur_step, batch_size, halflife_lr, lr, mult=1.0):
    if halflife_lr > 0 and (cur_step > 0) and (cur_step % 10000 == 0):
        #ratio = (cur_step * batch_size) / halflife_lr
        ratio=cur_step/10000-11
        lr_mul = 0.5 ** ratio
        lr_w = lr_mul * lr * mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_w
        return lr_w
    return None


def r1_loss(D, images, augment_fn):
    images_aug = augment_fn(images).detach()
    images_aug.requires_grad = True
    d_real = D(images_aug)
    grad_real, = autograd.grad(outputs=d_real.sum(), inputs=images_aug,
                               create_graph=True, retain_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def _sample_generator(G, num_samples, enable_grad=True,imgs=None,illus=None):
    latent_samples = G.sample_latent(num_samples)
    #print(f"输入的x=imgs形状为:{imgs.device}输入的latent_samples的形状为{latent_samples.device}")
    if enable_grad:
        generated_data = G(x=imgs,input=latent_samples,illu=illus)
        #print(summary(G, imgs.shape,latent_samples.shape))
    else:
        with torch.no_grad():
            generated_data = G(x=imgs,input=latent_samples,illu=illus)
            #print(summary(G, imgs.shape,latent_samples.shape))
    return generated_data

#改训练的epoch这里也要改！！！
@gin.configurable("options")
def get_options_dict(dataset=gin.REQUIRED,
                     loss=gin.REQUIRED,
                     batch_size=32, fid_size=10000,
                     max_steps=200, warmup=0, n_critic=1,
                     lr=0.002, lr_d=None, beta=(.0, .99),
                     lbd=1., lbd2=1.):
    if lr_d is None:
        lr_d = lr
    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "fid_size": fid_size,
        "loss": loss,
        "max_steps": max_steps, "warmup": warmup,
        "n_critic": n_critic,
        "lr": lr, "lr_d": lr_d, "beta": beta,
        "lbd": lbd, "lbd2": lbd2
    }


def train(P, opt, train_fn, models, optimizers, 
            ltrain_ltarget_pair_loader,
            presudo_pair_loader,
            val_pair_loader,logger):
    generator, discriminator_pair,discriminator_single, g_ema = models
    opt_G, opt_DP,opt_DS = optimizers
    for param_group in opt_G.param_groups:
        param_group['lr'] = opt["lr"]
        print(f"G_lr为{param_group['lr']}")
    for param_group in opt_DP.param_groups:
        param_group['lr'] = opt["lr"]
        print(f"DP_lr为{param_group['lr']}")
    for param_group in opt_DS.param_groups:
        param_group['lr'] = opt["lr"]
        print(f"DS_lr为{param_group['lr']}")
    
    losses = {'G_loss': [],'G_critic_loss': [], 'G_l_mse_loss': [], 'G_ch_loss': [],
                'DP_loss': [], 'DP_penalty': [],'DP_real': [], 'DP_gen': [], 'DP_r1': [],
                'DS_loss': [], 'DS_penalty': [],'DS_real': [], 'DS_gen': [], 'DS_r1': []
              }
    metrics={'fid_score':[]}
    # metrics={}
    # metrics['fid_score'] = FIDScore(opt['dataset'], opt['fid_size'], P.n_eval_avg)
    #data_range取决于图片是否归一化，应该就是有归一化就是1.0没有就是225.0
    metric_SSIM = SSIM(data_range=1.0)
    metric_PSNR = PSNR(data_range=1.0)
    #metric_FID = FID(num_features=192, feature_extractor=default_model)
    metric_SSIM.attach(default_evaluator, 'ssim')
    metric_PSNR.attach(default_evaluator, 'psnr')
    #metric_FID.attach(default_evaluator, 'fid')
    logger.log_dirname("Steps {}".format(P.starting_step))

    for step in range(P.starting_step, opt['max_steps'] + 1):
        if step % P.evaluate_every == 0:
            val_images,val_images128,val_target_images,val_illus=next(val_pair_loader)
            val_images = val_images.cuda()#batch_size除以4
            val_target_images=val_target_images.cuda()
            val_illus=val_illus.cuda()
            val_images128=val_images128.cuda()
            with torch.no_grad():
                val_gen_images = _sample_generator(generator, val_images.size(0),enable_grad=True,imgs=val_images,illus=val_illus)
                val_ch_loss=ch_loss(val_gen_images,img_size=128)
                val_target_loss=F.smooth_l1_loss(val_gen_images,val_target_images)
            #writer.add_scalar('stage2_val_fid',val_fid_value,step)
            #print(f"图片类型为{val_target_images.dtype}和{val_gen_images.dtype}")torch.float32
            #print(f"{torch.max(val_target_images)-torch.min(val_target_images)}")1.0
            #print(f"{torch.max(val_gen_images)-torch.min(val_gen_images)}")1.0
            #这里是跟目标图片算的PSNR和SSIM，应该这样吧？
            state = default_evaluator.run([[val_gen_images,val_target_images]])
            fid_value=my_fid_score(path_base='base_stats.npz', G=generator, size=val_images.size(0), batch_size=val_images.size(0), model=None, dims=192)#这里返回的应该就是实数吧？--是
            #先不让fid_score来影响训练，只是单纯打印它
            metrics['fid_score'].append(fid_value)
            writer.add_scalar('stage2_val_SSIM',state.metrics['ssim'],step)
            writer.add_scalar('stage2_val_PSNR',state.metrics['psnr'],step)
            writer.add_scalar('stage2_val_FID',fid_value,step)
            #writer.add_scalar('stage2_val_FID',state.metrics['fid'],step)
            writer.add_scalar('stage2_val_ch_loss',val_ch_loss,step)
            writer.add_scalar('stage2_val_target_loss',val_target_loss,step)
            # logger.log('[Steps %7d][stage2_val_SSIM %.7f][stage2_val_PSNR %.7f][stage2_val_FID %.7f][stage2_val_ch_loss %.7f] [stage2_val_target_loss %.14f]' %
            #     (step, state.metrics['ssim'], state.metrics['psnr'],state.metrics['fid'], val_ch_loss, val_target_loss))
            logger.log('[Steps %7d][stage2_val_SSIM %.7f][stage2_val_PSNR %.7f][stage2_val_ch_loss %.7f] [stage2_val_target_loss %.14f]' %
                (step, state.metrics['ssim'], state.metrics['psnr'],val_ch_loss, val_target_loss))
        d_regularize = (step % P.d_reg_every == 0) and (P.lbd_r1 > 0)

        if P.use_warmup:
            _update_warmup(opt_G, step, opt["warmup"], opt["lr"])
            _update_warmup(opt_DP, step, opt["warmup"], opt["lr_d"])
            _update_warmup(opt_DS, step, opt["warmup"], opt["lr_d"])
        if (not P.use_warmup) or step > opt["warmup"]:
            cur_lr_g = _update_lr(opt_G, step, opt["batch_size"], P.halflife_lr, opt["lr"])
            cur_lr_dp = _update_lr(opt_DP, step, opt["batch_size"], P.halflife_lr, opt["lr_d"])
            cur_lr_ds = _update_lr(opt_DS, step, opt["batch_size"], P.halflife_lr, opt["lr_d"])
            if cur_lr_dp and cur_lr_ds and cur_lr_g:
                logger.log('LR Updated: [G %.10f] [DP %.10f] [DS %.10f]' % (cur_lr_g, cur_lr_dp,cur_lr_ds))
        #实现ema（权重移动平均）：权重还是原来的权重，只是权重更新时采用的梯度不是当前梯度，而是梯度的滑动平均
        do_ema = (step * opt['batch_size']) > (P.ema_start_k * 1000)#过了一阵之后才采用滑动平均125000之后才采用滑动平均
        accum = P.accum if do_ema else 0
        accumulate(g_ema, generator, accum)#accum就是滑动平均中的decay

        # Start discriminator training
        generator.train()
        discriminator_pair.train()
        discriminator_single.train()

        



        
        images,target_images,illus,real_images=next(presudo_pair_loader)#这里的illus其实是改变了尺寸的训练图片
        images=images.cuda()
        target_images=target_images.cuda()
        illus=illus.cuda()
        real_images=real_images.cuda()


        #有标签的原图和有标签的目标图对
        ltrain_images, ltarget_images,lillus,lgan_images = next(ltrain_ltarget_pair_loader)
        ltrain_images=ltrain_images.cuda()
        ltarget_images=ltarget_images.cuda()
        lillus=lillus.cuda()
        lgan_images=lgan_images.cuda()


        set_grad(generator, False)
        set_grad(discriminator_pair, True)
        set_grad(discriminator_single, True)

        #条件图片和生成图片对
        l_img_input = torch.cat((lgan_images, ltarget_images), 1)
        ugen_images = _sample_generator(generator, images.size(0),enable_grad=True,imgs=images,illus=illus)
        # trans=transforms.Compose([
        #     transforms.ToPILImage(),#这里默认输入是RGB
        #     transforms.Resize((128, 128)),
        #     transforms.ToTensor()
        # ])

        
        #illus=illus.to(ugen_images.device)
        # print(f"illus在{illus.device}")
        #print(f"illus的大小为{illus.shape},ugen_images的大小为{ugen_images.shape}")
        # illu=illus.unsqueeze(0)
        # illu=illu.permute(1,0,2,3)
        u_img_input=torch.cat((real_images, ugen_images), 1)
        # print(f"u_img_input在{u_img_input.device}")
        #l_img_input=l_img_input.to(u_img_input.device)
        # print(f"l_img_input在{l_img_input.device}")
        #utrain_color_images和ugen_images的配对是假（D要判定为假，G要增加判定为真的概率）
        #但utrain_color_images和ugen_images的纹理应该一样，即灰度图应该一样
        # if d_regularize:
        #    u_img_input.requires_grad = True

        #ds和dp都用
        dp_loss, dp_aux = train_fn["train3_D_match"](P, discriminator_pair, opt,l_img_input,u_img_input)
        ds_loss, ds_aux = train_fn["train3_D_match"](P, discriminator_single, opt,real_images,ugen_images)
        loss = dp_loss+ ds_loss+ dp_aux['penalty']+ds_aux['penalty']
        # #只用ds
        # ds_loss, ds_aux = train_fn["train3_D_match"](P, discriminator_single, opt,real_images,ugen_images)
        # loss = ds_loss+ ds_aux['penalty']

        #if d_regularize:
        #    r1 = r1_loss(discriminator, images, P.augment_fn)
        #    lazy_r1 = (0.5 * P.lbd_r1) * r1 * P.d_reg_every
        #    loss = loss + lazy_r1
        #    losses['D_r1'].append(r1.item())
        #opt_DP.zero_grad()
        opt_DS.zero_grad()
        loss.backward()
        #opt_DP.step()
        opt_DS.step()

        losses['DP_loss'].append(dp_loss.item())
        losses['DP_real'].append(dp_aux['d_real'].item())
        losses['DP_gen'].append(dp_aux['d_gen'].item())
        losses['DP_penalty'].append(dp_aux['penalty'].item())
        losses['DS_loss'].append(ds_loss.item())
        losses['DS_real'].append(ds_aux['d_real'].item())
        losses['DS_gen'].append(ds_aux['d_gen'].item())
        losses['DS_penalty'].append(ds_aux['penalty'].item())




        # Start generator training
        set_grad(generator, True)
        set_grad(discriminator_pair, False)
        set_grad(discriminator_single, False)
        ugen_images = _sample_generator(generator, images.size(0),enable_grad=True,imgs=images,illus=illus)
        lgen_images = _sample_generator(generator, ltrain_images.size(0),enable_grad=True,imgs=ltrain_images,illus=lillus)
    
    
        u_img_input=torch.cat((real_images, ugen_images), 1)
        #print(f"ugen_images的位置在{ugen_images.device}")
        
        gp_loss ,gp_aux= train_fn["train3_G_wotargetloss_match"](P, discriminator_pair, opt,lgen_images,ltarget_images,u_img_input)#,ugen_images)
        gs_loss ,gs_aux= train_fn["train3_G_wotargetloss_match"](P, discriminator_single, opt,ugen_images,target_images,ugen_images)

        g_loss=gp_loss+gs_loss
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()


    
        losses['G_loss'].append(g_loss.item())
        losses['G_critic_loss'].append(gp_aux['critic_loss'].item()+gs_aux['critic_loss'].item())
        losses['G_l_mse_loss'].append(gp_aux['l_mse_loss'].item()+gs_aux['l_mse_loss'].item())
        losses['G_ch_loss'].append(gp_aux['g_ch_loss'].item()+gs_aux['g_ch_loss'].item())

      

        writer.add_scalars('stage2_G_D',{'g_loss': g_loss.item(), 'd_loss': loss.item()}, step)
        writer.add_scalars('stage2_G',{'g_critic_loss': losses['G_critic_loss'][-1], 'g_l_mse_loss': losses['G_l_mse_loss'][-1],'g_ch_loss':losses['G_ch_loss'][-1]}, step)

        generator.eval()
        discriminator_pair.eval()
        discriminator_single.eval()

        if step % P.print_every == 0:
            logger.log('[Steps %7d][G %.7f][G_critic %.7f] [G_l_mse %.7f] [G_ch_loss %.14f][DP %.7f][DP_real %.7f][DP_gen %.7f][DP_penalty %.7f][DS %.7f][DS_real %.7f][DS_gen %.7f][DS_penalty %.7f]' %
                       (step, losses['G_loss'][-1], losses['G_critic_loss'][-1], losses['G_l_mse_loss'][-1], losses['G_ch_loss'][-1],
                        losses['DP_loss'][-1],losses['DP_real'][-1], losses['DP_gen'][-1],losses['DP_penalty'][-1],
                        losses['DS_loss'][-1],losses['DS_real'][-1], losses['DS_gen'][-1],losses['DS_penalty'][-1]))

            for name in losses:
                values = losses[name]
                if len(values) > 0:
                    logger.scalar_summary('gan/train/' + name, values[-1], step)
    

        if step % P.evaluate_every == 0:
            logger.log_dirname("Steps {}".format(step + 1))
            #wandb.log({"augmented_real_images": wandb.Image(aug_grid), "generated_images": wandb.Image(fixed_gen.summary()[-1])}, step=step)
            #fid_score = metrics.get('fid_score')
            G_state_dict = generator.module.state_dict()
            DP_state_dict = discriminator_pair.module.state_dict()
            DS_state_dict = discriminator_single.module.state_dict()
            Ge_state_dict = g_ema.state_dict()
            #fid_value=my_fid_score(path_base='base_stats.npz', G=generator, size=images.size(0), batch_size=images.size(0), model=None, dims=192)#这里返回的应该就是实数吧？--是
            #先不让fid_score来影响训练，只是单纯打印它
            #metrics['fid_score'].append(fid_value)
            logger.log('[Steps %7d][fid_score %.7f]' %(step, metrics['fid_score'][-1]))


            # if fid_score:
            #     fid_avg = fid_score.update(step, g_ema)
            #     fid_score.save(logger.logdir + f'/results_fid_{P.eval_seed}.csv')
            #     logger.scalar_summary('gan/test/fid', fid_avg, step)
            #     logger.scalar_summary('gan/test/fid/best', fid_score.best, step)
            torch.save(G_state_dict, logger.logdir + '/gen_stage3_wotargetloss.pt')
            torch.save(DP_state_dict, logger.logdir + '/disP_stage3_wotargetloss.pt')
            torch.save(DS_state_dict, logger.logdir + '/disS_stage3_wotargetloss.pt')
            torch.save(Ge_state_dict, logger.logdir + '/gen_ema_stage3_wotargetloss.pt')
            # if fid_score and fid_score.is_best:
            #     torch.save(G_state_dict, logger.logdir + '/gen_best.pt')
            #     torch.save(DP_state_dict, logger.logdir + '/dis_best.pt')
            #     torch.save(DS_state_dict, logger.logdir + '/dis_best.pt')
            #     torch.save(Ge_state_dict, logger.logdir + '/gen_ema_best.pt')
            if step % P.save_every == 0:
                torch.save(G_state_dict, logger.logdir + f'/gen_stage3_wotargetloss_{step}.pt')
                torch.save(DP_state_dict, logger.logdir +f'/disP_stage3_wotargetloss_{step}.pt')
                torch.save(DS_state_dict, logger.logdir + f'/disS_stage3_wotargetloss_{step}.pt')
                torch.save(Ge_state_dict, logger.logdir + f'/gen_ema_stage3_wotargetloss_{step}.pt')
                torch.save({'epoch': step,'optim_G': opt_G.state_dict(),'optim_DP': opt_DP.state_dict(),'optim_DS': opt_DS.state_dict(),
            }, logger.logdir + f'/optim_stage3_wotargetloss_{step}.pt')
            
            torch.save({
                'epoch': step,
                'optim_G': opt_G.state_dict(),
                'optim_DP': opt_DP.state_dict(),
                'optim_DS': opt_DS.state_dict(),
            }, logger.logdir + '/optim_stage3_wotargetloss.pt')


def worker(P):
    gin.parse_config_files_and_bindings(['configs/defaults/gan.gin',
                                         'configs/defaults/augment.gin',
                                         P.gin_config], [])

    options = get_options_dict()
    seed=10
    torch.manual_seed(seed)
    ltrain_lgan_ltarget_pair_set,image_size=get_dataset(dataset='labeled_data_stage3')
    presudo_pair_set,resolution= get_dataset(dataset='unlabeled_data1_LAB_presudo_stage3')
    val_pair_set,val_resolution=get_dataset(dataset='val_data_stage3')

    seed=10
    torch.manual_seed(seed)
    ltrain_ltarget_pair_loader=DataLoader(ltrain_lgan_ltarget_pair_set, shuffle=True, pin_memory=True, num_workers=P.workers,
                              batch_size=options['batch_size'], drop_last=True)
    presudo_pair_loader=DataLoader(presudo_pair_set, shuffle=True, pin_memory=True, num_workers=P.workers,
                              batch_size=options['batch_size'], drop_last=True)
    val_pair_loader = DataLoader(val_pair_set, shuffle=False, pin_memory=False, num_workers=P.workers,
                              batch_size=50, drop_last=False)    
    ltrain_ltarget_pair_loader = cycle4(ltrain_ltarget_pair_loader)
    presudo_pair_loader=cycle4(presudo_pair_loader)
    val_pair_loader=cycle4(val_pair_loader)


    if P.no_lazy:
        P.d_reg_every = 1
    if P.ema_start_k is None:
        P.ema_start_k = P.halflife_k
        

    P.accum = 0.5 ** (options['batch_size'] / (P.halflife_k * 1000))
    #-----加载模型结构---------
    from vit_generator_skip import vit_my_8
    resolution = image_size[0]
    generator = vit_my_8(patch_size=16)
    g_ema = vit_my_8(patch_size=16)
    discriminator_pair = ResidualDiscriminatorP(size=resolution, small32=False,mlp_linear=True, d_hidden=512)
    discriminator_single = ResidualDiscriminatorP(size=resolution, small32=False,mlp_linear=True, d_hidden=512,input_channel=3)
    if P.resume:
        print(f"=> Loading checkpoint from '{P.resume}'")
        # state_G = torch.load(f"{P.resume}/gen_stage3.pt")
        # state_DP = torch.load(f"{P.resume}/disP_stage3.pt")
        # state_DS = torch.load(f"{P.resume}/disS_stage3.pt")
        # state_Ge = torch.load(f"{P.resume}/gen_ema_stage3.pt")
        state_G = torch.load(f"{P.resume}/gen_100000_stage2.pt")
        state_Ge = torch.load(f"{P.resume}/gen_ema_100000_stage2.pt")
        #state_G = torch.load(f"{P.resume}/gen_best.pt")
        #print(f"state_G为{state_G.items()}")
        generator.load_state_dict(state_G,strict=False)
        # discriminator_pair.load_state_dict(state_DP,strict=True)
        # discriminator_single.load_state_dict(state_DS,strict=True)
        g_ema.load_state_dict(state_Ge,strict=False)
        
    if P.finetune:
        print(f"=> Loading checkpoint for fine-tuning: '{P.finetune}'")
        state_DP = torch.load(f"{P.finetune}/dis_pair.pt")
        discriminator_pair.load_state_dict(state_DP, strict=False)
        discriminator_pair.reset_parameters(discriminator_pair.linear)
        state_DS = torch.load(f"{P.finetune}/dis_single.pt")
        discriminator_single.load_state_dict(state_DS, strict=False)
        discriminator_single.reset_parameters(discriminator_single.linear)
        P.comment += 'ft'

    generator = generator.cuda()
    discriminator_pair = discriminator_pair.cuda()
    discriminator_single = discriminator_single.cuda()
    g_ema = g_ema.cuda()
    g_ema.eval()

    for name, param in generator.named_parameters():
        if "cls_token" in name or "pos_embed" in name or "style." in name or "blocks." in name or "norm." in name:
            param.requires_grad=False
            #print(f"层名分别为{name}") 
    G_optimizer = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()),lr=options["lr"], betas=options["beta"]) 
    # G_optimizer = optim.Adam(generator.parameters(),
    #                          lr=options["lr"], betas=options["beta"])
    D_optimizer_pair = optim.Adam(discriminator_pair.parameters(),
                             lr=options["lr_d"], betas=options["beta"])
    D_optimizer_single = optim.Adam(discriminator_single.parameters(),
                             lr=options["lr_d"], betas=options["beta"])

    if P.resume:
        logger = Logger(None, resume=P.resume)
        #wandb.init(project='vitgan', name=f'{P.gin_stem}_{P.architecture}_' + f'{P.filename}_{_desc}{P.comment}', resume=True)

        #wandb.config.update(P)
        #wandb.config.update(options)

        #wandb.watch(generator)
        #wandb.watch(discriminator)
    else:
        _desc = f"R{P.lbd_r1}_H{P.halflife_k}"
        if P.halflife_lr > 0:
            _desc += f"_lr{P.halflife_lr / 1000000:.1f}M"
        _desc += f"_NoLazy" if P.no_lazy else "_Lazy"
        _desc += f"_Warmup" if P.use_warmup else "_NoWarmup"

        logger = Logger(f'{P.filename}_{_desc}{P.comment}', subdir=f'gan_dp/{P.gin_stem}/{P.architecture}')
        #wandb.init(project='vitgan', name=f'{P.gin_stem}_{P.architecture}_' + f'{P.filename}_{_desc}{P.comment}')
        
        #wandb.config.update(P)
        #wandb.config.update(options)

        #wandb.watch(generator)
        #wandb.watch(discriminator)

        shutil.copy2(P.gin_config, f"{logger.logdir}/config.gin")
    P.logdir = logger.logdir
    P.eval_seed = np.random.randint(10000)

    if P.resume:
        opt = torch.load(f"{P.resume}/optim_stage2_100000.pt")
        # G_optimizer.load_state_dict(opt['optim_G'])
        # D_optimizer_pair.load_state_dict(opt['optim_DP'])
        # D_optimizer_single.load_state_dict(opt['optim_DS'])
        logger.log(f"Checkpoint loaded from '{P.resume}'")
        P.starting_step = opt['epoch'] + 1
    else:
        logger.log(generator)
        logger.log(discriminator_pair)
        logger.log(discriminator_single)
        logger.log(f"# Params - G: {count_parameters(generator)}, D_pair: {count_parameters(discriminator_pair)}, D_single: {count_parameters(discriminator_single)}")
        logger.log(options)
        P.starting_step = 1
    logger.log(f"Use G moving average: {P.accum}")

    if P.finetune:
        logger.log(f"Checkpoint loaded from '{P.finetune}'")

#augment_fn是从这里来的！！！！我们的P.aug是diffaug，DiffAugment使我们对生成的样本采用可微增广，有效地稳定了训练，并使其收敛得更好。
#以防止Discriminator直接记住真实的数据集

    P.augment_fn = get_augment(mode=P.aug).cuda()
    generator = nn.DataParallel(generator)
    g_ema = nn.DataParallel(g_ema)
    #generator = BalancedDataParallel(2, module=generator, dim=0).cuda()
    #g_ema = BalancedDataParallel(2, module=g_ema, dim=0).cuda()
    generator.sample_latent = generator.module.sample_latent
    discriminator_pair = nn.DataParallel(discriminator_pair)
    #discriminator_pair=BalancedDataParallel(2, module=discriminator_pair, dim=0).cuda()
    discriminator_single = nn.DataParallel(discriminator_single)
    #discriminator_single=BalancedDataParallel(2, module=discriminator_single,dim=0).cuda()

    train(P, options, P.train_fn,
          models=(generator, discriminator_pair, discriminator_single, g_ema),
          optimizers=(G_optimizer, D_optimizer_pair,D_optimizer_single),
            ltrain_ltarget_pair_loader=ltrain_ltarget_pair_loader,
            presudo_pair_loader=presudo_pair_loader,
            val_pair_loader=val_pair_loader,logger=logger)


if __name__ == '__main__':
    P = parse_args()
    if P.comment:
        P.comment = '_' + P.comment
    P.gin_stem = Path(P.gin_config).stem
    P = setup(P)
    P.distributed = False
    worker(P)
