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
from utils import cycle,cycle3

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
from tensorboardX import SummaryWriter
writer=SummaryWriter('out/log_dino_noSLN_smoothL1')#可视化数据放在这个文件夹

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
                        help='Do not use lazy regularization')#lazy regulation每个16个mini-batch才进行一次R1正则化，这个应该不是不能收敛的理由
    parser.add_argument("--d_reg_every", type=int, default=16,
                        help='Interval of applying R1 when lazy regularization is used')
    parser.add_argument("--lbd_r1", type=float, default=10, help='R1 regularization')
    parser.add_argument('--style_mix', default=0.9, type=float, help='Style mixing regularization')
    parser.add_argument('--halflife_k', default=20, type=int,
                        help='Half-life of exponential moving average in thousands of images')
    parser.add_argument('--ema_start_k', default=None, type=int,
                        help='When to start the exponential moving average of G (default: halflife_k)')
    parser.add_argument('--halflife_lr', default=0, type=int, help='Apply LR decay when > 0')

    parser.add_argument('--use_nerf_proj', action='store_true', help='是否采用卷积来加速收敛')

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
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to logdir to resume the training')
    parser.add_argument('--finetune', default=None, type=str,
                        help='Path to logdir that contains a pre-trained checkpoint of D')

    return parser.parse_args()

def _update_warmup(optimizer, cur_step, warmup, lr):
    if warmup > 0:
        ratio = min(1., (cur_step + 1) / (warmup + 1e-8))
        lr_w = ratio * lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_w


def _update_lr(optimizer, cur_step, batch_size, halflife_lr, lr, mult=1.0):
    if halflife_lr > 0 and (cur_step > 0) and (cur_step % 10000 == 0):
        ratio = (cur_step * batch_size) / halflife_lr
        #ratio=cur_step/1000-18
        lr_mul = 0.5 ** ratio
        lr_w = lr_mul * lr * mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_w
        return lr_w
    return None

def _sample_generator(G, num_samples, enable_grad=True,imgs=None,illus=None):
    latent_samples = G.sample_latent(num_samples)
    if enable_grad:
        generated_data = G(x=imgs,input=latent_samples,illu=illus)
    else:
        with torch.no_grad():
            generated_data = G(x=imgs,input=latent_samples,illu=illus)
    return generated_data

#把下面这些参数绑定到options中可以直接用
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

def train(P,opt,train_fn,models,optimizers,pair_loader,val_pair_loader,logger):
    #opt是各种超参数，optimizer是worker建立的优化器，如果是resume的话是加载过之前的参数的
    #如果当前设置的lr跟之前不一样，如果有warmup会将优化器的学习率更新为设置的lr
    #如果没有warmup，halflife_lr又为0，学习率会更新成新的吗，并不会--->加了

    generator,g_ema=models
    opt_G=optimizers

    losses={'G_train2_loss':[]}

    logger.log_dirname("Steps {}".format(P.starting_step))


    for param_group in opt_G.param_groups:
            param_group['lr'] = opt["lr"]
            print(f"lr为{param_group['lr']}")  
    
    best_val_loss=float('inf')

    for step in range(P.starting_step, opt['max_steps'] + 1):
        if step % P.evaluate_every == 0:
            val_images,val_target_images,val_illus=next(val_pair_loader)
            val_images = val_images.cuda()#batch_size除以4
            val_target_images=val_target_images.cuda()
            val_illus=val_illus.cuda()
            with torch.no_grad():
                val_gen_images = _sample_generator(generator, val_images.size(0),enable_grad=True,imgs=val_images,illus=val_illus)
                g_val_loss = train_fn["train2"](P, opt, val_target_images,val_gen_images)
            writer.add_scalar('stage1_val_loss',g_val_loss.item(),step)
        
        if P.use_warmup:
            _update_warmup(opt_G, step, opt["warmup"], opt["lr"])
        if (not P.use_warmup) or step > opt["warmup"]:
            cur_lr_g = _update_lr(opt_G, step, opt["batch_size"], P.halflife_lr, opt["lr"])
            if cur_lr_g:
                logger.log('LR Updated: [G %.10f] ' % (cur_lr_g))
        #实现ema（权重移动平均）：权重还是原来的权重，只是权重更新时采用的梯度不是当前梯度，而是梯度的滑动平均
        do_ema = (step * opt['batch_size']) > (P.ema_start_k * 1000)#过了一阵之后才采用滑动平均
        accum = P.accum if do_ema else 0#accum就是滑动平均中的decay
        accumulate(g_ema, generator, accum)
        generator.train()
       
        images,target_images,illus=next(pair_loader)
        
        # #-----检查一下数据是否加载成功---------
        # print(f"检查一下数据是否加载成功")
        # illus=illus.unsqueeze(0)
        # illus=illus.permute(1,0,2,3)
        # g=illus.repeat(1,3,1,1)
        # for i in range(10):
        #     print(f"现在的时间是{time.time()}")
        #     images_PIL= transforms.ToPILImage()(images[i])
        #     images_PIL.save(f"./load_image/train_set/{step}_{i}.png")
        #     target_images_PIL= transforms.ToPILImage()(target_images[i])
        #     target_images_PIL.save(f"./load_image/target_set/{step}_{i}.png")   
        #     g_PIL=transforms.ToPILImage()(g[i])
        #     g_PIL.save(f"./load_image/g_set/{step}_{i}.png")        
            
        images = images.cuda()#batch_size除以4
        target_images=target_images.cuda()
        illus=illus.cuda()
        #illus=None
        #target_images=target_images.cuda()

        set_grad(generator, True)
        #print(f"images在{images.device},illus在{illus.device}")
        gen_images = _sample_generator(generator, images.size(0),enable_grad=True,imgs=images,illus=illus)
        
        g_loss = train_fn["train2"](P, opt, target_images,gen_images)
        opt_G.zero_grad()
        g_loss.backward()
        #查看梯度回传是否正常
        # for name, parms in generator.named_parameters():
        #     if 'style.' in name:	
        #         print(f"name: {name}, -->grad_requirs:{parms.requires_grad},-->grad_value:{parms.grad}")
        opt_G.step()
        writer.add_scalar('stage1_g_loss',g_loss.item(),step)
        if step % P.evaluate_every == 0:
            writer.add_scalars('stage1_train_val',{'g_train_loss': g_loss.item(), 'g_val_loss': g_val_loss.item()}, step)
        losses['G_train2_loss'].append(g_loss.item())
        generator.eval()

        if step % P.print_every == 0:
            logger.log('[Steps %7d] [G %.7f]' %(step, losses['G_train2_loss'][-1]))
            for name in losses:
                values = losses[name]
                if len(values) > 0:
                    logger.scalar_summary('train/train2' + name, values[-1], step)#logger.scalar_summary(tag, value, idx)
            #wandb.log({"G_train2_loss": losses['G_train2_loss'][-1]}, step=step)
            #-----查看模型更新梯度是否正常------
            # for name, parms in generator.named_parameters():
            #     if parms.requires_grad:
            #         print(f"{name}更新的梯度为{parms.grad}")
        
            
        if step % P.evaluate_every == 0:
            #writer.add_scalars('stage1_train_val',{'g_train_loss': g_loss.item(), 'g_val_loss': g_val_loss.item()}, step)
            #-----各评价指标------
            logger.log_dirname("Steps {}".format(step + 1))
            G_state_dict = generator.module.state_dict()
            Ge_state_dict = g_ema.module.state_dict()
            torch.save(G_state_dict, logger.logdir + '/gen_stage2.pt')
            torch.save(Ge_state_dict, logger.logdir + '/gen_ema_stage2.pt')
            # if g_val_loss < best_val_loss:
            #     best_val_loss=g_val_loss
            #     torch.save(G_state_dict, logger.logdir + f'/gen_best_stage2.pt')
            #     torch.save(Ge_state_dict, logger.logdir + f'/gen_ema_best_stage2.pt')
            #     torch.save({'epoch': step,'optim_G': opt_G.state_dict(),}, logger.logdir + f'/optim_stage2_best.pt')#模型当前参数

            if step % P.save_every == 0:
                torch.save(G_state_dict, logger.logdir + f'/gen_{step}_stage2.pt')
                torch.save(Ge_state_dict, logger.logdir + f'/gen_ema_{step}_stage2.pt')
                torch.save({'epoch': step,'optim_G': opt_G.state_dict(),}, logger.logdir + f'/optim_stage2_{step}.pt')#模型当前参数
            torch.save({
                'epoch': step,
                'optim_G': opt_G.state_dict(),
            }, logger.logdir + '/optim_stage2.pt')#模型当前参数



def worker(P):
    gin.parse_config_files_and_bindings(['configs/defaults/gan.gin',
                                         'configs/defaults/augment.gin',
                                         P.gin_config], [])    

    options = get_options_dict()
    #train_set, _, target_set,image_size = get_dataset(dataset=options['labeled_data'])
    print(f"get_dataset(dataset=options['dataset'])为{get_dataset(dataset=options['dataset'])}")
    pair_set,resolution= get_dataset(dataset=options['dataset'])#"unlabeled_data1_LAB_presudo"
    val_pair_set,val_resolution=get_dataset(dataset='val_data')
    # -----检查一下数据是否加载成功---------
    # for i in range(5):
    #     #print(f"现在的时间是{time.time()}")
    #     images_PIL= transforms.ToPILImage()(train_set[i][0])
    #     images_PIL.save(f"./load_image/train_set/{i}.png")
    #     target_images_PIL= transforms.ToPILImage()(target_set[i][0])
    #     target_images_PIL.save(f"./load_image/target_set/{i}.png")   
    seed=10
    torch.manual_seed(seed)
    # train_loader = DataLoader(train_set, shuffle=False, pin_memory=True, num_workers=P.workers,
    #                           batch_size=options['batch_size'], drop_last=True)
    # target_loader = DataLoader(target_set, shuffle=False, pin_memory=True, num_workers=P.workers,
    #                           batch_size=options['batch_size'], drop_last=True)
    pair_loader = DataLoader(pair_set, shuffle=True, pin_memory=False, num_workers=P.workers,
                              batch_size=options['batch_size'], drop_last=True)
    val_pair_loader = DataLoader(val_pair_set, shuffle=False, pin_memory=False, num_workers=P.workers,
                              batch_size=92, drop_last=True)
    # train_loader = cycle(train_loader)
    # target_loader = cycle(target_loader)
    pair_loader = cycle3(pair_loader)
    val_pair_loader=cycle3(val_pair_loader)
 

    if P.ema_start_k is None:
        P.ema_start_k = P.halflife_k

    P.accum = 0.5 ** (options['batch_size'] / (P.halflife_k * 1000))#accum是由这俩决定的

    #-----加载模型结构---------
    from vit_generator_skip import vit_small,vit_my,vit_my_8
    #resolution = image_size[0]
    generator = vit_my_8(patch_size=16,noise=False)
    g_ema = vit_my_8(patch_size=16,noise=False)
    #-----加载train1训练参数---------
    #-----加载dino的预训练参数---------
    #print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
    url = None
    #vit_small patch_size=16
    #url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth" # model used for visualizations in our paper
    if url is not None:
        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        info_generator=generator.load_state_dict(state_dict, strict=False)
        info_g_ema=g_ema.load_state_dict(state_dict, strict=False)
        print(f"generator的预训练参数加载结果为{info_generator}")
        print(f"g_ema的预训练参数加载结果为{info_g_ema}")

    if P.resume:
        print(f"=> Loading checkpoint from '{P.resume}'")
        state_G = torch.load(f"{P.resume}/gen_stage2.pt")
        state_Ge = torch.load(f"{P.resume}/gen_ema_stage2.pt")        
        info_generator=generator.load_state_dict(state_G,strict=False)
        info_g_ema=g_ema.load_state_dict(state_Ge,strict=False)
        print(f"generator的train1参数加载结果为{info_generator}")
        print(f"g_ema的train1参数加载结果为{info_g_ema}")
    #print(f"成功加载train1参数")
    generator = generator.cuda()
    g_ema = g_ema.cuda()
    g_ema.eval()

    G_optimizer = optim.Adam(generator.parameters(),
                             lr=options["lr"], betas=options["beta"])

    if P.resume:
        logger = Logger(None, resume=P.resume)  
        # wandb.init(project='"train2_vitgenerator"', name=f'{P.gin_stem}_{P.architecture}_' + f'{P.filename}_{_desc}{P.comment}', resume=True)
        # wandb.config.update(P)
        # wandb.config.update(options)

        # wandb.watch(generator)
        # #wandb.watch(discriminator)    
    else:
        _desc = f"R{P.lbd_r1}_H{P.halflife_k}"
        if P.halflife_lr > 0:
            _desc += f"_lr{P.halflife_lr / 1000000:.1f}M"
        _desc += f"_NoLazy" if P.no_lazy else "_Lazy"
        _desc += f"_Warmup" if P.use_warmup else "_NoWarmup"

        logger = Logger(f'{P.filename}_{_desc}{P.comment}', subdir=f'gan_dp/{P.gin_stem}/{P.architecture}')  
        # wandb.init(project='"train2_vitgenerator"', name=f'{P.gin_stem}_{P.architecture}_' + f'{P.filename}_{_desc}{P.comment}')
        
        # wandb.config.update(P)
        # wandb.config.update(options)

        # wandb.watch(generator)
        # #wandb.watch(discriminator)
        #使用shutil.copy2()方法将文件从源复制到目标
        shutil.copy2(P.gin_config, f"{logger.logdir}/config.gin")
    P.logdir = logger.logdir
    P.eval_seed = np.random.randint(10000)
    #P.starting_step = 0
    if P.resume:
        opt = torch.load(f"{P.resume}/optim_stage2.pt")
        G_optimizer.load_state_dict(opt["optim_G"])#!
        logger.log(f"Checkpoint loaded from '{P.resume}'")
        P.starting_step = opt['epoch'] + 1
    else:
        logger.log(generator)
        logger.log(f"# Params - G: {count_parameters(generator)}")
        logger.log(options)
        P.starting_step = 1
    logger.log(f"Use G moving average: {P.accum}")  
    P.augment_fn = get_augment(mode=P.aug).cuda()   
    generator = nn.DataParallel(generator)
    g_ema = nn.DataParallel(g_ema)
    generator.sample_latent = generator.module.sample_latent

    train(P,options,P.train_fn,
            models=(generator,g_ema),
            optimizers=(G_optimizer),
            pair_loader=pair_loader,val_pair_loader=val_pair_loader,logger=logger)


if __name__ == '__main__':
    P = parse_args()
    if P.comment:
        P.comment = '_' + P.comment
    P.gin_stem = Path(P.gin_config).stem
    P = setup(P)
    P.distributed = False
    worker(P)


                    






