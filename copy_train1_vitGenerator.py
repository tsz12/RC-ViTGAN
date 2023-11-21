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
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# from tensorboardX import SummaryWriter
# writer = SummaryWriter(log_dir='log')
# from torchsummary import summary
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
    parser.add_argument("--lbd_r1", type=float, default=10)
    parser.add_argument('--style_mix', default=0.9, type=float, help='Style mixing regularization')
    parser.add_argument('--halflife_k', default=20, type=int,
                        help='Half-life of exponential moving average in thousands of images')
    parser.add_argument('--ema_start_k', default=None, type=int,
                        help='When to start the exponential moving average of G (default: halflife_k)')
    parser.add_argument('--halflife_lr', default=0, type=int, help='Apply LR decay when > 0')

    parser.add_argument('--use_nerf_proj', action='store_true')

    # Options for logging specification
    parser.add_argument('--no_fid', action='store_true',
                        help='Do not track FIDs during training')
    parser.add_argument('--no_gif', action='store_true',
                        help='Do not save GIF of sample generations from a fixed latent periodically during training')
    parser.add_argument('--n_eval_avg', default=3, type=int,
                        help='How many times to average FID and IS')
    parser.add_argument('--print_every', help='', default=1000, type=int)
    parser.add_argument('--evaluate_every', help='', default=2000, type=int)
    parser.add_argument('--save_every', help='', default=10000, type=int)
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
        #ratio = (cur_step * batch_size) / halflife_lr
        ratio=cur_step/1000
        lr_mul = 0.5 ** ratio
        lr_w = lr_mul * lr * mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_w
        return lr_w
    return None

def _sample_generator(G, num_samples, enable_grad=True,imgs=None,illu=None):
    latent_samples = G.sample_latent(num_samples)
    if enable_grad:
        generated_data = G(x=imgs,input=latent_samples,illu=illu)
        #print(summary(G, imgs.shape,latent_samples.shape))
    else:
        with torch.no_grad():
            generated_data = G(x=imgs,input=latent_samples,illu=illu)
            #print(summary(G, imgs.shape,latent_samples.shape))
    return generated_data

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

def train(P,opt,train_fn,models,optimizers,pair_loader,logger):
    generator,g_ema=models
    opt_G=optimizers
    losses={'G_train1_loss':[],'G_train1_AB_loss':[]}

    logger.log_dirname("Steps {}".format(P.starting_step))

    for param_group in opt_G.param_groups:
            param_group['lr'] = opt["lr"]
            print(f"lr为{param_group['lr']}")  

    if (not P.use_warmup) and P.halflife_lr==0 and P.resume:
        for param_group in opt_G.param_groups:
            param_group['lr'] = opt["lr"]   

    for step in range(P.starting_step, opt['max_steps'] + 1):
        #print(f"运行到了{step}")
        if P.use_warmup:
            _update_warmup(opt_G, step, opt["warmup"], opt["lr"])
        if (not P.use_warmup) or step > opt["warmup"]:
            cur_lr_g = _update_lr(opt_G, step, opt["batch_size"], P.halflife_lr, opt["lr"])
            if cur_lr_g:
                logger.log('LR Updated: [G %.5f] ' % (cur_lr_g))
         

        do_ema = (step * opt['batch_size']) > (P.ema_start_k * 1000)
        accum = P.accum if do_ema else 0
        accumulate(g_ema, generator, accum)
        generator.train()
        # images, labels = next(train_loader)
        # target_images, labels = next(target_loader)
        images,target_images,illus=next(pair_loader)
        # print(f"images:{target_images[0]}")
        # images,target_images=next(pair_loader)
        # print(f"images:{target_images[0]}")
        # images,target_images=next(pair_loader)
        # print(f"images:{target_images[0]}")

        # print(f"images shape:{images.size()}")
        # print(f"target_images shape:{target_images.size()}")        
        #images, labels,target_images, labels=next(zip(train_loader,target_loader))
        #pair_images = next(pair_loader)

        images = images.cuda()
        target_images=target_images.cuda()
        illus=illus.cuda()
        #pair_images=pair_images.cuda()
        #target_images=target_images.cuda()
        set_grad(generator, True)
        gen_images = _sample_generator(generator, images.size(0),enable_grad=True,imgs=images,illu=illus)
        g_loss = train_fn["train2"](P, opt, target_images,gen_images)
        #g_AB_loss=train_fn["train1_AB"](P, opt, target_images,gen_images)
        #g_loss = train_fn["train1"](P, opt, pair_images[1][0],gen_images)
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()
        losses['G_train1_loss'].append(g_loss.item())
        #losses['G_train1_AB_loss'].append(g_AB_loss.item())
        generator.eval()
        # print(f"images.shape为{images.shape}")
        # print(summary(generator, (3,128,128),(3,128,128),batchsize=84))
        if step % P.print_every == 0:
            logger.log('[Steps %7d] [G %.3f]' %(step, losses['G_train1_loss'][-1]))
            for name in losses:
                values = losses[name]
                if len(values) > 0:
                    logger.scalar_summary('train/train1' + name, values[-1], step)#logger.scalar_summary(tag, value, idx)
            #wandb.log({"G_train1_loss": losses['G_train1_loss'][-1]}, step=step)
        if step % P.evaluate_every == 0:
            logger.log_dirname("Steps {}".format(step + 1))
            G_state_dict = generator.module.state_dict()
            Ge_state_dict = g_ema.state_dict()
            #torch.save(generator,logger.logdir + '/gen_all.pt')
            torch.save(G_state_dict, logger.logdir + '/gen_stage2.pt')
            torch.save(Ge_state_dict, logger.logdir + '/gen_ema_stage2.pt')
            if step % P.save_every == 0:
                torch.save(G_state_dict, logger.logdir + f'/gen_stage2_{step}.pt')
                torch.save(Ge_state_dict, logger.logdir + f'/gen_ema_stage2_{step}.pt')
                torch.save({'epoch': step,'optim_G': opt_G.state_dict(),}, logger.logdir + f'/optim_stage2_{step}.pt')
            torch.save({
                'epoch': step,
                'optim_G': opt_G.state_dict(),
            }, logger.logdir + '/optim_stage2.pt')

def worker(P):
    gin.parse_config_files_and_bindings(['configs/defaults/gan.gin',
                                         'configs/defaults/augment.gin',
                                         P.gin_config], [])    

    options = get_options_dict()
    #train_set, _, target_set,image_size = get_dataset(dataset=options['labeled_data'])
    #print(f"get_dataset(dataset=options['dataset'])为{get_dataset(dataset=options['dataset'])}")
    pair_set,resolution = get_dataset(dataset=options['dataset'])
    print(f"resolution为{resolution}")
    
    #print(f"pair_set为{train_set.size()}")
    seed=10
    torch.manual_seed(seed) 
    # train_loader = DataLoader(train_set, shuffle=True, pin_memory=False, num_workers=P.workers,
    #                            batch_size=options['batch_size'], drop_last=True)
    # target_loader = DataLoader(target_set, shuffle=True, pin_memory=False, num_workers=P.workers,
    #                            batch_size=options['batch_size'], drop_last=True)
    pair_loader = DataLoader(pair_set, shuffle=True, pin_memory=False, num_workers=P.workers,
                              batch_size=options['batch_size'], drop_last=True)
    # images,target_images=next(iter(pair_loader))
    # train_loader = cycle(train_loader)
    # target_loader = cycle(target_loader)
    pair_loader = cycle3(pair_loader)
 

    if P.ema_start_k is None:
        P.ema_start_k = P.halflife_k

    P.accum = 0.5 ** (options['batch_size'] / (P.halflife_k * 1000))

    from vit_generator_skip import vit_small,vit_my,vit_my_8
    #resolution = image_size[0]
    generator = vit_my_8(patch_size=16,noise=False)
    g_ema = vit_my_8(patch_size=16,noise=False)
    count=0
    for name, param in generator.named_parameters():
        count=count+1
    print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
    #url = None
    url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth" # model used for visualizations in our paper
    if url is not None:
        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        info_generator=generator.load_state_dict(state_dict, strict=False)
        info_g_ema=g_ema.load_state_dict(state_dict, strict=False)
    else:
        print("There is no reference weights available for this model => We use random weights.")
        print(f"=> Loading checkpoint from '{P.resume}'")
        state_G = torch.load(f"{P.resume}/gen_stage1_10000.pt")
        state_Ge = torch.load(f"{P.resume}/gen_ema_stage1_10000.pt")
        # generator_dict=generator.state_dict()
        # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in generator_dict  and v.shape ==generator_dict[k].shape}
        # generator_dict.update(pretrained_dict)
        # generator.load_state_dict(generator_dict)
        # g_ema.load_state_dict(generator_dict)
        info_generator=generator.load_state_dict(state_G,strict=False)
        info_g_ema=g_ema.load_state_dict(state_Ge,strict=False)  
        

    generator = generator.cuda()
    g_ema = g_ema.cuda()
    g_ema.eval()


    for name, param in generator.named_parameters():
        param.requires_grad=True
    #G_optimizer = optim.Adam(generator.parameters(),lr=options["lr"], betas=options["beta"])
    G_optimizer = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()),lr=options["lr"], betas=options["beta"]) 
    if P.resume:
        _desc = f"R{P.lbd_r1}_H{P.halflife_k}"
        if P.halflife_lr > 0:
            _desc += f"_lr{P.halflife_lr / 1000000:.1f}M"
        _desc += f"_NoLazy" if P.no_lazy else "_Lazy"
        _desc += f"_Warmup" if P.use_warmup else "_NoWarmup"
        logger = Logger(None, resume=P.resume)  
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
        
        # wandb.config.update(P)
        # wandb.config.update(options)

        # wandb.watch(generator)
        # #wandb.watch(discriminator)
        shutil.copy2(P.gin_config, f"{logger.logdir}/config.gin")
    P.logdir = logger.logdir
    P.eval_seed = np.random.randint(10000)

    if P.resume:
        opt = torch.load(f"{P.resume}/optim_stage1_10000.pt")
        G_optimizer.load_state_dict(opt['optim_G'])
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
    generator.sample_latent = generator.module.sample_latent
    train(P,options,P.train_fn,
            models=(generator,g_ema),
            optimizers=(G_optimizer),
            pair_loader=pair_loader,logger=logger)


if __name__ == '__main__':
    P = parse_args()
    if P.comment:
        P.comment = '_' + P.comment
    P.gin_stem = Path(P.gin_config).stem
    P = setup(P)
    P.distributed = False
    worker(P)


                    






