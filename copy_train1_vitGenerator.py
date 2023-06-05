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


#writer就相当于一个日志，保存你要做图的所有信息。第二句就是在你的项目目录下建立一个文件夹log，存放画图用的文件。刚开始的时候是空的
# from tensorboardX import SummaryWriter
# writer = SummaryWriter(log_dir='log') #将tensorboard文件保存在哪里
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
                        help='Do not use lazy regularization')#lazy regulation每个16个mini-batch才进行一次R1正则化，这个应该不是不能收敛的理由
    parser.add_argument("--d_reg_every", type=int, default=16,
                        help='Interval of applying R1 when lazy regularization is used')
    parser.add_argument("--lbd_r1", type=float, default=10, help='R1 regularization这是用于discriminator训练的')
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
        ratio = min(1., (cur_step + 1) / (warmup + 1e-8))#相当于到warmup个epoch的时候lr刚好更新到设定的lr值
        lr_w = ratio * lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_w


def _update_lr(optimizer, cur_step, batch_size, halflife_lr, lr, mult=1.0):
    if halflife_lr > 0 and (cur_step > 0) and (cur_step % 10000 == 0):#每1000个epoch更新一次学习率
        #ratio = (cur_step * batch_size) / halflife_lr
        ratio=cur_step/1000
        lr_mul = 0.5 ** ratio
        lr_w = lr_mul * lr * mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_w
        print(f"[steps:{cur_step}]更新后的学习率为{lr_w}")
        return lr_w
    return None

def _sample_generator(G, num_samples, enable_grad=True,imgs=None,illu=None):
    latent_samples = G.sample_latent(num_samples)
    #print(f"latent_samples出现NaN了吗{torch.isnan(latent_samples).any()}为:{latent_samples}")
    #print(f"输入的x=imgs形状为:{imgs.shape}输入的latent_samples的形状为{latent_samples.shape}")
    if enable_grad:
        generated_data = G(x=imgs,input=latent_samples,illu=illu)
        #print(summary(G, imgs.shape,latent_samples.shape))
    else:
        with torch.no_grad():
            generated_data = G(x=imgs,input=latent_samples,illu=illu)
            #print(summary(G, imgs.shape,latent_samples.shape))
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

def train(P,opt,train_fn,models,optimizers,pair_loader,logger):
    #print(f"成功进入train函数")
    generator,g_ema=models
    opt_G=optimizers
    # for param_group in opt_G.param_groups:
    #     print(f"学习率为{param_group['lr']}")
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
         
        #实现ema（权重移动平均）：权重还是原来的权重，只是权重更新时采用的梯度不是当前梯度，而是梯度的滑动平均
        do_ema = (step * opt['batch_size']) > (P.ema_start_k * 1000)#过了一阵之后才采用滑动平均
        accum = P.accum if do_ema else 0#accum就是滑动平均中的decay
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
        #pair_images = next(pair_loader)#应该就是这里出了问题！！！可能元组的加载情况不同
        #print(f"成功加载pair_images，大小为{pair_images.size()}")
    #-----检查一下数据是否加载成功---------
        # for i in range(5):
        #     print(f"现在的时间是{time.time()}")
        #     images_PIL= transforms.ToPILImage()(images[i])
        #     images_PIL.save(f"./load_image/train_set/{step}_{i}.png")
        #     target_images_PIL= transforms.ToPILImage()(target_images[i])
        #     target_images_PIL.save(f"./load_image/target_set/{step}_{i}.png")           
        #     #target_images,labels=next(target_loader)
        images = images.cuda()#batch_size除以4
        target_images=target_images.cuda()#会不会一组图片对被分到两个不同的服务器上
        illus=illus.cuda()
        #pair_images=pair_images.cuda()
        # for i in range(84):
        #     print(f'images和target_images是否在同一个设备上{images.device==target_images.device}')
        #target_images=target_images.cuda()
        set_grad(generator, True)
        # for param_group in opt_G.param_groups:
        #     print(f"学习率为{param_group['lr']}")
        gen_images = _sample_generator(generator, images.size(0),enable_grad=True,imgs=images,illu=illus)
        #print(f"成功产生图片")
        g_loss = train_fn["train2"](P, opt, target_images,gen_images)
        #g_AB_loss=train_fn["train1_AB"](P, opt, target_images,gen_images)
        #print(f"g_loss出现NaN了吗{torch.isnan(g_loss).any()}")
        #gen_images = _sample_generator(generator, pair_images[0][0].size(0),enable_grad=True,imgs=pair_images[0][0])
        #print(f"成功计算g_loss")
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
            #-----查看模型更新梯度是否正常------
            # for name, parms in generator.named_parameters():
            #     if parms.requires_grad:
            #         print(f"{name}更新的梯度为{parms.grad}")
        #当前的训练情况（包括优化器的参数）是P.evaluate_every个epoch保存一次，每P.save_every个epoch额外保存一次模型参数
        if step % P.evaluate_every == 0:
            #-----各评价指标------
            logger.log_dirname("Steps {}".format(step + 1))
            G_state_dict = generator.module.state_dict()
            Ge_state_dict = g_ema.state_dict()
            #torch.save(generator,logger.logdir + '/gen_all.pt')
            torch.save(G_state_dict, logger.logdir + '/gen_stage2.pt')
            torch.save(Ge_state_dict, logger.logdir + '/gen_ema_stage2.pt')
            if step % P.save_every == 0:
                torch.save(G_state_dict, logger.logdir + f'/gen_stage2_{step}.pt')
                torch.save(Ge_state_dict, logger.logdir + f'/gen_ema_stage2_{step}.pt')
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
    #print(f"get_dataset(dataset=options['dataset'])为{get_dataset(dataset=options['dataset'])}")
    pair_set,resolution = get_dataset(dataset=options['dataset'])
    print(f"resolution为{resolution}")
    
    #print(f"pair_set为{train_set.size()}")
    #-----检查一下数据是否加载成功---------
    # for i in range(5):
    #     #print(f"现在的时间是{time.time()}")
    #     images_PIL= transforms.ToPILImage()(pair_set[i][0])
    #     images_PIL.save(f"./load_image/train_set/{i}.png")
    #     target_images_PIL= transforms.ToPILImage()(pair_set[i][1])
    #     target_images_PIL.save(f"./load_image/target_set/{i}.png")   
    seed=10
    torch.manual_seed(seed) 
    # train_loader = DataLoader(train_set, shuffle=True, pin_memory=False, num_workers=P.workers,
    #                            batch_size=options['batch_size'], drop_last=True)
    # target_loader = DataLoader(target_set, shuffle=True, pin_memory=False, num_workers=P.workers,
    #                            batch_size=options['batch_size'], drop_last=True)
    pair_loader = DataLoader(pair_set, shuffle=True, pin_memory=False, num_workers=P.workers,
                              batch_size=options['batch_size'], drop_last=True)
    # images,target_images=next(iter(pair_loader))
    # print(f"images的type为{images.size()},target_images的type为{target_images.size()}")
    # train_loader = cycle(train_loader)
    # target_loader = cycle(target_loader)
    pair_loader = cycle3(pair_loader)
    # -----检查一下数据是否加载成功---------
    # images, labels = next(train_loader)
    # target_images, labels = next(target_loader)
    # for i in range(20):
    #         #print(f"现在的时间是{time.time()}")
    #         #print(f"是否正确加载{train_set[i][0].equal(images[i][0])}")
    #         images_PIL= transforms.ToPILImage()(images[i])
    #         images_PIL.save(f"./load_image/train_set/worker_{i}.png")
    #         #print(f"是否正确加载{target_set[i][0].equal(target_images[i][0])}")
    #         target_images_PIL= transforms.ToPILImage()(target_images[i])
    #         target_images_PIL.save(f"./load_image/target_set/worker_{i}.png")  

    if P.ema_start_k is None:
        P.ema_start_k = P.halflife_k

    P.accum = 0.5 ** (options['batch_size'] / (P.halflife_k * 1000))#accum是由这俩决定的

    #-----加载模型结构---------
    from vit_generator_skip import vit_small,vit_my,vit_my_8
    #resolution = image_size[0]
    generator = vit_my_8(patch_size=16,noise=False)
    g_ema = vit_my_8(patch_size=16,noise=False)
    count=0
    for name, param in generator.named_parameters():
        count=count+1
    print(f"允许更新的参数为{count}")
    #print(f"模型加载成功")
    #-----加载dino的预训练参数---------
    print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
    #url = None
    url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth" # model used for visualizations in our paper
    if url is not None:
        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        info_generator=generator.load_state_dict(state_dict, strict=False)
        info_g_ema=g_ema.load_state_dict(state_dict, strict=False)
        print(f"generator的预训练参数加载结果为{info_generator}")
        print(f"g_ema的预训练参数加载结果为{info_g_ema}")
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
        print(f"generator的预训练参数加载结果为{info_generator}")
        print(f"g_ema的预训练参数加载结果为{info_g_ema}")       
        

    generator = generator.cuda()
    g_ema = g_ema.cuda()
    g_ema.eval()
    #-----查看各层的名字，将encoder部分的参数冻结---------
    # for name, param in generator.named_parameters():
    #     #print(f"层名分别为{name}")
    #     if "convs" in name or "to_rgb" in name or "norm." in name or "style" in name or "mlp_gamma" in name or "mlp_beta" in name or "blur.kernel" in name:
    #         param.requires_grad=True
    #     else:
    #         param.requires_grad=False
    # count=0
    # for name, param in generator.named_parameters():
    #     if param.requires_grad:
    #         count=count+1
    #         print(f"加载的参数为{name}")
    # print(f"加载的参数个数为{count}")
    #---------将模型写入tensorboard-------------
    # init_img = torch.zeros(84,3,128,128).cuda()   
    # writer.add_graph(generator, (init_img,init_img))


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
        # wandb.init(project='"train1_vitgenerator"', name=f'查看loss1为什么就是不下降', resume=True)
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
        # wandb.init(project='"train1_vitgenerator"', name=f'查看loss1为什么就是不下降')
        
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


                    






