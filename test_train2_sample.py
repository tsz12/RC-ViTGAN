from argparse import ArgumentParser
from pathlib import Path
import os
import math

import gin
import torch
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

from datasets import get_dataset
#from models.gan import get_architecture
from torch.utils.data import DataLoader
from training.gan import setup

from torchviz import make_dot
from color_harmonization import ch_loss
from torchvision import transforms
#from PaletteNet_PyTorch.palettenet import viz_image_ori_new_out
import cv2
from evaluate.gan import FIDScore, FixedSampleGeneration, ImageGrid
from inception_score_pytorch import inception_score
# import for gin binding
import penalty
import augment
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = ArgumentParser(description='Testing script: Random sampling from G')
    parser.add_argument('model_path', type=str, help='Path to the (generator) model checkpoint')
    #parser.add_argument('architecture', type=str, help='Architecture')

    parser.add_argument('--n_samples', default=10, type=int,
                        help='Number of samples to generate (default: 10000)')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size (default: 500)')
    #下面这2行是我加的！！
    parser.add_argument('--use_nerf_proj', action='store_true', help='Use warmup strategy on LR')
    parser.add_argument('--workers', default=8, type=int, metavar='N',help='number of data loading workers (default: 0)')
    return parser.parse_args()




def _sample_generator(G, num_samples, enable_grad=True,imgs=None,illus=None):
    latent_samples = G.sample_latent(num_samples)
    #latent_samples=torch.ones(32,384)
    #latent_samples=-torch.full((24,384),-3.0)  
    #print(f"latent_samples的大小为{latent_samples.shape}")#latent_samples的大小为torch.Size([12, 384]
    if enable_grad:
        generated_data = G(x=imgs,input=latent_samples,illu=illus)
    else:
        with torch.no_grad():
            generated_data = G(x=imgs,input=latent_samples,illu=illus)
    return generated_data

# def _sample_generator(G, num_samples,imgs):
#     latent_samples = G.sample_latent(num_samples)
#     generated_data = G(latent_samples,imgs=imgs)
#     return generated_data

#用来解析gin文件，得到参数字典，并加上一些新的参数
@gin.configurable("options")
def get_options_dict(dataset=gin.REQUIRED,
                     loss=gin.REQUIRED,
                     batch_size=64, fid_size=10000,
                     max_steps=200000,
                     warmup=0,
                     n_critic=1,
                     lr=2e-4, lr_d=None, beta=(.5, .999),
                     lbd=10., lbd2=10.):
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


if __name__ == '__main__':
    #device=torch.device('cpu')
    P = parse_args()
    print(f"p解析出来为{P}")
    logdir = Path(P.model_path).parent#返回文件路径的父路径
    print(f"文件夹路径为:{logdir}")
    gin_config = sorted(logdir.glob("*.gin"))[0]
    gin.parse_config_files_and_bindings(['configs/defaults/gan.gin',
                                         'configs/defaults/augment.gin',
                                         gin_config], [])
    options = get_options_dict()
    print(f"options['dataset']为{options['dataset']}")

    ltrain_pair_set,ltarget_pair_set,utrain_pair_set,ltrain_lgan_ltarget_pair_set,unlabeled_set,image_size = get_dataset(dataset='semi_labeled_data')
    
    test_train1_train23_pair_set=get_dataset(dataset='test_data')
    #test_train1_train23_illu_set=get_dataset(dataset='test_data_LAB')#返回testdata的灰度图、相应原图和亮度通道
    test_train1_train23_illu_set=get_dataset(dataset='test_data_LAB_palette')
    train_set, val_set,target_set,pair_set,image_size=get_dataset(dataset='unlabeled_data1')
    unlabeled_data1_LAB_presudo_pair,resolution=get_dataset(dataset='unlabeled_data1_LAB_presudo')#返回unlabeldata的原图、相应伪标签和亮度通道
    val_pair_set,image_size=get_dataset(dataset='val_data')
    
    #-----加载模型结构---------
    #from vit_generator import vit_small,vit_my,vit_my_8
    from vit_generator_skip import vit_small,vit_my,vit_my_8
    #from vit_generator_skip_add import vit_small,vit_my,vit_my_8
    #resolution = image_size[0]
    generator = vit_my_8(patch_size=16)
    g_ema = vit_my_8(patch_size=16)
 
    print(f"checkpoint的文件路径为:{P.model_path}")
    checkpoint = torch.load(f"{P.model_path}")
    generator.load_state_dict(checkpoint)#加载模型参数
    # # 查看噪声是否能控制颜色
    # with torch.no_grad():
    #     for name, param in generator.named_parameters():
    #         if 'style.' in name:
    #             param.copy_(torch.zeros(param.shape))
    #             print(f"{name}修改后的值为{generator.state_dict()[name]}")


    # 这里与模型文件参数的区别在于：这是针对代码中创建的模型对象，查看其各个layer的名称与tensor值
    generator.to(device)
    import torch.nn as nn
    generator=nn.DataParallel(generator)
    generator.eval()
    generator.sample_latent = generator.module.sample_latent#这个应该是dataparallel的时候才需要
    
    #在这里加载测试数据集
    #----------------------------------------------------------
    # test_loader = DataLoader(ltrain_lgan_ltarget_pair_set, shuffle=False, pin_memory=True, num_workers=P.workers,
    #                          batch_size=P.n_samples, drop_last=True)
    # test_u_loader = DataLoader(utrain_pair_set, shuffle=False, pin_memory=True, num_workers=P.workers,
    #                          batch_size=P.n_samples, drop_last=True)
    unlabeled_1_loader = DataLoader(pair_set, shuffle=False, pin_memory=True, num_workers=P.workers,
                             batch_size=P.n_samples, drop_last=True)#无标签灰度图-原图 pair
    unlabeled_1_loader=iter(unlabeled_1_loader)

    unlabeled_data1_LAB_presudo_pair_loader = DataLoader(unlabeled_data1_LAB_presudo_pair, shuffle=False, pin_memory=True, num_workers=P.workers,
                             batch_size=P.n_samples, drop_last=True)#无标签原图-伪标签-亮度通道 pair
    unlabeled_data1_LAB_presudo_pair_loader=iter(unlabeled_data1_LAB_presudo_pair_loader)

    test_loader_all = DataLoader(test_train1_train23_pair_set, shuffle=False, pin_memory=True, num_workers=P.workers,
                             batch_size=P.n_samples, drop_last=True)#测试集灰度图-原图 pair
    test_loader_all=iter(test_loader_all)

    test_illu_loader_all = DataLoader(test_train1_train23_illu_set, shuffle=False, pin_memory=True, num_workers=P.workers,
                             batch_size=P.n_samples, drop_last=True)#测试集灰度图-相应原图-亮度通道 pair
    test_illu_loader_all=iter(test_illu_loader_all)

    val_loader= DataLoader(val_pair_set, shuffle=False, pin_memory=True, num_workers=P.workers,
                             batch_size=P.n_samples, drop_last=True)#测试集灰度图-相应原图-亮度通道 pair
    val_loader=iter(val_loader)   
    
    # #加载unlabel图片
    # test_1_images,target_1_images=next(unlabeled_1_loader)#无标签灰度图-原图 pair
    # train_image, target_image,illu=next(unlabeled_data1_LAB_presudo_pair_loader)#无标签原图-伪标签-亮度通道 pair

    #加载test图片
    # next(test_illu_loader_all)
    # next(test_illu_loader_all)
    # next(test_illu_loader_all)
    # next(test_illu_loader_all)
    # next(test_illu_loader_all)
    # next(test_illu_loader_all)
    # next(test_illu_loader_all)
    # next(test_illu_loader_all)
    # next(test_illu_loader_all)
    # next(test_illu_loader_all)
    # # next(test_illu_loader_all)
    # # next(test_illu_loader_all)

    test_stage1_images,test_stage23_images,test_illus,cho_test_illus,test_palettes=next(test_illu_loader_all)#测试集原图（432，288）-相应原图（224，224）-亮度通道 -调色盘pair
    val_train_image, val_target_image,val_illu=next(val_loader)
    # train_image=train_image.cuda()
    # target_image=target_image.cuda()
    # illu=illu.cuda()
    # illu=transforms.Resize([128,128])(illu)
    # illu=illu.unsqueeze(0)
    # illu=illu.permute(1,0,2,3)
    #illu=None



    #测试集
    test_stage1_images=test_stage1_images.cuda()
    test_stage23_images=test_stage23_images.cuda()
    test_illus=test_illus.cuda()
    cho_test_illus=cho_test_illus.cuda()
    test_palettes=test_palettes.cuda()
    #test_illus=transforms.Resize([128,128])(test_illus)
    #test_illus=test_illus.unsqueeze(0)
    #test_illus=test_illus.permute(1,0,2,3)
    #test_illus=None    

    #验证集
    val_train_image=val_train_image.cuda()
    val_target_image=val_target_image.cuda()
    val_illu=val_illu.cuda()
    # #用测试集进行测试stage1
    # subdir_path = logdir / f"samples_stage1TestImage14000_{np.random.randint(10000)}_n{P.n_samples}"
  
    # if not os.path.exists(subdir_path):
    #     os.mkdir(subdir_path)
    # print("Sampling in %s" % subdir_path)
    # with torch.no_grad():
    #     samples = _sample_generator(generator,  test_stage1_images.size(0),enable_grad=False,imgs= test_stage1_images,illus=illus)
    # from torchvision import transforms
    # for j in range(samples.size(0)):
    #         save_image(test_stage1_images[j], f"{subdir_path}/test_{j}.png")
    #         save_image(test_stage23_images[j], f"{subdir_path}/target_{j}.png")
    #         save_image(samples[j], f"{subdir_path}/{j}.png")

 
    # #用无标签集进行测试stage1
    # subdir_path = logdir / f"samples_stage1UnLabelImage50000_{np.random.randint(10000)}_n{P.n_samples}"
    # if not os.path.exists(subdir_path):
    #     os.mkdir(subdir_path)
    # print("Sampling in %s" % subdir_path)
    # with torch.no_grad():
    #     samples = _sample_generator(generator,  test_1_images.size(0),enable_grad=False,imgs= test_1_images)
    # from torchvision import transforms
    # for j in range(samples.size(0)):
    #         save_image(test_1_images[j], f"{subdir_path}/test_{j}.png")
    #         save_image(target_1_images[j], f"{subdir_path}/target_{j}.png")
    #         save_image(samples[j], f"{subdir_path}/{j}.png")    
    #         #save_image(transforms.Grayscale()(test_1_images[j]), f"{subdir_path}/gray_{j}.png")
    #         #save_image(test_stage23_images[j], f"{subdir_path}/target_{j}.png")
    #         #save_image(test_stage23_images[j]-transforms.Grayscale()(test_stage1_images[j]), f"{subdir_path}/target-gray_{j}.png")
    #         #save_image(samples[j]-transforms.Grayscale()(samples[j]), f"{subdir_path}/sample-gray_{j}.png")

    #用测试集进行测试stage2
    subdir_path = logdir / f"samples_stage3TestImage8000_Ds_{np.random.randint(10000)}_n{P.n_samples}"
    metrics={}
    #metrics['fid_score'] = FIDScore(opt['dataset'], opt['fid_size'], P.n_eval_avg)
  
    if not os.path.exists(subdir_path):
        os.mkdir(subdir_path)
    print("Sampling in %s" % subdir_path)
    with torch.no_grad():
        samples = _sample_generator(generator,  test_stage23_images.size(0),enable_grad=False,imgs= test_stage23_images,illus=test_illus)
        #score=inception_score(samples, cuda=True, batch_size=32, resize=True, splits=4)#batchsize要小于输入inception V3的图片数
    for j in range(samples.size(0)):
            save_image(test_stage23_images[j], f"{subdir_path}/test_{j}.png")
            #save_image(test_stage23_images[j], f"{subdir_path}/target_{j}.png")
            save_image(samples[j], f"{subdir_path}/{j}.png")

    # #用无标签集进行测试stage2
    # subdir_path = logdir / f"samples_stage3UnLabelImage39000_{np.random.randint(10000)}_n{P.n_samples}"
    # if not os.path.exists(subdir_path):
    #     os.mkdir(subdir_path)
    # print("Sampling in %s" % subdir_path)
    # with torch.no_grad():
    #     samples = _sample_generator(generator,  train_image.size(0),enable_grad=False,imgs= train_image,illus=illu)
    # #from torchvision import transforms 
    # for j in range(samples.size(0)):
    #         save_image(train_image[j], f"{subdir_path}/test_{j}.png")
    #         save_image(target_image[j], f"{subdir_path}/target_{j}.png")
    #         save_image(samples[j], f"{subdir_path}/{j}.png")    
    #         save_image(transforms.Grayscale()(test_1_images[j]), f"{subdir_path}/gray_{j}.png")
    #         save_image(test_stage23_images[j], f"{subdir_path}/target_{j}.png")
    #         save_image(test_stage23_images[j]-transforms.Grayscale()(test_stage1_images[j]), f"{subdir_path}/target-gray_{j}.png")
    #         save_image(samples[j]-transforms.Grayscale()(samples[j]), f"{subdir_path}/sample-gray_{j}.png")

    # #用验证数据集测试Stage2
    # subdir_path = logdir / f"samples_stage2valImage50000_{np.random.randint(10000)}_n{P.n_samples}"
    # if not os.path.exists(subdir_path):
    #     os.mkdir(subdir_path)
    # print("Sampling in %s" % subdir_path)
    # with torch.no_grad():
    #     samples = _sample_generator(generator,  val_train_image.size(0),enable_grad=False,imgs= val_train_image,illus=val_illu)
    # #from torchvision import transforms 
    # for j in range(samples.size(0)):
    #         save_image(val_train_image[j], f"{subdir_path}/val_{j}.png")
    #         save_image(samples[j], f"{subdir_path}/vaLG{j}.png")
    #         save_image(val_target_image[j], f"{subdir_path}/val_target{j}.png")
    # # -------用palettenet------生成
    # palettenet_pre_state = torch.load("/home/xsx/dino/PaletteNet_PyTorch/FE_RD.pth")
    # palettenet_adv_state = torch.load("/home/xsx/dino/PaletteNet_PyTorch/adv_FE_RD.pth")
    # from PaletteNet_PyTorch.palettenet import FeatureEncoder,RecoloringDecoder
    # FE = FeatureEncoder().float().to(device)
    # #FE = FeatureEncoder().float().cuda()
    # RD = RecoloringDecoder().float().to(device)
    # FE = nn.DataParallel(FE)
    # RD = nn.DataParallel(RD)
    # #RD = RecoloringDecoder().float().cuda()
    # # FE.load_state_dict(palettenet_pre_state['FE'])
    # # RD.load_state_dict(palettenet_pre_state['RD'])
    # FE.load_state_dict(palettenet_adv_state['FE'])
    # RD.load_state_dict(palettenet_adv_state['RD'])
    # FE=FE.eval()
    # RD=RD.eval()
    # transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((432, 288)),
    # transforms.ToTensor(),
    # ])
    # c1, c2, c3, c4=FE(test_stage1_images.float())
    # print(f"test_stage23_images的大小为{test_stage23_images.shape}")
    # bz,c,h,w=test_stage23_images.shape
    # # seed=10
    # # torch.manual_seed(seed)
    # #test_palettes
    # #target_palette=torch.rand(bz,18)
    # import random
    # changes=torch.rand(bz,3).cuda()
    # pos=random.randint(0, 5)*3
    # indexs=torch.LongTensor([pos,pos+1,pos+2]).unsqueeze(0).cuda()
    
    # for i in range(1,bz):
    #     pos=random.randint(0, 5)*3
    #     index=torch.LongTensor([pos,pos+1,pos+2]).unsqueeze(0).cuda()
    #     indexs=torch.cat([indexs,index], dim=0)
    # print(f"indexs的大小为{indexs.shape},indexs为{indexs}")
    
    # target_palette=test_palettes.scatter(1, indexs, changes)
    # flat_palette=target_palette.flatten()

    # out_image = RD.forward(c1, c2, c3, c4, flat_palette.float(), cho_test_illus.float())
    # for j in range(out_image.size(0)):
    #     result = []
    #     result_origin = []
    #     result_width = 200
    #     result_height_per_center = 80
    #     k=6
    #     n_channels=3
    #     centroids_origin=test_palettes[j].reshape(6,3)*255
    #     centroids=target_palette[j].reshape(6,3)*255
    #     centroids_origin=centroids_origin.cpu()
    #     centroids=centroids.cpu()

    #     #print(f"{j}的centroids为{centroids}[顺序为bgr]")
    #     # b=centroids[:,0]
    #     # g=centroids[:,1]
    #     # r=centroids[:,2]
    #     # centroids[:,0]=b
    #     # centroids[:,1]=g
    #     # centroids[:,2]=r
    #     for center_index in range(k):
    #         result.append(np.full((result_width * result_height_per_center, n_channels), centroids[center_index], dtype=int))
    #     result = np.array(result)
    #     result = result.reshape((result_height_per_center * k, result_width, n_channels))
    #     result=result.astype(np.uint8)

    #     for center_index in range(k):
    #         result_origin.append(np.full((result_width * result_height_per_center, n_channels), centroids_origin[center_index], dtype=int))
    #     result_origin = np.array(result_origin)
    #     result_origin = result_origin.reshape((result_height_per_center * k, result_width, n_channels))
    #     result_origin = result_origin.astype(np.uint8)

    #     #print(f"result的大小为{result.shape}")
    #     #cv2.imwrite(f'palette_result.jpg', result)
    #     cv2.imwrite(f"{subdir_path}/target_palette_{j}.png", result)
    #     #cv2.imwrite(f"{subdir_path}/origin_palette_{j}.png", result_origin)
    #     #save_image(result, f"{subdir_path}/target_palette_{j}.png")
    #     save_image(out_image[j], f"{subdir_path}/palettenet_{j}.png")





