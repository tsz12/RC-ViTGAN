from vit_generator_skip import vit_small,vit_my,vit_my_8
import torch
generator = vit_my_8(patch_size=16)
g_ema = vit_my_8(patch_size=16)
model_path="logs/gan_dp/c10_style64/vitgan/aug_both_diffaug_bcr_R0.1_H1000_NoLazy_NoWarmup/3347/gen_90000_stage2.pt"
print(f"checkpoint的文件路径为:{model_path}")
checkpoint = torch.load(model_path)
generator.load_state_dict(checkpoint)#加载模型参数
# 这里与模型文件参数的区别在于：这是针对代码中创建的模型对象，查看其各个layer的名称与tensor值
# 获取模型中所有layer的名称
layer_name = list(generator.state_dict().keys())

length=len(layer_name)
# 查看指定layer的tensor值
not_zero=0
for i in range(0,length):
    layer=generator.state_dict()[layer_name[i]]
    #print(f"{layer_name[i]}为layer")
    a=layer.shape
    tmp=torch.zeros(a)
    #'''判断两个tensor是否相等'''
    if "style." in layer_name[i]:
        print(f"{layer_name[i]}为{layer}")
    if not torch.equal(tmp,layer):
        not_zero=not_zero+1
        #print(layer_name[i])
    if torch.equal(tmp,layer):
        print(f"{layer_name[i]}权重为0")
#print(f"layer_name为{len(layer_name)}")
print(f"非0的层数为{not_zero}")