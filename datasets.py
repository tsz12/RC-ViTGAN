import os

from torchvision import datasets, transforms#torchvision里提供的数据集
import torch
import pathlib
import numpy as np
import cv2 
from skimage.color import rgb2lab, lab2rgb
import matplotlib
import pickle
DATA_PATH = os.environ.get('DATA_DIR', 'data/')
# from torch.utils.data import Dataset
# train_dir = "../data/unlabeled_data/train"
# var_dir = "../data/unlabeled_data/test"

import glob
import random
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.io import read_image

class MyImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir,transform=None,target_transform=None):
        self.img_labels=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform


    def __getitem__(self, idx):
        img_path=os.path.join(self.img_dir,str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        image=Image.open(img_path).convert('RGB')
        label=self.img_labels.iloc[idx,1]
        train_image=self.transform(image)
        target_image=self.target_transform(image)           


        return train_image, target_image

    def __len__(self):
        return len(self.img_labels)   

class MyLabeledImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir,transform=None,target_transform=None):
        self.img_labels=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform


    def __getitem__(self, idx):
        img_path_train=os.path.join(self.img_dir,'train',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        img_path_target=os.path.join(self.img_dir,'target',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        train_image=Image.open(img_path_train).convert('RGB')
        target_image=Image.open(img_path_target).convert('RGB')
        label=self.img_labels.iloc[idx,1]
        train_image=self.transform(train_image)
        target_image=self.target_transform(target_image)           


        return train_image, target_image

    def __len__(self):
        return len(self.img_labels)   

class MyGANImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir,transform=None,target_transform=None,gan_transform=None):
        self.img_labels=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
        self.gan_transform=gan_transform


    def __getitem__(self, idx):
        img_path_train=os.path.join(self.img_dir,'train',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        img_path_target=os.path.join(self.img_dir,'target',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        train_image=Image.open(img_path_train).convert('RGB')
        target_image=Image.open(img_path_target).convert('RGB')
        label=self.img_labels.iloc[idx,1]
        t_train_image=self.transform(train_image)
        target_image=self.target_transform(target_image)    
        gan_image=self.gan_transform(train_image)           


        return t_train_image, gan_image,target_image

    def __len__(self):
        return len(self.img_labels)  
    
def get_illuminance(img):
    """
    Get the luminance of an image. Shape: (h, w)
    """
    img = img.permute(1, 2, 0)  # (h, w, channel) 
    img = img.numpy()
    img = img.astype(np.float) / 255.0
    img_LAB = rgb2lab(img)
    img_L = img_LAB[:,:,0]  # luminance  # (h, w)
    return torch.from_numpy(img_L)
   
class MyColorTransferImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir,transform=None,target_transform=None,illu_transform=None,cho_illu_transform=None):
        self.img_labels=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
        self.illu_transform=illu_transform
        self.cho_illu_transform=cho_illu_transform
 

    def __getitem__(self, idx):
        img_path_train=os.path.join(self.img_dir,str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        img_path_target=os.path.join(self.img_dir,str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        palette_path=os.path.join(self.img_dir,'palette/',str(self.img_labels.iloc[idx,1]))

        train_image_cv2=cv2.imread(img_path_train)
        target_image_cv2=cv2.imread(img_path_train)
        train_image=self.transform(cv2.cvtColor(train_image_cv2,cv2.COLOR_BGR2RGB))
        target_image=self.target_transform(cv2.cvtColor(target_image_cv2,cv2.COLOR_BGR2RGB))
        illu=get_illuminance(self.illu_transform(cv2.cvtColor(target_image_cv2,cv2.COLOR_BGR2RGB)))
        cho_illu=get_illuminance(self.cho_illu_transform(cv2.cvtColor(target_image_cv2,cv2.COLOR_BGR2RGB)))
        
        from pathlib import Path
        f = Path(str(self.img_labels.iloc[idx,0])) 
        palette = pickle.load(open(str(palette_path)+'/'+f.stem +'.pkl', 'rb'))
        palette = palette[:, :6, :].ravel() / 255.0

        train_image=train_image.float()
        target_image=target_image.float()
        illu=illu.float()
        palette = torch.from_numpy(palette).float()

        return train_image, target_image,illu,cho_illu,palette

    def __len__(self):
        return len(self.img_labels) 

class MyColorTransferImageDataset1(Dataset):
    def __init__(self, annotations_file, img_dir,transform=None,target_transform=None):
        self.img_labels=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
 

    def __getitem__(self, idx):
        img_path_train=os.path.join(self.img_dir,'train/',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        img_path_target=os.path.join(self.img_dir,'presudo_target/',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        #print(f"img_path_train为{img_path_train}")
        #print(f"img_path_target为{img_path_target}")
        train_image_cv2=cv2.imread(img_path_train)
        target_image_cv2=cv2.imread(img_path_target)
        train_image=self.transform(cv2.cvtColor(train_image_cv2,cv2.COLOR_BGR2RGB))
        target_image=self.target_transform(cv2.cvtColor(target_image_cv2,cv2.COLOR_BGR2RGB))
        #illu=self.target_transform(cv2.cvtColor(train_image_cv2,cv2.COLOR_BGR2RGB))
        illu=get_illuminance(target_image)

        train_image=train_image.float()
        target_image=target_image.float()
        illu=illu.float()
        return train_image, target_image,illu

    def __len__(self):
        return len(self.img_labels)   

class MyColorTransferImageDataset2(Dataset):
    def __init__(self, annotations_file, img_dir,transform=None,target_transform=None):
        self.img_labels=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
 

    def __getitem__(self, idx):
        img_path_train=os.path.join(self.img_dir,'train/',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        img_path_target=os.path.join(self.img_dir,'presudo_target/',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        #print(f"img_path_train为{img_path_train}")
        #print(f"img_path_target为{img_path_target}")
        train_image_cv2=cv2.imread(img_path_train)
        target_image_cv2=cv2.imread(img_path_target)
        train_image=self.transform(cv2.cvtColor(train_image_cv2,cv2.COLOR_BGR2RGB))
        target_image=self.target_transform(cv2.cvtColor(target_image_cv2,cv2.COLOR_BGR2RGB))
        real_image=self.target_transform(cv2.cvtColor(train_image_cv2,cv2.COLOR_BGR2RGB))
        illu=get_illuminance(target_image)

        train_image=train_image.float()
        target_image=target_image.float()
        illu=illu.float()
        real_image=real_image.float()
        return train_image, target_image,illu,real_image

    def __len__(self):
        return len(self.img_labels)   
    
class MyColorTransferImageDataset3(Dataset):
    def __init__(self, annotations_file, img_dir,transform=None,target_transform=None):
        self.img_labels=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
 

    def __getitem__(self, idx):
        img_path_train=os.path.join(self.img_dir,'train/',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        img_path_target=os.path.join(self.img_dir,'target/',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        #print(f"img_path_train为{img_path_train}")
        #print(f"img_path_target为{img_path_target}")
        train_image_cv2=cv2.imread(img_path_train)
        target_image_cv2=cv2.imread(img_path_target)
        train_image=self.transform(cv2.cvtColor(train_image_cv2,cv2.COLOR_BGR2RGB))
        target_image=self.target_transform(cv2.cvtColor(target_image_cv2,cv2.COLOR_BGR2RGB))
        real_image=self.target_transform(cv2.cvtColor(train_image_cv2,cv2.COLOR_BGR2RGB))
        illu=get_illuminance(target_image)

        train_image=train_image.float()
        target_image=target_image.float()
        illu=illu.float()
        real_image=real_image.float()
        return train_image, target_image,illu,real_image

    def __len__(self):
        return len(self.img_labels)   

class MyColorTransferImageDatasetval(Dataset):
    def __init__(self, annotations_file, img_dir,transform=None,target_transform=None):
        self.img_labels=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
 

    def __getitem__(self, idx):
        img_path_train=os.path.join(self.img_dir,'val/',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        img_path_target=os.path.join(self.img_dir,'val_target/',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        #print(f"img_path_train为{img_path_train}")
        #print(f"img_path_target为{img_path_target}")
        train_image_cv2=cv2.imread(img_path_train)
        target_image_cv2=cv2.imread(img_path_target)
        train_image=self.transform(cv2.cvtColor(train_image_cv2,cv2.COLOR_BGR2RGB))
        target_image=self.target_transform(cv2.cvtColor(target_image_cv2,cv2.COLOR_BGR2RGB))
        #illu=self.target_transform(cv2.cvtColor(train_image_cv2,cv2.COLOR_BGR2RGB))
        illu=get_illuminance(target_image)

        train_image=train_image.float()
        target_image=target_image.float()
        illu=illu.float()
        return train_image, target_image,illu

    def __len__(self):
        return len(self.img_labels)   
    
class MyColorTransferImageDatasetvalstage3(Dataset):
    def __init__(self, annotations_file, img_dir,transform=None,target_transform=None):
        self.img_labels=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
 

    def __getitem__(self, idx):
        img_path_train=os.path.join(self.img_dir,'val/',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        img_path_target=os.path.join(self.img_dir,'val_target/',str(self.img_labels.iloc[idx,1]),self.img_labels.iloc[idx,0])
        #print(f"img_path_train为{img_path_train}")
        #print(f"img_path_target为{img_path_target}")
        train_image_cv2=cv2.imread(img_path_train)
        target_image_cv2=cv2.imread(img_path_target)
        train_image=self.transform(cv2.cvtColor(train_image_cv2,cv2.COLOR_BGR2RGB))
        train128_image=self.target_transform(cv2.cvtColor(train_image_cv2,cv2.COLOR_BGR2RGB))
        target_image=self.target_transform(cv2.cvtColor(target_image_cv2,cv2.COLOR_BGR2RGB))
        #illu=self.target_transform(cv2.cvtColor(train_image_cv2,cv2.COLOR_BGR2RGB))
        illu=get_illuminance(target_image)

        train_image=train_image.float()
        train128_image=train128_image.float()
        target_image=target_image.float()
        illu=illu.float()
        return train_image,train128_image,target_image,illu

    def __len__(self):
        return len(self.img_labels)   

def get_dataset(dataset):
    if dataset == 'labeled_data':
        image_size = (224, 224, 3)
        train_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])

        train_dir = os.path.join(DATA_PATH, 'labeled_data/train')
        val_dir = os.path.join(DATA_PATH, 'labeled_data/val')
        target_dir = os.path.join(DATA_PATH, 'labeled_data/target')
        img_dir=os.path.join(DATA_PATH, 'labeled_data/')
        annotations_file="img_l.csv"
        train_set = datasets.ImageFolder(train_dir, train_transform)
        target_set = datasets.ImageFolder(target_dir, target_transform)
        val_set = datasets.ImageFolder(val_dir, test_transform)
        pair_set=MyLabeledImageDataset(annotations_file, img_dir,transform=train_transform,target_transform=target_transform)

        return train_set, val_set, target_set,pair_set,image_size

    elif dataset=='unlabeled_data1':
        image_size = (224, 224, 3)
        train_transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
            transforms.Resize([128,128]),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(DATA_PATH, 'unlabeled_data/train/')
        val_dir = os.path.join(DATA_PATH, 'unlabeled_data/val/')
        annotations_file="img.csv"
        train_set = datasets.ImageFolder(train_dir, train_transform)
        target_set=datasets.ImageFolder(train_dir, target_transform)
        val_set = datasets.ImageFolder(val_dir, test_transform)
        pair_set=MyImageDataset(annotations_file, train_dir,transform=train_transform,target_transform=target_transform)
        

        return train_set, val_set,target_set,pair_set,image_size
    
    elif dataset=='unlabeled_data1_LAB':
        image_size = (224, 224, 3)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([128,128]),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(DATA_PATH, 'unlabeled_data/train/')

        annotations_file="img.csv"

        pair_set=MyColorTransferImageDataset(annotations_file, train_dir,transform=train_transform,target_transform=target_transform)
        

        return pair_set,image_size
    
    elif dataset=='unlabeled_data1_LAB_presudo':
        image_size = (128, 128, 3)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([128,128]),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(DATA_PATH, 'unlabeled_data/')

        annotations_file="img.csv"

        pair_set=MyColorTransferImageDataset1(annotations_file, train_dir,transform=train_transform,target_transform=target_transform)
        

        return pair_set,image_size
    
    elif dataset=='unlabeled_data1_LAB_presudo_stage3':
        image_size = (128, 128, 3)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([128,128]),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(DATA_PATH, 'unlabeled_data/')

        annotations_file="img.csv"

        pair_set=MyColorTransferImageDataset2(annotations_file, train_dir,transform=train_transform,target_transform=target_transform)
        

        return pair_set,image_size
    elif dataset=='semi_labeled_data':
        image_size = (128, 128, 3)
        gan_transform = transforms.Compose([
            transforms.Resize([128,128]),
            transforms.ToTensor()
        ])
        en_zero_transform= transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor()
        ])
        de_zero_transform= transforms.Compose([
            transforms.Resize([128,128]),
            transforms.ToTensor()
        ])

        img_dir_l_train=os.path.join(DATA_PATH, 'labeled_data/train/')
        img_dir_l_target=os.path.join(DATA_PATH, 'labeled_data/target/')
        img_dir_u_train=os.path.join(DATA_PATH, 'unlabeled_data/train/')
        annotations_file_l="img_l.csv"
        annotations_file_u="img.csv"
        ltrain_pair_set=MyImageDataset(annotations_file_l, img_dir_l_train,transform=en_zero_transform,target_transform=de_zero_transform)
        ltarget_pair_set=MyImageDataset(annotations_file_l, img_dir_l_target,transform=en_zero_transform,target_transform=de_zero_transform)
        utrain_pair_set=MyImageDataset(annotations_file_u, img_dir_u_train,transform=en_zero_transform,target_transform=de_zero_transform)

        img_dir_l=os.path.join(DATA_PATH, 'labeled_data/')
        ltrain_ltarget_pair_set=MyLabeledImageDataset(annotations_file_l, img_dir_l,transform=en_zero_transform,target_transform=de_zero_transform)
        ltrain_lgan_ltarget_pair_set=MyGANImageDataset(annotations_file_l, img_dir_l,transform=en_zero_transform,target_transform=de_zero_transform,gan_transform=gan_transform)
        unlabeled_set=datasets.ImageFolder(img_dir_u_train, en_zero_transform)
     

        return ltrain_pair_set,ltarget_pair_set,utrain_pair_set,ltrain_lgan_ltarget_pair_set,unlabeled_set,image_size
    
    elif dataset=='labeled_data_stage3':
        image_size = (128, 128, 3)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([128,128]),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(DATA_PATH, 'labeled_data/')

        annotations_file="img_l.csv"
        ltrain_lgan_ltarget_pair_set=MyColorTransferImageDataset3(annotations_file, train_dir,transform=train_transform,target_transform=target_transform)
        return ltrain_lgan_ltarget_pair_set,image_size

    elif dataset=='test_data':
        
        en_train1_transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
        en_train23_transform= transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor()
        ])

        annotations_file_test='img_test.csv'
        img_dir_test=os.path.join(DATA_PATH, 'test_data/')
        test_train1_train23_pair_set=MyImageDataset(annotations_file_test, img_dir_test,transform=en_train1_transform,target_transform=en_train23_transform)
        
        return test_train1_train23_pair_set
    
    elif dataset=='test_data_LAB':
        
        en_train1_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224,224]),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
        en_train23_transform= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224,224]),
            transforms.ToTensor()
        ])

        annotations_file_test='img_test.csv'
        img_dir_test=os.path.join(DATA_PATH, 'test_data/')
        test_train1_train23_ill_set=pair_set=MyColorTransferImageDataset(annotations_file_test, img_dir_test,transform=en_train1_transform,target_transform=en_train23_transform)

        return test_train1_train23_ill_set

    elif dataset=='test_data_LAB_palette':
        
        en_train1_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([432, 288]),
            transforms.ToTensor()
        ])
        en_train23_transform= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
        illu_transform= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([128, 128]),
            transforms.ToTensor()
        ])
        cho_illu_transform= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([432, 288]),
            transforms.ToTensor()
        ])

        annotations_file_test='img_test.csv'
        img_dir_test=os.path.join(DATA_PATH, 'test_data/')
        test_train1_train23_ill_set=pair_set=MyColorTransferImageDataset(annotations_file_test, img_dir_test,transform=en_train1_transform,target_transform=en_train23_transform,
                                illu_transform=illu_transform,cho_illu_transform=cho_illu_transform)

        return test_train1_train23_ill_set

    elif dataset=='val_data':  
        image_size = (128, 128, 3)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([128,128]),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(DATA_PATH, 'unlabeled_data/')

        annotations_file="img_val.csv"

        pair_set=MyColorTransferImageDatasetval(annotations_file, train_dir,transform=train_transform,target_transform=target_transform)
        #pair_set=train_image, target_image,illu

        return pair_set,image_size
    elif dataset=='val_data_stage3':  
        image_size = (128, 128, 3)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([128,128]),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(DATA_PATH, 'labeled_data/')

        annotations_file="labeled_img_val.csv"

        pair_set=MyColorTransferImageDatasetvalstage3(annotations_file, train_dir,transform=train_transform,target_transform=target_transform)
        #pair_set=train_image,train128_image,target_image,illu

        return pair_set,image_size

    elif dataset=='label_fid':
        image_size = (128, 128, 3)
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        train_dir = os.path.join(DATA_PATH, 'labeled_data/target/')
        label_set=datasets.ImageFolder(train_dir,transform)
        return label_set
    elif dataset=='original_fid':
        image_size = (128, 128, 3)
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        train_dir = os.path.join(DATA_PATH, 'unlabeled_data/train/')
        label_set=datasets.ImageFolder(train_dir,transform)
        return label_set
