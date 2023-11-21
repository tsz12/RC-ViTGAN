# RC-ViTGAN
![Overview of RC-ViTGAN](/assets/fig2.png "Overview of RC-ViTGAN")
## Dataset
RC500: https://drive.google.com/file/d/1x3tZmPw0IS9fxoKXel4xGT38nK8zd9hR/view?usp=sharing  

test_dataset: https://drive.google.com/file/d/1Y3_4tuNcGbzj5bYr2JWhniVvm6W6J-Gb/view?usp=drive_link  

original_images: https://drive.google.com/file/d/1i9hv2yrG8cImo7In3KUiFnusn7GNC-7e/view?usp=drive_link  
## Experimental details
### Environment
In this project, we use python 3.7.12 and pytorch 1.8.0, torchvision 0.9.0, cuda 11.1
### Hardware conditions
We train the model using four GeForce RTX 3060. 
### Hyperparameters
bs = 8  

lr= 0.0001  

beta for EMA = (0.0, 0.99)  

Supervised Pre-training max_steps=100000  

Adversarial Training max_steps=20000
## Results
### Qualitative Results of Ablation Study
![Qualitative Results of Ablation Study](/assets/fig6.png "Qualitative Results of Ablation Study")
