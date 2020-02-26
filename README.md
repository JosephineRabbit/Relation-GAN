# RelationGAN
Source Code for Our Paper "When Relation Networks meet GANs: Relation GANs with Triplet Loss"

# Requirements:
* python3
* pytorch
* torchvision
* numpy
* scipy
* tensorflow-gpu

# Introduction
We provide PyTorch implementations for Relation GAN and some measuring tools.

## This code include:

| GAN loss        | name           | FID(Cifar10) |
| ------------- |:-------------:| -----:|
| WGAN-GP      | wgangp | 63.7±0.11 |
| LS_GAN      | ls_gan      | __14.9±0.11__|
| vanilla GAN | sgan      |   26.4±0.16 |
| Relativistic_GAN | rele      |   24.1±0.19 |
| Our | relu_mean     |    13.5±0.080 |


## Measuring tools
Frechet Inception Distance(https://github.com/mseitzer/pytorch-fid)
Inception Score (https://github.com/google/compare_gan)
Kernel Inception distance (https://github.com/google/compare_gan)
Multi-scale Structural Similarity for Image Quality (https://github.com/google/compare_gan)

All gan loss function is in 'model' file folder.

In order to evaluate all model in a generally recognized method.We use both tensorflow model and pytorch model to get final result. 

Our pytorch inceptionv3 model can be download here (https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth)

Our tensorflow inceptionv3 model can be download here (http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz)

# How to use? 
```
git clone 
cd Final_RelationGAN\Final\
```
```
#train gan model with relation loss and 64 resolution.
python train.py --name Relation --which_loss mean_relu --dataroot path_to_img --gpu_id 1 --resize_or_crop resize_and_crop --which_step lateset --loadSize 64 --which_model_netG basic_64 --which_model_netD relation_64 
#test gan model with relation loss and 64 resolution.
python test.py --name Relation --which_loss mean_relu --result_path path_to_save_reult --gpu_id 1 --which_step lateset  --loadSize 64 --which_model_netG basic_64 --which_model_netD relation_64
#get FID SCORE with Inception-v3 model trained by pytorch.
python FID_Measure.py --name Relation --which_loss mean_relu --dataroot path_to_img --gpu_id 1 --resize_or_crop resize_and_crop  --loadSize 64 --which_model_netG basic_64 --which_model_netD relation_64
#get IS,KID,MS_SSIM with Inception-v3 model trained by tensorflow.
python IS_Score_Tensorflow.py --name Relation --which_loss mean_relu --dataroot path_to_img --gpu_id 1 --resize_or_crop resize_and_crop --which_step lateset  --loadSize 64 --which_model_netG basic_64 --which_model_netD relation_64
```
# Acknowledgments
Our code is based on pytorch-CycleGAN-and-pix2pix(https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)



