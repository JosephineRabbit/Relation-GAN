

Requirements:
* python3
* pytorch
* torchvision
* numpy
* scipy
* tensorflow-gpu


For example
```
#train
python train.py --name Relation --which_loss mean_relu --dataroot path_to_img --gpu_id 1 --resize_or_crop resize_and_crop --which_step lateset --loadSize 64 --which_model_netG basic_64 --which_model_netD relation_64 
#test
python test.py --name Relation --which_loss mean_relu --result_path path_to_save_reult --gpu_id 1 --which_step lateset  --loadSize 64 --which_model_netG basic_64 --which_model_netD relation_64
#get FID SCORE
python FID_Measure.py --name Relation --which_loss mean_relu --dataroot path_to_img --gpu_id 1 --resize_or_crop resize_and_crop  --loadSize 64 --which_model_netG basic_64 --which_model_netD relation_64
#get IS,KID,MS_SSIM
python IS_Score_Tensorflow.py --name Relation --which_loss mean_relu --dataroot path_to_img --gpu_id 1 --resize_or_crop resize_and_crop --which_step lateset  --loadSize 64 --which_model_netG basic_64 --which_model_netD relation_64
```

Our pytorch inceptionv3 model is (https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth)

Our tensorflow inceptionv3 model is (http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz)



