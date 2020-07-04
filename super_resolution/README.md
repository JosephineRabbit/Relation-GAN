# SRGAN
A PyTorch implementation of SRGAN based on CVPR 2017 paper 
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c pytorch
```
- opencv
```
conda install opencv
```

## Datasets

### Train、Val Dataset
The train and val datasets are sampled from [VOC2012](http://cvlab.postech.ac.kr/~mooyeol/pascal_voc_2012/).
Train dataset has 16700 images and Val dataset has 425 images.
Download the datasets from [here](https://pan.baidu.com/s/1xuFperu2WiYc5-_QXBemlA)(access code:5tzp), and then extract it into `data` directory.

### Test Image Dataset
The test image dataset are sampled from 
| **Set 5** |  [Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)
| **Set 14** |  [Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests)
| **BSD 100** | [Martin et al. ICCV 2001](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
| **Sun-Hays 80** | [Sun and Hays ICCP 2012](http://cs.brown.edu/~lbsun/SRproj2012/SR_iccp2012.html)
| **Urban 100** | [Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr).
Download the image dataset from [here](https://pan.baidu.com/s/1vGosnyal21wGgVffriL1VQ)(access code:xwhy), and then extract it into `data` directory.

### Test Video Dataset
The test video dataset are three trailers. Download the video dataset from 
[here](https://pan.baidu.com/s/1NUZKm5xCHRj1O0JlCZIu8Q)(access code:zabi).

## Usage

### Train
```
python train.py

optional arguments:
--crop_size                   training images crop size [default value is 88]
--upscale_factor              super resolution upscale factor [default value is 4](choices:[2, 4, 8])
--num_epochs                  train epoch number [default value is 100]
```
The output val super resolution images are on `training_results` directory.

### Test Benchmark Datasets
```
python test_benchmark.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution images are on `benchmark_results` directory.

### Test Single Image
```
python test_image.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--test_mode                   using GPU or CPU [default value is 'GPU'](choices:['GPU', 'CPU'])
--image_name                  test low resolution image name
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution image are on the same directory.

### Test Single Video
```
python test_video.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--video_name                  test low resolution video name
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution video and compared video are on the same directory.

## Benchmarks
**Upscale Factor = 2**

Epochs with batch size of 64 takes ~2 minute 30 seconds on a NVIDIA GTX 1080Ti GPU. 


