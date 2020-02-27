import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator_nobn
os.environ['CUDA_VISIBLE_DEVICES']='2'
parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()
opt.name='no_bn_relation'
UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

model = Generator_nobn(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()



test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
#test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
#test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')


length = 100
max_ssim = 0
best_iter=0
for epoch in range(97,length):

    model.load_state_dict(torch.load('./epochs/relation_cat_mr/netG_epoch_4_%d.pth'%(epoch+1)))

    out_path = 'benchmark_results/relation_cat_mr' + str(epoch) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        with torch.no_grad():
            image_name = image_name[0]
            lr_image = Variable(lr_image, volatile=True)
            hr_image = Variable(hr_image, volatile=True)
            if torch.cuda.is_available():
                lr_image = lr_image.cuda()
                hr_image = hr_image.cuda()

            sr_image = model(lr_image)
            mse = ((hr_image - sr_image) ** 2).data.mean()
            psnr = 10 * log10(1 / mse)
            ssim = float(pytorch_ssim.ssim(sr_image, hr_image))

            test_images = torch.stack(
                [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
                 display_transform()(sr_image.data.cpu().squeeze(0))])
            image = utils.make_grid(test_images, nrow=3, padding=5)
            utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                             image_name.split('.')[-1], padding=5)

            # save psnr\ssim
            results[image_name.split('_')[0]]['psnr'].append(psnr)
            results[image_name.split('_')[0]]['ssim'].append(ssim)

    out_path = 'statistics/'
    saved_results = {'psnr': [], 'ssim': []}

    for iter,item in enumerate(results.values()):

        psnr = np.array(item['psnr'])
        ssim = np.array(item['ssim'])
        if (len(psnr) == 0) or (len(ssim) == 0):
            psnr = 'No data'
            ssim = 'No data'
        else:
            psnr = psnr.mean()
            ssim = ssim.mean()
        saved_results['psnr'].append(psnr)
        saved_results['ssim'].append(ssim)
        if iter == 0:
            ##compare with max
            if ssim>max_ssim:
                max_ssim=ssim
                best_iter = epoch
            else:
                pass
        else:
            pass


    data_frame = pd.DataFrame(saved_results, results.keys())
    data_frame.to_csv(out_path + 'no_bn_' + str(epoch) + '_test_results.csv', index_label='DataSet')
    print(best_iter,max_ssim)
