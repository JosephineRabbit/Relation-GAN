from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
import torch
import numpy as np
from tensorflow_measure.MS_SSIM import MS_SSIM

import os

def text_save(filename, data):
    file = open(filename,'a')
    for i in data:
        file.write(str(i))
        file.write('\t')
    file.write('\n')
    file.close()

def Get_List(path):
    files = os.listdir(path);
    dirList = []
    fileList = []
    for f in files:
        if (os.path.isdir(path + '/' + f)):
            if (f[0] == '.'):
                pass
            else:
                dirList.append(f)
        if (os.path.isfile(path + '/' + f)):
            fileList.append(f)
    dirList.sort(), fileList.sort()
    return [dirList, fileList]

# --name TripletLoss_mix_up_no_g_max --dataroot /home/kun/Documents/Dataset/data/img_align_celeba/ --resize_or_crop resize_and_crop --no_dropout --gpu_ids 1

if __name__ == '__main__':
    with torch.no_grad():
        opt = TestOptions().parse()
        opt.nThreads = 4
        opt.batchSize = 100
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.fid_count = True
        Measure_data_length = 10000


        opt.phase = 'test'
        data_loader = CreateDataLoader(opt)
        test_dataset = data_loader.load_data()
        model = create_model(opt)

        # generate all test data
        # ms_ssim -1~1 -> 0~1
        ms_ssim_loss = MS_SSIM(max_val=1)
        index = 71
        model.load_networks(str(index))
        model.netG.eval()
        n_batches = Measure_data_length // opt.batchSize
        n_used_imgs = n_batches * opt.batchSize
        fake_all = None
        for i, data in enumerate(test_dataset):
            if i >= Measure_data_length // opt.batchSize:
                break
            start = i * opt.batchSize
            end = start + opt.batchSize
            # test
            model.set_input(data)
            result = model.test().detach()

            if fake_all is None:
                fake_all = result.cpu().data.numpy()
            else:
                fake_all = np.concatenate((fake_all, result.cpu().data.numpy()), 0)
        data_length = fake_all.shape[0]
            # get ssim value
        ms_ssim_value = float(ms_ssim_loss((fake_all[:data_length // 2] + 1)/2, (fake_all[data_length // 2:] + 1)/2).cpu().data)
        print(ms_ssim_value)


