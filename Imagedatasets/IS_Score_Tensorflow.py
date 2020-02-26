
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
import os
import torch
import numpy as np
from tensorflow_measure.all_measure import KID_Score,Inception_Score,inception_score
from tensorflow_measure.MS_SSIM import MS_SSIM
from util import util
import cv2
from PIL import Image
import time
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


if __name__ == '__main__':
    with torch.no_grad():
        opt = TestOptions().parse()
        opt.nThreads = 4
        opt.batchSize = 100
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.fid_count = True
        Measure_data_length = 10000
        # get save root
        save_path = opt.result_path
        save_result_flag = False
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        save_root = os.path.join(save_path, opt.name)
        if not os.path.isdir(save_root):
            os.mkdir(save_root)
        IS_txt_path = os.path.join(save_root, '%s_IS_Score.txt' % opt.name)
        KID_txt_path = os.path.join(save_root, '%s_KID_Score.txt' % opt.name)
        SSIM_txt_path = os.path.join(save_root, '%s_MS_SSIM_Score.txt' % opt.name)
        # init data
        KID = None
        ms_ssim_value = None
        mean_fid = None
        std_fid = None
        KID_model = KID_Score(opt.batchSize)
        IS_model = Inception_Score(opt.batchSize)

        # get real img inception score
        opt.phase = 'train'
        data_loader = CreateDataLoader(opt)
        test_dataset = data_loader.load_data()
        img_all =None
        for i, data in enumerate(test_dataset):
            if i >= Measure_data_length // opt.batchSize:
                break
            start = i * opt.batchSize
            end = start + opt.batchSize
            result = data['img']
            if img_all is None:
                img_all = result.cpu().data.numpy()
            else:
                img_all = np.concatenate((img_all, result.cpu().data.numpy()), 0)
        img_all = (img_all+1)*255/2

        # creat test_data
        index = 1
        max_is_score = 0
        max_is_score_delt = 0
        max_is_index = 0

        min_kid_score = 1000000
        min_kid_score_delt = 1000000
        min_kid_index = 0

        max_ms_ssim_value = 0
        max_ms_ssim_index = 0

        opt.phase = 'test'
        data_loader = CreateDataLoader(opt)
        test_dataset = data_loader.load_data()
        model = create_model(opt)

        # generate all test data
        # ms_ssim -1~1 -> 0~1
        ms_ssim_loss = MS_SSIM(max_val=1)

        while 1:
            time_start = time.time()
            fake_all = None
            if not model.load_networks(str(index)):
                break
            model.netG.eval()
            n_batches = Measure_data_length // opt.batchSize
            n_used_imgs = n_batches * opt.batchSize

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
            fake_all = (fake_all + 1) * 255 / 2

            mean_is, std_is = IS_model.get_score(fake_all)
            if KID is None:
                KID = KID_model.get_score(img_all, fake_all, act_img=None)
            else:
                KID = KID_model.get_score(img_all, fake_all, act_img=KID['act_img'])

            text_save(IS_txt_path, [index, mean_is,std_is])
            text_save(SSIM_txt_path, [index, ms_ssim_value])
            text_save(KID_txt_path, [index, KID['KID_mean'],  KID['KID_stddev']])
            print('%d\tKID mean = %.4f\tIS mean = %.4f\tms_ssim_value = %.4f\tuse time %.4f'
                  %(index,KID['KID_mean'],mean_is,ms_ssim_value,time.time() - time_start))
            # print('use time %.4f'%(time.time() - time_start))

            if max_is_score<mean_is:
                max_is_score = mean_is
                max_is_index = index
                max_is_score_delt = std_is

            if max_ms_ssim_value<ms_ssim_value:
                max_ms_ssim_value = ms_ssim_value
                max_ms_ssim_index = index

            if min_kid_score>KID['KID_mean']:
                min_kid_score = KID['KID_mean']
                min_kid_index = index
                min_kid_score_delt = KID['KID_stddev']

            if KID['KID_mean']<10:
                index+=2
            else:
                index += 10

        text_save(IS_txt_path, [max_is_index, max_is_score, max_is_score_delt])
        text_save(SSIM_txt_path, [max_ms_ssim_index, max_ms_ssim_value])
        text_save(KID_txt_path, [index, min_kid_score, min_kid_score_delt])
        print('finished')
        print('KID min = %.4f\tfid max = %.4f\tms_ssim_value max = %.4f'%(min_kid_score,max_is_score,max_ms_ssim_value))




