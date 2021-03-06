from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util import util
from PIL import Image
import os
import torch
import numpy as np
from inception import InceptionV3,_compute_statistics_of_path,calculate_frechet_distance,adaptive_avg_pool2d

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
        opt.batchSize = 200
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.fid_count = True
        fid_data_length = 20000
        last_measure = 10
        save_path = opt.result_path
        save_result_flag = True
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        # creat data loader
        opt.phase = 'train'
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        # creat test_data

        # ground truth create
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        Inception_model = InceptionV3([block_idx])
        Inception_model.cuda()
        m1, s1 = _compute_statistics_of_path(dataset, Inception_model, opt.batchSize,
                                             2048,fid_data_length)
        print('finished dataset value')

        opt.phase = 'test'
        data_loader = CreateDataLoader(opt)
        test_dataset = data_loader.load_data()
        model = create_model(opt)

        index = 1
        save_number = 1000 # 每一步保存图片的个数
        save_root = os.path.join(save_path, opt.name)
        min_fid = 1000
        min_in = 0

        if not os.path.isdir(save_root):
            os.mkdir(save_root)
        while 1:
            if not model.load_networks(str(index)):
                break
            model.netG.eval()
            save_number = 600
            name_len = save_number
            save_root_sub = os.path.join(save_root,str(index))
            if not os.path.isdir(save_root_sub):
                os.mkdir(save_root_sub)
            # init
            n_batches = fid_data_length // opt.batchSize
            n_used_imgs = n_batches * opt.batchSize
            act = np.empty((n_used_imgs, 2048))

            for i, data in enumerate(test_dataset):
                if i >= fid_data_length // opt.batchSize:
                    break
                start = i * opt.batchSize
                end = start + opt.batchSize
                # test
                model.set_input(data)
                result = model.test().detach()
                # save
                if save_result_flag and save_number>0:
                    img = util.tensor2imMulit(result)
                    for j,im in enumerate(img):
                        name_img = str(i)+'_'+str(j)+'.png'
                        save_path = os.path.join(save_root_sub,name_img)
                        im = Image.fromarray(im)
                        im.save(save_path, 'PNG')
                        save_number-=1

                pred = Inception_model(result)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                act[start:end] = pred.cpu().data.numpy().reshape(opt.batchSize, -1)

            mu = np.mean(act, axis=0)
            sigma = np.cov(act, rowvar=False)

            fid_value = calculate_frechet_distance(m1, s1, mu, sigma)
            text_save(os.path.join(save_root,'%s.txt'%opt.name), [index, fid_value])
            if fid_value<min_fid:
                min_fid = fid_value
                min_in = index

            if fid_value<30:
                index+=2
            else:
                index += 10
            print(index,fid_value)
        text_save(os.path.join(save_root, '%s.txt' % opt.name), [min_in, min_fid])

        # final measure
        model.load_networks(str(min_in))
        fid_all = []
        for i in range(last_measure):
            # init
            n_batches = fid_data_length // opt.batchSize
            n_used_imgs = n_batches * opt.batchSize
            act = np.empty((n_used_imgs, 2048))

            for i, data in enumerate(test_dataset):
                if i >= fid_data_length // opt.batchSize:
                    break
                start = i * opt.batchSize
                end = start + opt.batchSize
                # test
                model.set_input(data)
                result = model.test().detach()
                pred = Inception_model(result)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                act[start:end] = pred.cpu().data.numpy().reshape(opt.batchSize, -1)

            mu = np.mean(act, axis=0)
            sigma = np.cov(act, rowvar=False)

            fid_all.append(calculate_frechet_distance(m1, s1, mu, sigma))

        mean_value = np.mean(fid_all)
        delt_value = np.std(fid_all)
        max_value = max(fid_all)
        min_value = min(fid_all)
        text_save(os.path.join(save_root, '%s.txt' % opt.name),fid_all)
        print("finished")






