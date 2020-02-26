import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
import torch
from util import util
from PIL import Image
import numpy as np
from data.base_dataset import get_transform
# import cv2
import sys
if __name__ == '__main__':
    with torch.no_grad():
        opt = TestOptions().parse()
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 64  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip

        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        model = create_model(opt)
        name_len = len(str(dataset.__len__()))
        save_root = opt.result_path
        if not os.path.isdir(save_root):
            os.mkdir(save_root)
        for i, data in enumerate(dataset):
            if i>=1000:
                break
            # data['target_img'] = img_target
            name_img = str(i)+'.png'
            save_path = os.path.join(save_root,name_img)

            model.set_input(data)
            result = model.test()
            result = util.tensor2im(result)
            result = Image.fromarray(result)
            result.save(save_path, 'PNG')
            print(i/dataset.__len__())
        print("finished")





