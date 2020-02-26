import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.CenterCrop(opt.fineSize))
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'resize':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    # if opt.isTrain and not opt.no_flip:
    #     transform_list.append(transforms.RandomHorizontalFlip())
    #
    # if opt.Imagenet:
    #     transform_list += [transforms.ToTensor(),
    #                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                             std=[0.229, 0.224, 0.225])]
    # else:
    transform_list += [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)

def img_process(img,loadsize,flg_pose):
    try :
        w, h = img.size
    except:
        print("hah")

    result = np.zeros((loadsize,loadsize,3))
    if not flg_pose:
        result[...,1] = 255
    if h >= w:
        w = int(w*loadsize/h)
        h = loadsize
        img = img.resize((w,h),Image.ANTIALIAS)
        bias = int((loadsize - w)/2)
        img = np.array(img)
        result[0:h,bias:bias+w,...] = img[0:h,0:w,...]
    if w > h:
        h = int(h * loadsize / w)
        w = loadsize
        img = img.resize((w,h),Image.ANTIALIAS)
        bias = int((loadsize - h)/2)
        img = np.array(img)
        result[bias:bias+h,0:w,...] = img[0:h,0:w,...]
    result = result.astype(np.uint8)
    return Image.fromarray(result)

