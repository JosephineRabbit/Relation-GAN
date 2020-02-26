
import torch.utils.data as data

from PIL import Image
import os
import os.path
from data.base_dataset import BaseDataset, get_transform
import torchvision.transforms as transforms
import torch
import random
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def default_loader(path):
    try :
        img = Image.open(path).convert('RGB')
    except:
        return False
    return img



class ImageFolder(BaseDataset):

    def __init__(self,opt,root):
        super(BaseDataset, self).__init__()
        self.opt = opt

        if opt.phase == "train":
            self.transform_img = get_transform(opt)
            self.loader = default_loader
            self.imgs = make_dataset(root)
            if len(self.imgs) == 0:
                raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                                   "Supported image extensions are: " +
                                    ",".join(IMG_EXTENSIONS)))
        self.z_dim = opt.z_dim

    def __getitem__(self, index):
        if self.opt.phase == "train":
            while 1:
                img = self.loader(self.imgs[index])
                if  img != False:
                    break
                index = random.randint(0,len(self.imgs)-1)
            img = self.transform_img(img)
            noise = torch.randn(self.z_dim).view(self.z_dim, 1, 1)
            return {'img':img,'noise':noise}
        else:
            noise = torch.randn(self.z_dim).view(self.z_dim, 1, 1)
            return {'noise': noise}

    def __len__(self):
        if self.opt.phase == "train":
            return len(self.imgs)
        else:
            return 100000000
