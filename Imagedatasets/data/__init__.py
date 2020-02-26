import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataset(opt,test=False):
    if opt.phase == 'train':
        from data.image_folder import ImageFolder
        target_dataset = ImageFolder(opt,opt.dataroot)
        print("dataset was created")
        return target_dataset
    else:
        from data.image_folder import ImageFolder
        source_dataset = ImageFolder(opt,opt.dataroot)
        print("dataset was created")
    return source_dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
       # if opt.phase == 'test':
        #    shuffle_flg = False
        #else:
        shuffle_flg = True
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=shuffle_flg,
            num_workers=int(opt.nThreads),
            drop_last=True
            )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

    def class_idx(self):
        return self.class_idx
