#coding=utf-8
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn import functional as F
from .ResGenerater_256 import GoodDiscriminator,GoodGenerator,GoodDiscriminator_relation
import numpy as np
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal'):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=0.02)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=1)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    return init_func


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type=init_type)
    net.apply(init_weights(init_type))
    return net


def define_G(input_nc, output_nc,which_model_netG,ngf=64,input_size=0, init_type='normal',gpu_ids=[]):
    netG = None

    if which_model_netG == 'basic_32':
        netG = Basic_Generator_32(input_nc, input_size,ngf)
    elif which_model_netG == 'basic_64':
        netG = Basic_Generator_64(input_nc, output_nc,input_size)
    elif which_model_netG == 'basic_128':
        netG = Basic_Generator_128(input_nc, input_size)
    elif which_model_netG == 'basic_256':
        netG = GoodGenerator(ngf, input_size)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)


def define_D(input_nc, ndf, which_model_netD, init_type='normal', gpu_ids=[]):

    if which_model_netD == 'basic_32':
        netD = Basic_Discriminator_32(input_nc)
    elif which_model_netD == 'basic_64':
        netD = Basic_Discriminator_64(input_nc)
    elif which_model_netD == 'basic_128':
        netD = Basic_Discriminator_128(input_nc)
    elif which_model_netD == 'basic_256':
        netD = GoodDiscriminator(input_nc, ndf)
        
    elif which_model_netD == 'relation_32':
        netD = RelationGAN_Discriminator_32(input_nc)
    elif which_model_netD == 'relation_64':
        netD = RelationGAN_Discriminator_64(input_nc)
    elif which_model_netD == 'relation_128':
        netD = RelationGAN_Discriminator_128(input_nc)
    elif which_model_netD == 'relation_256':
        netD = GoodDiscriminator_relation(input_nc, ndf)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, gpu_ids)


##############################################################################
# Classes
##############################################################################

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor,l1use=False):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
        if l1use == True:
            self.loss = nn.L1Loss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class Basic_Generator_32(nn.Module):
    def __init__(self, z_dim, size, nfilter=64, nfilter_max=512):
        super(Basic_Generator_32,self).__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.z_dim = z_dim

        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)
        self.fc = nn.Linear(z_dim, self.nf0*s0*s0)
        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetbnBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]
        blocks += [
            ResnetBlock(nf, nf),
        ]
        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z):
        batch_size = z.size(0)
        out = self.fc(z.view(batch_size,-1))
        out = out.view(batch_size, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))
        out = F.tanh(out)
        return out


class Basic_Discriminator_32(nn.Module):
    def __init__(self, size,embed_size=256, nfilter=64, nfilter_max=1024):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)
        blocks = [
            ResnetBlock(nf, nf)
        ]
        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]
        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, 1)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0*self.s0*self.s0)
        out = self.fc(actvn(out))
        return out


class RelationGAN_Discriminator_32(nn.Module):
    def __init__(self,size, nlabels=1,  embed_size=256, nfilter=64, nfilter_max=1024):
        super(self).__init__()
        self.embed_size = embed_size
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            ResnetBlock(nf, nf)
        ]
        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            self.nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, self.nf1),
            ]
        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.em_resnet = nn.Sequential(*blocks)
        self.re = nn.Sequential(
            nn.Conv2d(self.nf1, self.nf1 , 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.nf1, self.nf1//2, 3, 1, 1, bias=False),
        )
        self.fc = nn.Linear(self.nf1//2*s0*s0, nlabels)

    def forward(self, x1, x2):
        all = torch.cat([x1,x2],dim=0)
        batch_size = x1.size(0)
        out = self.conv_img(all)
        em = self.em_resnet(out)
        re = self.re(em[:batch_size]+em[batch_size:])
        re = re.view(batch_size, -1)
        re = self.fc(re)

        return re


##############################################################################
# 64 Network
##############################################################################

class Basic_Generator_64(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(Basic_Generator_64, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, input_size//8*input_size//8*512),
            nn.BatchNorm1d(input_size//8*input_size//8*512),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.output_dim, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        input = input.view(input.size()[0], -1)
        x = self.fc(input)
        x = x.view(-1, 512, (self.input_size // 8), (self.input_size // 8))
        x = self.deconv(x)
        return x

class Basic_Discriminator_64(nn.Module):
    def __init__(self,input_size=64):
        super(Basic_Discriminator_64, self).__init__()
        self.input_size = input_size

        self.embdimg = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # (B,1,64,64) > (B,64,32,32)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # (B,64,32,32) > (B,64*2,16,16)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),  # (B,64*2,16,16) > (B,64*4,8,8)
            nn.Conv2d(128, 256, 4, 2, 1),  # (B,64,32,32) > (B,64*2,16,16)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),  # (B,64*2,16,16) > (B,64*4,8,8)
            nn.Conv2d(256, 512, 4, 2, 1),  # (B,64,32,32) > (B,64*2,16,16)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),  # (B,64*2,16,16) > (B,64*4,8,8)
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(input_size//8*input_size//8*512,1),
        )
    def forward(self, x):
        x = self.embdimg(x)
        x = x.view(x.size()[0],-1)
        x = self.fc(x)
        return x

class RelationGAN_Discriminator_64(nn.Module):
    """docstring for discriminator"""

    def __init__(self, input_size,ndf=64,):
        super(RelationGAN_Discriminator_64, self).__init__()

        self.embdimg = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # (B,1,64,64) > (B,64,32,32)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # (B,64,32,32) > (B,64*2,16,16)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),  # (B,64*2,16,16) > (B,64*4,8,8)
            nn.Conv2d(128, 256, 4, 2, 1),  # (B,64,32,32) > (B,64*2,16,16)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),  # (B,64*2,16,16) > (B,64*4,8,8)
            nn.Conv2d(256, 512, 4, 2, 1),  # (B,64,32,32) > (B,64*2,16,16)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),  # (B,64*2,16,16) > (B,64*4,8,8)
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(input_size//8*input_size//8*512,1),
        )


        self.relation = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
        )
    # forward method

    def forward(self, input1,input2):
        input1 = self.embdimg(input1)
        input2 = self.embdimg(input2)
        # out = self.relation(torch.cat([input1,input2],1))
        out = self.relation(input1+input2)
        out = out.view(out.size()[0],-1)
        out = self.fc(out)
        return out


##############################################################################
# 128 Network
##############################################################################
def conv_out_size_same(size, stride):
  return size// stride

class Basic_Generator_128(nn.Module):
    def __init__(self, z_dim, size):
        super().__init__()

        self.size_1 = conv_out_size_same(size, 2)
        self.size_2 = conv_out_size_same(self.size_1, 2)
        self.size_3 = conv_out_size_same(self.size_2, 2)
        self.z_dim = z_dim

        self.fc = []
        self.fc.append(nn.Linear(z_dim, self.size_3 * self.size_3 * 512))
        # self.fc.append(nn.BatchNorm2d(size_3*size_3*512))
        self.fc.append(nn.ReLU(inplace=True))

        self.conv = []
        self.conv.append(nn.ConvTranspose2d(512, 256, 4, 2, 1))
        # self.conv.append(nn.BatchNorm2d(256))
        self.conv.append(nn.ReLU(inplace=True))

        self.conv.append(nn.ConvTranspose2d(256, 128, 4, 2, 1))
        # self.conv.append(nn.BatchNorm2d(128))
        self.conv.append(nn.ReLU(inplace=True))

        self.conv.append(nn.ConvTranspose2d(128, 64, 4, 2, 1))
        # self.conv.append(nn.BatchNorm2d(64))
        self.conv.append(nn.ReLU(inplace=True))

        self.conv.append(nn.ConvTranspose2d(64, 3, 3, 1, 1))
        # self.conv.append(nn.BatchNorm2d(64))
        # self.conv.append(nn.functional.tanh(inplace=True))
        self.fc = nn.Sequential(*self.fc)
        self.conv = nn.Sequential(*self.conv)

    def forward(self, z):
        batch_size = z.size(0)
        out = self.fc(z.view(-1, self.z_dim))
        out = out.view(batch_size, 512, self.size_3, self.size_3)

        out = self.conv(out)
        out = F.tanh(out)

        return out

class Basic_Discriminator_128(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.conv = []
        self.fc = []

        self.size_1 = conv_out_size_same(size, 2)
        self.size_2 = conv_out_size_same(self.size_1, 2)
        self.size_3 = conv_out_size_same(self.size_2, 2)

        self.conv.append(nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))

        self.conv.append(nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        #
        self.conv.append(nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        #
        self.conv.append(nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        #
        self.conv.append(nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))

        self.conv.append(nn.Conv2d(512,512,kernel_size=4,stride=2,padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))

        self.conv.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))

        self.fc.append(nn.Linear(self.size_3 * self.size_3 * 512,1))

        self.fc = nn.Sequential(*self.fc)
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

class RelationGAN_Discriminator_128(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.conv = []
        self.fc = []

        self.size_1 = conv_out_size_same(size, 2)
        self.size_2 = conv_out_size_same(self.size_1, 2)
        self.size_3 = conv_out_size_same(self.size_2, 2)

        self.conv.append(nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))

        self.conv.append(nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        #
        self.conv.append(nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        #
        self.conv.append(nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        #
        self.conv.append(nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))

        self.conv.append(nn.Conv2d(512,512,kernel_size=4,stride=2,padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))

        self.conv.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.conv.append(nn.LeakyReLU(0.1, inplace=True))

        self.conv_relation = []
        self.conv_relation.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.conv_relation.append(nn.LeakyReLU(0.1, inplace=True))
        self.conv_relation.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))

        self.fc.append(nn.Linear(self.size_3 * self.size_3 * 512,1))

        self.fc = nn.Sequential(*self.fc)
        self.conv = nn.Sequential(*self.conv)
        self.conv_relation = nn.Sequential(*self.conv_relation)

    def forward(self, x1,x2):
        batch_size = x1.size(0)
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x = self.conv_relation(x1+x2)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super(ResnetBlock,self).__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class ResnetbnBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super(ResnetbnBlock,self).__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Sequential(nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(self.fhidden))
        self.conv_1 = nn.Sequential(nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias),
                                    nn.BatchNorm2d(self.fout))
        if self.learned_shortcut:
            self.conv_s = nn.Sequential(nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False),
                                        nn.BatchNorm2d(self.fout))


    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out