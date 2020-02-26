import torch
from torch.autograd import Variable
import torch.utils.data
from .base_model import BaseModel
from . import networks
import itertools

class LS_GAN(BaseModel):
    """docstring for SiameseGAN"""
    def name(self):
        return 'LS_GAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.loss_names = ['D','G']

        self.visual_names = ['input_img','fake']

        if self.isTrain:
            self.model_names = ['D', 'G']
        else:  # during test time, only load Gs
            self.model_names = ['G']


        self.netG = networks.define_G(input_nc = self.opt.z_dim,output_nc=3,which_model_netG=opt.which_model_netG,
                                      input_size=opt.loadSize,init_type = opt.init_type,gpu_ids=self.gpu_ids)
        if not self.isTrain:
            self.netG.eval()

        if self.isTrain:
            self.netD = networks.define_D(input_nc = opt.loadSize, ndf = 64,which_model_netD = opt.which_model_netD,
                                          gpu_ids =self.gpu_ids)

        if self.isTrain:
            self.margin = 100
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=True, tensor=self.Tensor)
            self.criterionTriplet = torch.nn.TripletMarginLoss(margin=100, p=2.0)
            self.criterionMax = torch.nn.ReLU()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()),
                                                lr=self.opt.lrg,
                                                betas=(self.opt.beta1, self.opt.beta2))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()),
                                                lr=self.opt.lrd,
                                                   betas=(self.opt.beta1, self.opt.beta2))

            self.alpha = opt.lambda_alpha
            self.c = opt.clipping_value

        self.load_networks(opt.which_step)
        self.print_networks(opt.verbose)

    def set_input(self, input):
        if self.isTrain:
            input_img = input['img']
            input_noise = input['noise']
            if len(self.gpu_ids) > 0:
                input_img = input_img.cuda(self.gpu_ids[0])
                input_noise = input_noise.cuda(self.gpu_ids[0])
            self.input_img = Variable(input_img)
            self.input_noise = Variable(input_noise)
        else:
            input_noise = input['noise']
            if len(self.gpu_ids) > 0:
                input_noise = input_noise.cuda(self.gpu_ids[0])
            self.input_noise = input_noise


    def get_noise(self):
        return self.input_noise

    def get_img(self):
        return self.input_img

    def test(self):
        self.fake = self.netG(self.input_noise)
        return self.fake

    def backward_D(self):
        self.input_img = self.input_img.detach()
        self.fake = self.fake.detach()

        D_real = self.netD(self.input_img)
        D_fake = self.netD(self.fake)

        self.loss_D = self.criterionGAN(D_real,True) + self.criterionGAN(D_fake,False)
        self.loss_D.backward()


    def backward_G(self):

        self.fake = self.netG(self.input_noise)
        self.loss_G = self.criterionGAN(self.netD(self.fake),True)

        self.loss_G.backward()



    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()



