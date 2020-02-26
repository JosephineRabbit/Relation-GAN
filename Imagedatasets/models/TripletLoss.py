import torch
from torch.autograd import Variable
import torch.utils.data
from .base_model import BaseModel
from . import networks
import itertools

class TripletLoss(BaseModel):
    def name(self):
        return 'TripletLoss'

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
            self.criterionL2 = torch.nn.MSELoss()
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
        self.input_img_last = self.input_img_last.detach()
        self.fake = self.fake.detach()
        self.fake_last = self.fake_last.detach()

        img = self.netD(self.input_img)
        img_last = self.netD(self.input_img_last)
        fake = self.netD(self.fake)
        fake_last = self.netD(self.fake)

        self.loss_D = self.criterionTriplet(img, img_last, fake)
        self.loss_D += self.criterionTriplet(fake, fake_last, img)
        self.loss_D.backward()


    def backward_G(self):

        if not hasattr(self,'input_img_last'):
            setattr(self,'input_img_last',self.input_img)
            setattr(self, 'fake_last', self.netG(self.input_noise).detach())

        self.fake = self.netG(self.input_noise)
        anchor = self.netD(self.fake)
        positive = self.netD(self.input_img)
        negative = self.netD(self.fake_last)
        self.loss_G = self.criterionL2(anchor,positive) - self.criterionL2(negative,anchor)
        self.loss_G.backward()



    def optimize_parameters(self):
        # forward
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.input_img = self.input_img.detach()
        self.fake = self.fake.detach()
        self.input_img_last = self.input_img
        self.fake_last = self.fake


