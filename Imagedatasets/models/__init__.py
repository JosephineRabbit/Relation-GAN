def create_model(opt):
    model = None
    if opt.which_loss == 'mean_relu' or opt.which_loss == 'relu_mean':
        from .Relation import RelationGAN
        model = RelationGAN()
    elif opt.which_loss == 'wgangp':
        from .WGAN_GP import WGAN_GP
        model = WGAN_GP()
    elif opt.which_loss == 'rele':
        from .Relativistic_GAN import Relativistic_GAN
        model = Relativistic_GAN()
    elif opt.which_loss == 'ls_gan':
        from .LS_GAN import LS_GAN
        model = LS_GAN()
    elif opt.which_loss == 'sgan':
        from .GAN import GAN
        model = GAN()
    elif opt.which_loss == 'triplet':
        from .TripletLoss import TripletLoss
        model = TripletLoss()
    else:
        print('error no loss function was found')

    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model