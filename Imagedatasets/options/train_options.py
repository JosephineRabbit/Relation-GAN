from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--lambda_alpha', type=float, default=1,help='weight for G_loss = g_loss + lambda_alpha*g_loss_2')
        self.parser.add_argument('--clipping_value', type=float, default=0.01,help='weight for G_loss = g_loss + lambda_alpha*g_loss_2')
        self.parser.add_argument('--display_freq', type=int, default=50, help='frequency of showing training results on screen')
        self.parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=50, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000,  help='frequency of saving the latest results')
        self.parser.add_argument('--save_step_freq', type=int, default=1000, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_step', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=30, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=20000000000000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        self.parser.add_argument('--lrg', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lrd', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')


        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=80, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
