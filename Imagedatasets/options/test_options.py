from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--result_path', type=str, default='./result', help='path to save result')
        self.parser.add_argument('--which_step', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.isTrain = False
