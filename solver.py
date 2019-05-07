from model import Encoder, Transformer, Reconstructor, Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import time
import datetime


class Solver(object):
    """
    solver for training and testing ModularGAN
    """

    def __init__(self, config):
        """
        initialize configurations from argument
        """
        # model configurations
        self.crop_size = config.crop_size
        self.image_size = config.image_size
        self.e_conv_dim = config.e_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.e_repeat_num = config.e_repeat_num
        self.t_repeat_num = config.t_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.attr_dims = config.attr_dims
        self.selected_attrs = config.selected_attrs

        # training configurations
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # test configurations
        self.test_iters = config.test_iters

        # miscellaneous
        self.mode = config.mode
        self.num_workers = config.num_workers
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # directories
        self.image_dir = config.image_dir
        self.attr_path = config.attr_path
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # build data loader
        self.data_loaders = self.build_loaders()

        # build the model and tensorboard
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_loaders(self):
        """
        build data loader for different modulars
        """
        from dataloader import get_loader
        data_loaders = []
        ind = 0
        for c_dim in self.attr_dims:
            selected_attrs = self.selected_attrs[ind:ind+c_dim]
            loader = get_loader(self.image_dir, self.attr_path, selected_attrs,
                                self.crop_size, self.image_size, self.batch_size,
                                self.mode, self.num_workers)
            data_loaders.append(iter(loader))
            ind += c_dim
        return data_loaders

    def build_model(self):
        """
        create network modulars
        """
        # create modulars
        self.E = Encoder(conv_dim=self.e_conv_dim, repeat_num=self.e_repeat_num)

        self.T = torch.nn.ModuleList()
        for c_dim in self.attr_dims:
            self.T.append(Transformer(conv_dim=self.e_conv_dim*4, c_dim=c_dim, repeat_num=self.t_repeat_num))

        self.R = Reconstructor(conv_dim=self.e_conv_dim*4)
        self.R.to(self.device)

        self.D = torch.nn.ModuleList()
        for c_dim in self.attr_dims:
            self.D.append(Discriminator(image_size=self.image_size, conv_dim=self.d_conv_dim, c_dim=c_dim, repeat_num=self.d_repeat_num))
        self.D.to(self.device)

        # optimizer
        self.g_optimizer = torch.optim.Adam(list(self.E.parameters())+list(self.T.parameters())+list(self.R.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # print information
        self.print_network('Encoder', self.E)
        self.print_network('Transformers', self.T)
        self.print_network('Reconstructor', self.R)
        self.print_network('Discriminators', self.D)
        print('\n')

        # move to device
        self.E.to(self.device)
        self.T.to(self.device)
        self.R.to(self.device)
        self.D.to(self.device)

    def print_network(self, name, model):
        """
        print out the network information
        """
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print("The number of parameters: {} in {}".format(num_params, name))

    def build_tensorboard(self):
        """
        build a tensorboard logger
        """
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def restore_model(self, resume_iters):
        """
        restore the trained model
        """
        print('Loading the trained models from step {}...'.format(resume_iters))
        E_path = os.path.join(self.model_save_dir, '{}-E.ckpt'.format(resume_iters))
        T_path = os.path.join(self.model_save_dir, '{}-T.ckpt'.format(resume_iters))
        R_path = os.path.join(self.model_save_dir, '{}-R.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.E.load_state_dict(torch.load(E_path, map_location=lambda storage, loc: storage))
        self.T.load_state_dict(torch.load(E_path, map_location=lambda storage, loc: storage))
        self.R.load_state_dict(torch.load(E_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def train(self):
        """
        train model
        """
        # initialize learning rate and decay later
        g_lr = self.g_lr
        d_lr = self.d_lr

        # start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # start training
        print('Start training...')
        start_time = time.time()
        tbar = tqdm(range(start_iters, self.num_iters))
        for i in tbar:

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            x_real, label_org = next(self.data_loaders[0])
            print(label_org)
            break
