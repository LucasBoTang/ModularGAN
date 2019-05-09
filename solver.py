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
import random


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
        self.transformer_num = len(self.attr_dims)
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
        self.device = self.get_device()

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

    def get_device(self):
        """
        get device
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Device:')
            for i in range(torch.cuda.device_count()):
                print('    {}:'.format(i), torch.cuda.get_device_name(i))
        else:
            device = torch.device('cpu')
            print('Device: CPU')
        print('\n')
        return device

    def build_loaders(self):
        """
        build data loader for different modulars
        """
        from dataloader import get_loader
        random.seed(135)
        data_loaders = []
        ind = 0
        for c_dim in self.attr_dims:
            seed = random.randint(0, 200)
            selected_attrs = self.selected_attrs[ind:ind+c_dim]
            loader = get_loader(self.image_dir, self.attr_path, selected_attrs,
                                self.crop_size, self.image_size, self.batch_size,
                                self.mode, self.num_workers, seed)
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

    def gradient_penalty(self, y, x):
        """
        gradient penalty for Wasserstein GAN
        (L2_norm(dy/dx) - 1)**2
        """
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def reset_grad(self):
        """
        reset the gradient to zero
        """
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

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
            for j in range(self.transformer_num):

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # get data and domain label
                x_real, c_org_t = next(self.data_loaders[j])
                # generate target domain labels for transform randomly
                if c_org_t.size(1) > 1:
                    # randomly shuffle
                    c_trg_t = c_org_t[torch.randperm(c_org_t.size(0))]
                else:
                    # reverse value
                    c_trg_t = 1 - c_org_t

                # copy domain labels for computing classification loss
                c_org_l = c_org_t.clone()
                c_trg_l = c_trg_t.clone()

                # to device
                x_real = x_real.to(self.device)
                c_org_t = c_org_t.to(self.device)
                c_trg_t = c_trg_t.to(self.device)
                c_org_l = c_org_l.to(self.device)
                c_trg_l = c_trg_l.to(self.device)

                # generate fake images
                x_fake = self.R(self.T[j](self.E(x_real), c_trg_t))

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # compute loss with real images
                out_src, out_cls = self.D[j](x_real)
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = F.binary_cross_entropy_with_logits(out_cls, c_org_l, size_average=False) / self.batch_size

                # compute loss with fake images
                out_src, _ = self.D[j](x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D[j](x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # compute discrimination loss
                d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp + self.lambda_cls * d_loss_cls

                # backward and optimize
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # log in
                loss = {}
                loss['D{}_loss_real'.format(j)] = d_loss_real.item()
                loss['D{}_loss_fake'.format(j)] = d_loss_fake.item()
                loss['D{}_loss_cls'.format(j)] = d_loss_cls.item()
                loss['D{}_loss_gp'.format(j)] = d_loss_gp.item()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i+1) % self.n_critic == 0:

                    # compute loss with fake images
                    out_src, out_cls = self.D[j](x_fake)
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = F.binary_cross_entropy_with_logits(out_cls, c_trg_l, size_average=False) / self.batch_size

                    # compute loss with cyclic reconstruction
                    x_rec = self.R(self.E(x_real))
                    f_trs = self.T[i](self.E(x_real), c_trg_t)
                    f_rec = self.E(x_fake)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_rec)) / self.transformer_num + torch.mean(torch.abs(f_trs - f_rec))

                    # compute generation loss
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls

                    # backward and optimize
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G{}_loss_fake'.format(j)] = g_loss_fake.item()
                    loss['G{}_loss_rec'.format(j)] = g_loss_rec.item()
                    loss['G{}_loss_cls'.format(j)] = g_loss_cls.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # show the training information
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)

            break
