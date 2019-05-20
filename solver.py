from model import Encoder, Transformer, Reconstructor, Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import time
import datetime


class Solver(object):
    """
    solver for training and testing ModularGAN
    """

    def __init__(self, config, data_loader):
        """
        initialize configurations from argument
        """
        # model configurations
        self.image_size = config.image_size
        self.e_conv_dim = config.e_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.e_repeat_num = config.e_repeat_num
        self.t_repeat_num = config.t_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_cyc = config.lambda_cyc
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
        self.num_workers = config.num_workers
        self.use_tensorboard = config.use_tensorboard
        self.device = self.get_device()

        # directories
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # data loader
        self.data_loader = data_loader

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
        if type(model) == torch.nn.modules.container.ModuleList:
            print(model[0])
        else:
            print(model)
        print('The number of parameters: {} in {}'.format(num_params, name))
        print('\n')

    def build_tensorboard(self):
        """
        build a tensorboard logger
        """
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def create_labels(self):
        """
        generate target domain labels for debugging and testing
        """
        label_list = []
        batch = min(self.batch_size, 8)
        for c_dim in self.attr_dims:
            if c_dim > 1:
                labels = []
                for i in range(c_dim):
                    label = torch.zeros([batch, c_dim]).to(self.device)
                    label[:, i] = 1
                    labels.append(label)
            else:
                labels = [torch.zeros([batch, 1]).to(self.device), torch.ones([batch, 1]).to(self.device)]
            label_list.append(labels)
        return label_list

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
        self.T.load_state_dict(torch.load(T_path, map_location=lambda storage, loc: storage))
        self.R.load_state_dict(torch.load(R_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def generate_labels(self, label_org):
        """
        generate target domain labels for transform randomly
        """
        # shuffle
        label_trg = label_org.clone()
        label_trg = label_trg[torch.randperm(label_trg.size(0))]
        ind = 0
        for i, c_dim in enumerate(self.attr_dims):
            if c_dim == 1:
                # reverse
                label_trg[:, ind] = 1 - label_org[:, ind]
            else:
                # avoid empty label
                for j in range(label_org.size(0)):
                    label = label_trg[j, ind:ind + c_dim]
                    if torch.sum(label) == 0:
                        label[0] = 1
                        label[:] = label[torch.randperm(label.size(0))]
            ind += c_dim
        return label_trg.detach()

    def label_slice(self, label, ind):
        """
        slice label for different transformers and discriminators
        """
        start = 0
        for i, c_dim in enumerate(self.attr_dims):
            if i >= ind:
                label_slice = label[:, start:start+c_dim]
                break
            else:
                start += c_dim
        return label_slice

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

    def denorm(self, x):
        """
        convert the range from [-1, 1] to [0, 1]
        """
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def update_lr(self, g_lr, d_lr):
        """
        decay learning rates of the generator and discriminator
        """
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def save_sample(self, x_fixed, c_trg_list, ind):
        """
        save translated image for debug and test
        """
        with torch.no_grad():
            x_list = [x_fixed]
            # transform
            for j in range(len(c_trg_list)):
                for c_trg in c_trg_list[j]:
                    x_fake = self.R(self.T[j](self.E(x_fixed), c_trg))
                    x_list.append(x_fake)
            x_concat = torch.cat(x_list, dim=3)
            # save images
            sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(ind))
            save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
            print('Saved real and fake images into {}...'.format(sample_path))

    def save_checkpoint(self, ind):
        """
        save checkpoints
        """
        E_path = os.path.join(self.model_save_dir, '{}-E.ckpt'.format(ind))
        T_path = os.path.join(self.model_save_dir, '{}-T.ckpt'.format(ind))
        R_path = os.path.join(self.model_save_dir, '{}-R.ckpt'.format(ind))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(ind))
        torch.save(self.E.state_dict(), E_path)
        torch.save(self.T.state_dict(), T_path)
        torch.save(self.R.state_dict(), R_path)
        torch.save(self.D.state_dict(), D_path)
        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def train(self):
        """
        train model
        """
        # initialize learning rate and decay later
        g_lr = self.g_lr
        d_lr = self.d_lr

        # iterate dataset
        data_iter = iter(self.data_loader)

        # fetch fixed images for debugging
        x_fixed, _ = next(data_iter)
        x_fixed = x_fixed.to(self.device)[:8]
        c_trg_list = self.create_labels()

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

            # reset loss record
            d_loss_dict = {'D/loss_src':0, 'D/loss_gp':0}
            if i and i % self.n_critic == 0:
                g_loss_dict = {'G/loss_src':0, 'G/loss_cyc':0}

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # get data and domain label
            try:
                x_real, c_org_t = next(data_iter)
            # avoid StopIteration
            except:
                data_iter = iter(self.data_loader)
                x_real, c_org_t = next(data_iter)

            # generate target domain labels for transform randomly
            c_trg_t = self.generate_labels(c_org_t)

            # copy domain labels for computing classification loss
            c_org_l = c_org_t.clone()
            c_trg_l = c_trg_t.clone()

            # to device
            x_real = x_real.to(self.device)
            c_org_t = c_org_t.to(self.device)
            c_trg_t = c_trg_t.to(self.device)
            c_org_l = c_org_l.to(self.device)
            c_trg_l = c_trg_l.to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # initialize total loss of discriminator
            d_loss = 0

            for j in range(self.transformer_num):

                # clear gradient
                self.reset_grad()

                # slice label for current transformer
                c_trg_t_j = self.label_slice(c_trg_t, j)
                c_org_l_j = self.label_slice(c_org_l, j)

                # compute classification loss with real images
                out_src, out_cls = self.D[j](x_real)
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = F.binary_cross_entropy_with_logits(out_cls, c_org_l_j, size_average=False) / x_real.size(0)

                # generate fake images
                x_fake = self.R(self.T[j](self.E(x_real), c_trg_t_j))
                # compute classification loss with fake images
                out_src, _ = self.D[j](x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # compute classification loss for gradient penalty
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D[j](x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # compute discrimination loss
                d_loss_j = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp + self.lambda_cls * d_loss_cls

                # backward and optimize
                d_loss_j.backward()
                self.d_optimizer.step()

                # logging
                d_loss_dict['D/loss_src'] += d_loss_real.item() + d_loss_fake.item()
                d_loss_dict['D/loss_gp'] += d_loss_gp.item()
                d_loss_dict['D/loss_cls{}'.format(j)] = d_loss_cls.item()
                d_loss += d_loss_j.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if i and i % self.n_critic == 0:

                # clear gradient
                self.reset_grad()

                # compute l1 loss with cyclic reconstruction for encoder and decoder
                x_rec = self.R(self.E(x_real))
                g_loss_cyc = torch.mean(torch.abs(x_real - x_rec))

                # compute generation loss and backward
                g_loss = self.lambda_cyc * g_loss_cyc
                g_loss.backward()

                # logging
                g_loss_dict['G/loss_cyc'] += g_loss_cyc.item()

                for j in range(self.transformer_num):

                    # slice label for current transformer
                    c_org_t_j = self.label_slice(c_org_t, j)
                    c_trg_t_j = self.label_slice(c_trg_t, j)
                    c_trg_l_j = self.label_slice(c_trg_l, j)

                    # generate fake images
                    f_trs = self.T[j](self.E(x_real), c_trg_t_j)
                    x_fake = self.R(f_trs)

                    # compute classification loss with fake images
                    out_src, out_cls = self.D[j](x_fake)
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = F.binary_cross_entropy_with_logits(out_cls, c_trg_l_j, size_average=False) / x_real.size(0)

                    # compute l1 loss with cyclic reconstruction for encoded features
                    f_rec = self.E(x_fake)
                    g_loss_cyc = torch.mean(torch.abs(f_trs - f_rec))

                    # compute generation loss and backward
                    g_loss = g_loss_fake + self.lambda_cyc * g_loss_cyc + self.lambda_cls * g_loss_cls
                    g_loss.backward()

                    # logging
                    g_loss_dict['G/loss_src'] += g_loss_fake.item()
                    g_loss_dict['G/loss_cyc'] += g_loss_cyc.item()
                    g_loss_dict['G/loss_cls{}'.format(j)] = g_loss_cls.item()

                # backward and optimize
                self.g_optimizer.step()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # show the training information
            if i and i % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = 'Elapsed [{}], Iteration [{}/{}]'.format(et, i, self.num_iters)
                loss = {**g_loss_dict, **d_loss_dict}
                for tag, value in loss.items():
                    log += ', {}: {:.4f}'.format(tag, value)
                tbar.set_description('D_loss: {:.3f}, G_loss: {:.3f}'.format(d_loss, g_loss))

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i)

            # translate fixed images for debugging
            if i % self.sample_step == 0:
                self.save_sample(x_fixed, c_trg_list, i)

            # save checkpoints
            if i and i % self.model_save_step == 0:
                self.save_checkpoint(i)

            # decay learning rates
            if i % self.lr_update_step == 0 and i > self.num_iters - self.num_iters_decay:
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """
        translate images using StarGAN trained on a single dataset
        """
        # load the trained generator
        self.restore_model(self.test_iters)

        # create target domain labels
        c_trg_list = self.create_labels()

        with torch.no_grad():

            tbar = tqdm(enumerate(self.data_loaders[0]), total=len(self.data_loaders[0]))
            for i, (x_real, c_org) in tbar:

                # fecth input images
                x_real = x_real.to(self.device)

                # translate and save
                self.save_sample(x_real, c_trg_list, i)

        print('Saved real and fake images into {}...'.format(result_path))
