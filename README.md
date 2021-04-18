# ModularGAN

<p align="center"><img width="100%" src="image/cover.png" /></p>

### Introduction
This repository provides an unofficial PyTorch implementation of ModularGan. The original paper is [Modular Generative Adversarial Networks](https://arxiv.org/pdf/1804.03343.pdf).

ModularGAN consists of several reusable and composable modules that carry on different functions (e.g., encoding, decoding, transformations). These modules can be trained simultaneously, leveraging data from all domains, and then combined to construct specific GAN networks at test time, according to the specific image translation task.

**Attention:** **The code is an unofficial version, and the details of model are not exactly same as paper. Especially, the model architecture and hyperparameters are different.**

<br />

### Dependencies
* [Python 3.6+](https://www.python.org/)
* [PyTorch 1.0.1+](http://pytorch.org/)
* [tqdm 4.31+](https://tqdm.github.io/)
* [TensorFlow 1.13+](https://www.tensorflow.org/) (optional for tensorboard)

<br />

### Download

#### 1. Repository
```bash
$ git clone https://github.com/LucasBoTang/ModularGAN.git
```

#### 2. Dataset
To download the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html):
```bash
$ cd ModularGAN/
$ bash download.sh data
```
Or the zip file could be downloaded directly [here](https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0).

#### 3. Pretrained model
To download the pretrained model:
```bash
$ cd ModularGAN/
$ bash download.sh model
```
Or the zip file could be downloaded directly [here](https://www.dropbox.com/s/n1vxfdlbrbt4gk4/pretrained.zip?dl=0).

<br />

### Training and testing

#### 1. Training
To train the model
```bash
$ cd ModularGAN/
$ python main.py --mode train --batch_size 16 --num_epochs 20
```

#### 2. Loss curve
<p align="center"><img width="100%" src="image/loss_curve.png" /></p>

#### 3. Images generation
<p align="center"><img width="100%" src="image/sample.gif" /></p>

#### 4. Testing
To test the model
```bash
$ cd ModularGAN/
$ python main.py --mode test --test_epoch 20
```

#### 5. Cofiguration
Cofiguration (e.g. batch size, number of residual blocks) could be customized easily by using argparse.

##### Model configuration

--crop_size: image crop size

--image_size: image resolution

--e_conv_dim: number of conv filters in the first layer of Encoder

--d_conv_dim: number of conv filters in the first layer of Discriminator

--e_repeat_num: number of residual blocks in Encoder

--t_repeat_num: number of residual blocks in Transformer

--d_repeat_num: number of strided conv layers in Discriminator

'--lambda_cls: weight for domain classification loss

--lambda_cyc: weight for reconstruction loss

--lambda_gp: weight for gradient penalty

--attr_dims: separate attributes into different modules

--selected_attrs: selected attributes for the CelebA dataset

--batch_size: mini-batch size

--num_epochs: number of total iterations for training D

--num_epochs_decay: number of iterations for decaying lr

--g_lr: learning rate for Generation

--d_lr: learning rate for Discrimination

--n_critic: number of D updates per each G update

--beta1: beta1 for Adam optimizer

--beta2: beta2 for Adam optimizer

--lr_update_step: step of update learning rate

--resume_epoch: resume training from this step

##### Test configuration

--test_epoch: test on this epoch

--mode: train or test

--use_tensorboard: use tensorboard or not

##### Directories

--image_dir: directory of images

--attr_path: file path of attributes

--log_dir: directory of logs

--model_save_dir: directory to save model checkpoints

--sample_dir: directory to save samples

--result_dir: directory to save results

##### Sample step size

--log_step: step of log

--sample_step: step of sample

--model_save_step: step of saving model checkpoints


<br />

### Result
#### 1. Sample images
<p align="center"><img width="100%" src="image/sample1.png" /></p>
<p align="center"><img width="100%" src="image/sample2.png" /></p>

<br />

### Acknowledgement
The code is mainly based on the GitHub repository [StarGan](https://github.com/yunjey/stargan).
