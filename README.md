# ModularGAN

### Introduction
This repository provides an unofficial PyTorch implementation of ModularGan. The orginal paper is [Modular Generative Adversarial Networks](https://arxiv.org/pdf/1804.03343.pdf).

ModularGAN consists of several reusable and composable modules that carry on different functions (e.g., encoding, decoding, transformations). These modules can be trained simultaneously, leveraging data from all domains, and then combined to construct specific GAN networks at test time, according to the specific image translation task. 

### Update
Update on 2019/05/11: Provide model which could be trained and tested.

### TODO
- Provide model trained on Celeba
- Compute the classification error of each attribute

### Dependencies
* [Python 3.6+](https://www.continuum.io/downloads)
* [PyTorch 1.0.1+](http://pytorch.org/)
* [tqdm 4.31+](https://tqdm.github.io/)
* [TensorFlow 1.13+](https://www.tensorflow.org/) (optional for tensorboard)

### Usage

#### 1. Cloning the repository
```bash
$ git clone https://github.com/LucasBoTang/ModularGAN.git
$ cd ModularGAN/
```

#### 2. Downloading dataset
To download the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html):
```bash
$ bash download.sh data
```

#### 3. Training and testing
To train the model
```bash
$ python main.py --mode train
```

To test the model
```bash
$ python main.py --mode test --test_iters 100000
```

To customize configuration

Cofiguration (e.g. batch size, number of residual blocks) could be customized easily by using argparse.

### Acknowledgement
The code is mainly based on the GitHub repository StarGan (https://github.com/yunjey/stargan).
