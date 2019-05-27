# ModularGAN

### Introduction
This repository provides an unofficial PyTorch implementation of ModularGan. The original paper is [Modular Generative Adversarial Networks](https://arxiv.org/pdf/1804.03343.pdf).

ModularGAN consists of several reusable and composable modules that carry on different functions (e.g., encoding, decoding, transformations). These modules can be trained simultaneously, leveraging data from all domains, and then combined to construct specific GAN networks at test time, according to the specific image translation task.

**Attention:** **The code is an unofficial version, and the details of model is not exactly same as paper. Especially, the model architecture and hyperparameters are different.**

<br />

### Update
- Update on 2019/05/11: Provide model which could be trained and tested.
- Update on 2019/05/20: Modify model architecture
- Update on 2019/05/25: Change model architecture further
- Update on 2019/05/28: Use lower learning rate

<br />

### Todo
- [ ] Make sure the model could be trained correctly
- [ ] Provide model trained on Celeba
- [ ] Compute the classification error of each attribute

<br />

### Dependencies
* [Python 3.6+](https://www.continuum.io/downloads)
* [PyTorch 1.0.1+](http://pytorch.org/)
* [tqdm 4.31+](https://tqdm.github.io/)
* [TensorFlow 1.13+](https://www.tensorflow.org/) (optional for tensorboard)

<br />

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
$ python main.py --mode test --test_iters 200000
```

To customize configuration

Cofiguration (e.g. batch size, number of residual blocks) could be customized easily by using argparse.

<br />

### Acknowledgement
The code is mainly based on the GitHub repository StarGan (https://github.com/yunjey/stargan).
