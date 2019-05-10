from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class Loader(data.Dataset):
    """
    Dataset class for the CelebA dataset
    """
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode, seed):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.seed = seed
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """
        Preprocess the CelebA attribute file
        """
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()

        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(self.seed)
        random.shuffle(lines)
        cnt = 0
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            # avoid ambiguous label
            if len(self.selected_attrs) > 1 and not any(label):
                continue
            cnt += 1

            if cnt <= 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Build dataset with attributes:', ' '.join(self.selected_attrs))
        print("Train dataset: {} images.".format(len(self.train_dataset)))
        print("Test dataset: {} images.".format(len(self.test_dataset)))
        print('\n')

    def __getitem__(self, index):
        """
        Return one image and its corresponding attribute label
        """
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """
        Return the number of images
        """
        return self.num_images


def get_loader(image_dir='./data/celeba/images', attr_path='./data/celeba/list_attr_celeba.txt',
               selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair'], crop_size=178,
               image_size=128, batch_size=8, mode='train', num_workers=1, seed=246):
    """
    Build a data loader
    """
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = Loader(image_dir, attr_path, selected_attrs, transform, mode, seed)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

if __name__ == '__main__':
    """
    test code
    """
    data_loader = get_loader()
    data_iter = iter(data_loader)
    x_fixed, c_org = next(data_iter)
    print(x_fixed)
    print(x_fixed.size())
    print(c_org)
    print(c_org.size())
