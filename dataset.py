from torchvision import datasets, transforms
from datasets.clevr import CLEVRVAE


def mnist(data_root):
    data = {}
    data['train'] = datasets.MNIST(data_root,
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())
    data['test'] = datasets.MNIST(data_root,
                                  train=False,
                                  download=True,
                                  transform=transforms.ToTensor())
    return data


def cifar10(data_root):
    data = {}
    data['train'] = datasets.CIFAR10(data_root,
                                     train=True,
                                     download=True,
                                     transform=transforms.ToTensor())
    data['test'] = datasets.CIFAR10(data_root,
                                    train=False,
                                    download=True,
                                    transform=transforms.ToTensor())
    return data


def svhn(data_root):
    data = {}
    data['train'] = datasets.SVHN(data_root,
                                  split='train',
                                  download=True,
                                  transform=transforms.ToTensor())
    data['test'] = datasets.SVHN(data_root,
                                 split='test',
                                 download=True,
                                 transform=transforms.ToTensor())
    return data


def clevr(data_root):
    data = {}
    data['train'] = CLEVRVAE(mode='train', length=128 * 1024)
    data['test'] = CLEVRVAE(mode='test', length=128 * 6)
    return data


dataset_pool = {'mnist': mnist, 'cifar10': cifar10, 'svhn': svhn, 'clevr': clevr}


def get_dataset(name, data_root):
    return dataset_pool[name](data_root)
