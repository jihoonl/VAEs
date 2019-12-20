from torchvision import datasets, transforms

from datasets.clevr import CLEVRVAE
from datasets.gqn import GQNDataset, DATASETS as GQNDATASETS


def mnist(data_root, batch_size):
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


def cifar10(data_root, batch_size):
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


def svhn(data_root, batch_size):
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


def clevr(data_root, batch_size):
    data = {}
    data['train'] = CLEVRVAE(mode='train', length=batch_size * 1024)
    data['test'] = CLEVRVAE(mode='test', length=batch_size)
    return data


def gqn(name, data_root, batch_size):
    use_cache = True
    gqn_root = '/data/public/rw/datasets/gqn/torch'
    data = {}
    data['train'] = GQNDataset(gqn_root,
                               name,
                               'train',
                               use_cache=use_cache,
                               #length=batch_size * 4)
                               length=batch_size * 1024)
    data['test'] = GQNDataset(gqn_root,
                              name,
                              'test',
                              use_cache=use_cache,
                              length=batch_size)
    return data


dataset_pool = {
    'mnist': mnist,
    'cifar10': cifar10,
    'svhn': svhn,
    'clevr': clevr
}


def get_dataset(name, data_root, batch_size):

    if name in dataset_pool.keys():
        return dataset_pool[name](data_root, batch_size)
    elif name in GQNDATASETS:
        return gqn(name, data_root, batch_size)
    else:
        raise NotImplementedError('No dataset')
