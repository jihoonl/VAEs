from torchvision import datasets, transforms


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


dataset_pool = {'mnist': mnist, 'cifar10': cifar10}


def get_dataset(name, data_root):
    return dataset_pool[name](data_root)
