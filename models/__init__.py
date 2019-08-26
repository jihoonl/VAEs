from .vae import VAE


def get_model(name, d, h, w):

    if name == 'vae':
        model = VAE(d, h, w)
    else:
        raise NotImplementedError('Unsupported Model: {}'.format(name))
    return model
