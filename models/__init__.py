from .vae import VAE
from .draw import Draw


def get_model(name, d, h, w):

    if name == 'vae':
        model = VAE(d, h, w)
    elif name == 'draw':
        model = Draw(d, h, w, hdim=256, zdim=100, T=10, attention=False)
    else:
        raise NotImplementedError('Unsupported Model: {}'.format(name))
    return model
