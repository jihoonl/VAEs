from .vae import VAE
from .draw import Draw
from torch.optim import Adam


def get_model(name, learning_rate, zdim, d, h, w):

    if name == 'vae':
        model = VAE(d, h, w, zdim=zdim)
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif name == 'draw':
        model = Draw(d, h, w, hdim=256, zdim=zdim, T=10, attention=False)
        optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.99))
    else:
        raise NotImplementedError('Unsupported Model: {}'.format(name))
    return model, optimizer
