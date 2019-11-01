from .vae import VAE, TowerVAE
from .draw import Draw
from torch.optim import Adam


def get_model(name, learning_rate, param, d, h, w):

    if name == 'vae':
        model = VAE(d, h, w, **param)
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif name == 'tower_vae':
        model = TowerVAE(d, h, w, **param)
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif name == 'draw':
        model = Draw(d, h, w, **param)
        optimizer = Adam(model.parameters(),
                         lr=learning_rate,
                         betas=(0.5, 0.99))
    else:
        raise NotImplementedError('Unsupported Model: {}'.format(name))
    return model, optimizer
