from .vae import VAE, TowerVAE, TowerSBDVAE, SbdVAE
from .draw import Draw
from .conv_draw import ConvDraw
from .genesis import Genesis
from torch.optim import Adam


def get_model_vae(name, learning_rate, param, d, h, w):

    if name == 'vae':
        model = VAE(d, h, w, **param)
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif name == 'sbd_vae':
        model = SbdVAE(d, h, w, **param)
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif name == 'tower':
        model = TowerVAE(d, h, w, **param)
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif name == 'tower_sbd':
        model = TowerSBDVAE(d, h, w, **param)
        optimizer = Adam(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError('Unsupported Model: {}'.format(name))
    return model, optimizer

def get_model_draw(name, learning_rate, param, d, h, w):

    if name == 'draw':
        model = Draw(d, h, w, **param)
        optimizer = Adam(model.parameters(),
                         lr=learning_rate,
                         betas=(0.5, 0.99))
    elif name == 'conv_draw':
        model = ConvDraw(d, h, w, **param)
        optimizer = Adam(model.parameters(),
                         lr=learning_rate,
                         betas=(0.5, 0.99))
    else:
        raise NotImplementedError('Unsupported Model: {}'.format(name))
    return model, optimizer

def get_model_genesis(name, learning_rate, param, d, h, w):

    if name == 'genesis':
        model = Genesis(d, h, w, **param)
        optimizer = Adam(model.parameters(),
                         lr=learning_rate)
    else:
        raise NotImplementedError('Unsupported Model: {}'.format(name))
    return model, optimizer
