from .vae import VAE, TowerVAE, TowerSBDVAE, SbdVAE
from .draw import Draw
from .conv_draw import ConvDraw
from torch.optim import Adam


def get_model(name, learning_rate, param, d, h, w):

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
    elif name == 'draw':
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
