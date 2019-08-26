#!/usr/bin/env python
import argparse
import os

# Ignite
import torch
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import RunningAverage
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from dataset import get_dataset
from events import add_events
from models import get_model
from utils import device, get_logdir_name, logger, num_gpus, use_gpu


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('-m', '--model', default='vae', help='Model to train')
    parser.add_argument('-lr',
                        '--learning-rate',
                        help='learning rate',
                        default=1e-3,
                        type=float,
                        dest='learning_rate')
    parser.add_argument('--batch-size',
                        help='batch size',
                        default=128,
                        type=int,
                        dest='batch_size')
    parser.add_argument('--epoch', '-e', default=10, type=int, help='max epoch')
    parser.add_argument('--dataset',
                        default='mnist',
                        type=str,
                        help='Dataset to use')
    parser.add_argument('--data-root',
                        default='data',
                        type=str,
                        help='Dataset root to store')

    parser.add_argument('--log-root-dir',
                        default='log',
                        type=str,
                        help='log root')
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info('Num GPU: {}'.format(num_gpus))
    logger.info('Load Dataset')
    data = get_dataset(args.dataset, args.data_root)
    data1, _ = data['train'][0]

    dims = list(data1.shape)

    model = get_model(args.model, *dims)

    model = torch.nn.DataParallel(model) if num_gpus > 1 else model
    model.to(device)
    logger.info(model)

    kwargs = {
        'pin_memory': True if use_gpu else False,
        'shuffle': True,
        'num_workers': num_gpus * 4
    }

    logdir = get_logdir_name(args)
    logger.info('Log Dir: {}'.format(logdir))
    writer = SummaryWriter(logdir)

    os.makedirs(logdir, exist_ok=True)

    train_loader = DataLoader(data['train'], args.batch_size, **kwargs)
    kwargs['shuffle'] = False
    test_loader = DataLoader(data['test'], args.batch_size, **kwargs)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    def get_elbo(recon, x, mu, logvar):
        b, *xdims = x.shape
        bce = F.binary_cross_entropy(recon.view(b, -1), x.view(b, -1), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce, kl

    def step(engine, batch):
        model.train()
        x, _ = batch
        x = x.to(device)

        recon, mu, logvar = model(x)

        recon_error, kl = get_elbo(recon, x, mu, logvar)

        elbo = -recon_error + kl
        loss = -elbo

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']
        return {
            'elbo': elbo.item(),
            'recon_error': recon_error.item(),
            'kl': kl.item(),
            'mu': mu,
            'logvar': logvar,
            'lr': lr
        }

    trainer = Engine(step)
    metric_names = ['elbo', 'll', 'kl', 'mu', 'logvar', 'lr']

    for m in metric_names:
        RunningAverage(output_transform=lambda x: x[m]).attach(trainer, m)
    ProgressBar().attach(trainer, metric_names=metric_names)
    Timer(average=True).attach(trainer)

    add_events(trainer, model, writer, logdir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        model.eval()

        val_elbo = 0
        val_kl = 0
        val_recon_error = 0

        with torch.no_grad():
            for i, (x, _) in enumerate(test_loader):
                x = x.to(device)
                recon, mu, logvar = model(x)
                recon_error, kl = get_elbo(recon, x, mu, logvar)
                elbo = -recon_error + kl

                val_elbo += elbo
                val_kl += kl
                val_recon_error += recon_error
                if i == 0:
                    batch, *xdims = x.shape
                    row = 8
                    n = min(x.shape[0], row)
                    comparison = torch.cat([x[:n], recon[:n]])
                    grid = make_grid(comparison.detach().cpu().float(), nrow=row)
                    writer.add_image('val/reconstruction', grid,
                                     engine.state.iteration)
            val_elbo /= len(test_loader.dataset)
            val_kl /= len(test_loader.dataset)
            val_recon_error /= len(test_loader.dataset)
            writer.add_scalar('val/elbo', val_elbo.item(),
                              engine.state.iteration)
            writer.add_scalar('val/kl', val_kl.item(), engine.state.iteration)
            writer.add_scalar('val/recon_error', val_recon_error.item(),
                              engine.state.iteration)

    @trainer.on(Events.EXCEPTION_RAISED)
    def handler_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            logger.warn('KeyboardInterrupt caught. Exiting gracefully.')
        else:
            raise e

    logger.info(
        'Start training. Max epoch = {}, Batch = {}, # Trainset = {}'.format(
            args.epoch, args.batch_size, len(data['train'])))
    trainer.run(train_loader, args.epoch)
    logger.info('Done training')
    writer.close()


if __name__ == '__main__':
    main()
