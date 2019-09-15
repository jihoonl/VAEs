#!/usr/bin/env python
import argparse
import os

# Ignite
import torch
from torch.distributions import Bernoulli, Normal
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import RunningAverage
from tensorboardX import SummaryWriter
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
                        default='/data/private/exp/mnist_vae',
                        type=str,
                        help='log root')
    parser.add_argument('--log-interval', default=50, type=int, help='log root')
    parser.add_argument('--zdim',
                        default=10,
                        type=int,
                        help='latent space dimension')
    parser.add_argument('--hdim',
                        type=int,
                        default=400,
                        help='LSTM Hidden dimension')
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info('Num GPU: {}'.format(num_gpus))
    logger.info('Load Dataset')
    data = get_dataset(args.dataset, args.data_root)
    data1, _ = data['train'][0]

    dims = list(data1.shape)
    param = dict(zdim=args.zdim,
                 hdim=args.hdim)
    model, optimizer = get_model(args.model, args.learning_rate, param, *dims)

    model = torch.nn.DataParallel(model) if num_gpus > 1 else model
    model.to(device)
    logger.info(model)

    kwargs = {
        'pin_memory': True if use_gpu else False,
        'shuffle': True,
        'num_workers': num_gpus * 4
    }

    logdir = get_logdir_name(args, param)
    logger.info('Log Dir: {}'.format(logdir))
    writer = SummaryWriter(logdir)

    os.makedirs(logdir, exist_ok=True)

    train_loader = DataLoader(data['train'], args.batch_size, **kwargs)
    kwargs['shuffle'] = False
    test_loader = DataLoader(data['test'], args.batch_size, **kwargs)

    def get_recon_error(recon, x):
        ll = Bernoulli(recon).log_prob(x)
        return -ll.sum()

    def step(engine, batch):
        model.train()
        x, _ = batch
        x = x.to(device)

        recon, kl = model(x)

        nll = get_recon_error(recon, x)
        loss = nll + kl
        elbo = -loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        lr = optimizer.param_groups[0]['lr']
        ret = {
            'elbo': elbo.item() / len(x),
            'nll': nll.item() / len(x),
            'kl': kl.item() / len(x),
            'lr': lr
        }
        return ret

    trainer = Engine(step)
    metric_names = ['elbo', 'nll', 'kl', 'lr']

    RunningAverage(output_transform=lambda x: x['elbo']).attach(trainer, 'elbo')
    RunningAverage(output_transform=lambda x: x['nll']).attach(trainer, 'nll')
    RunningAverage(output_transform=lambda x: x['kl']).attach(trainer, 'kl')
    RunningAverage(output_transform=lambda x: x['lr']).attach(trainer, 'lr')

    ProgressBar().attach(trainer, metric_names=metric_names)
    Timer(average=True).attach(trainer)

    add_events(trainer, model, writer, logdir, args.log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        model.eval()

        val_elbo = 0
        val_kl = 0
        val_nll = 0

        with torch.no_grad():
            for i, (x, _) in enumerate(test_loader):
                x = x.to(device)
                recon, kl = model(x)
                nll = get_recon_error(recon, x)
                loss = nll + kl
                elbo = -loss

                val_elbo += elbo
                val_kl += kl
                val_nll += nll
                if i == 0:
                    batch, *xdims = x.shape
                    row = 8
                    n = min(x.shape[0], row)
                    comparison = torch.cat([x[:n], recon[:n]])
                    grid = make_grid(comparison.detach().cpu().float(),
                                     nrow=row)
                    writer.add_image('val/reconstruction', grid,
                                     engine.state.iteration)
            val_elbo /= len(test_loader.dataset)
            val_kl /= len(test_loader.dataset)
            val_nll /= len(test_loader.dataset)
            writer.add_scalar('val/elbo', val_elbo.item(),
                              engine.state.iteration)
            writer.add_scalar('val/kl', val_kl.item(), engine.state.iteration)
            writer.add_scalar('val/nll', val_nll.item(), engine.state.iteration)
            print('{:3d} /{:3d} : ELBO: {:.4f}, KL: {:.4f}, NLL: {:.4f}'.format(
                engine.state.epoch, engine.state.max_epochs, val_elbo, val_kl,
                val_nll))

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
