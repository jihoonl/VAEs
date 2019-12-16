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
from torch.distributions import Bernoulli, Normal
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from dataset import get_dataset
from datasets.gqn import Scene
from events import add_events
from models import get_model_genesis as get_model
from preprocess import Dummy, Quantization, Range
from utils import (device, get_logdir_name, get_ema, geco_beta_update, logger,
                   num_gpus, use_gpu)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('-m',
                        '--model',
                        default='genesis',
                        help='Model to train')
    parser.add_argument('-lr',
                        '--learning-rate',
                        help='learning rate',
                        default=1e-4,
                        type=float,
                        dest='learning_rate')
    parser.add_argument('--batch-size',
                        help='batch size',
                        default=128,
                        type=int,
                        dest='batch_size')
    parser.add_argument('--epoch', '-e', default=10, type=int, help='max epoch')
    parser.add_argument('--dataset',
                        default='rooms_ring_camera',
                        type=str,
                        help='Dataset to use')
    parser.add_argument('--data-root',
                        default='data',
                        type=str,
                        help='Dataset root to store')
    parser.add_argument('--log-root-dir',
                        default='/data/private/exp/genesis',
                        type=str,
                        help='log root')
    parser.add_argument('--log-interval', default=50, type=int, help='log root')
    parser.add_argument('--zdim',
                        default=64,
                        type=int,
                        help='latent space dimension')
    parser.add_argument('--hdim',
                        type=int,
                        default=128,
                        help='Hidden dimension')
    parser.add_argument('-nq',
                        '--no-quantization',
                        action='store_true',
                        help='Dont use Quantization as preprocesser')
    parser.add_argument('--layers',
                        type=int,
                        default=4,
                        help='Number of layers')
    parser.add_argument('--sigma', type=float, default=0.7, help='Sigma switch')
    parser.add_argument('--geco-init',
                        type=float,
                        default=1.0,
                        help='geco beta')
    parser.add_argument('--geco-alpha',
                        type=float,
                        default=0.99,
                        help='geco alpha')
    parser.add_argument('--geco-lr',
                        type=float,
                        default=1e-5,
                        help='geco learning rate')
    parser.add_argument('--geco-speedup',
                        type=float,
                        default=10.,
                        help='geco learning rate')
    parser.add_argument('--geco-goal',
                        type=float,
                        default=0.5655,
                        help='geco reconstruction goal')
    parser.add_argument('--postfix',
                        type=str,
                        default='',
                        help='Postfix of logdir')

    return parser.parse_args()


nll_ema = None
kl_ema = None


def main():
    args = parse_args()

    logger.info('Num GPU: {}'.format(num_gpus))
    logger.info('Load Dataset')
    data = get_dataset(args.dataset, args.data_root, args.batch_size)
    data1, _ = data['train'][0]

    dims = list(data1.shape)
    param = dict(
        zdim=args.zdim,
        hdim=args.hdim,
        quant=not args.no_quantization,
        layers=args.layers,
        sigma=args.sigma,
    )
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

    train_loader = DataLoader(data['train'], args.batch_size * num_gpus,
                              **kwargs)
    kwargs['shuffle'] = True
    test_loader = DataLoader(data['test'], args.batch_size * num_gpus, **kwargs)

    if not args.no_quantization:
        q = Quantization(device=device)
        raise NotImplementedError('It is using sigmoid now')
    else:
        q = Dummy()

    sigma_default = args.sigma * torch.ones(1, args.layers, 1, 1, 1)
    if use_gpu:
        sigma_default = sigma_default.cuda()
    else:
        sigma_default = sigma_default.cpu()
    #sigma_default = args.sigma

    def get_recon_error(x, sigma, x_mu_k, log_ms_k, recon):
        n = Normal(x_mu_k, sigma)
        log_x_mu = n.log_prob(x.unsqueeze(1))
        log_mx = log_x_mu + log_ms_k
        ll = torch.log(log_mx.exp().sum(dim=1))
        #ll = Normal(recon, sigma).log_prob(x)
        #ll = Bernoulli(recon).log_prob(x)
        return -ll.sum(dim=[1, 2, 3]).mean()

    def step(engine, batch):
        model.train()
        x, _ = batch
        x = x.to(device)
        x = q.preprocess(x)

        recon, recon_k, x_mu_k, log_ms_k, kl_m, kl_c = model(x)

        nll = get_recon_error(x, sigma_default, x_mu_k, log_ms_k, recon)
        kl_m = kl_m.sum(dim=[1, 2, 3, 4]).mean()
        kl_c = kl_c.sum(dim=[1, 2, 3, 4]).mean()
        optimizer.zero_grad()

        nll_ema = engine.global_info['nll_ema']
        kl_ema = engine.global_info['kl_ema']
        beta = engine.global_info['beta']

        nll_ema = get_ema(nll.detach(), nll_ema, args.geco_alpha)
        kl_ema = get_ema((kl_m + kl_c).detach(), kl_ema, args.geco_alpha)

        loss = nll + beta * (kl_c + kl_m)
        elbo = -loss
        loss.backward()
        optimizer.step()

        # GECO update
        n_pixels = x.shape[1] * x.shape[2] * x.shape[3]
        goal = args.geco_goal * n_pixels
        geco_lr = args.geco_lr
        beta = geco_beta_update(beta,
                                nll_ema,
                                goal,
                                geco_lr,
                                speedup=args.geco_speedup)

        engine.global_info['nll_ema'] = nll_ema
        engine.global_info['kl_ema'] = kl_ema
        engine.global_info['beta'] = beta

        lr = optimizer.param_groups[0]['lr']
        ret = {
            'elbo': elbo.item(),
            'nll': nll.item(),
            'kl_m': kl_m.item(),
            'kl_c': kl_c.item(),
            'lr': lr,
            'sigma': args.sigma,
            'beta': beta
        }
        return ret

    trainer = Engine(step)
    trainer.global_info = {
        'nll_ema': None,
        'kl_ema': None,
        'beta': torch.tensor(args.geco_init).to(device)
    }
    metric_names = ['elbo', 'nll', 'kl_m', 'kl_c', 'lr', 'sigma', 'beta']

    RunningAverage(output_transform=lambda x: x['elbo']).attach(trainer, 'elbo')
    RunningAverage(output_transform=lambda x: x['nll']).attach(trainer, 'nll')
    RunningAverage(output_transform=lambda x: x['kl_m']).attach(trainer, 'kl_m')
    RunningAverage(output_transform=lambda x: x['kl_c']).attach(trainer, 'kl_c')
    RunningAverage(output_transform=lambda x: x['lr']).attach(trainer, 'lr')
    RunningAverage(output_transform=lambda x: x['sigma']).attach(
        trainer, 'sigma')
    RunningAverage(output_transform=lambda x: x['beta']).attach(trainer, 'beta')

    ProgressBar().attach(trainer, metric_names=metric_names)
    Timer(average=True).attach(trainer)

    add_events(trainer, model, writer, logdir, args.log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        model.eval()

        val_elbo = 0
        val_kl_m = 0
        val_kl_c = 0
        val_nll = 0

        beta = engine.global_info['beta']
        with torch.no_grad():
            for i, (x, _) in enumerate(test_loader):
                x = x.to(device)
                x_processed = q.preprocess(x)
                recon_processed, recon_k_processed, x_mu_k, log_ms_k, kl_m, kl_c = model(
                    x_processed)
                # nll = get_recon_error(recon_processed, x_processed, args.sigma)
                nll = get_recon_error(x_processed, sigma_default, x_mu_k,
                                      log_ms_k, recon_processed)
                kl_m = kl_m.sum(dim=[1, 2, 3, 4]).mean()
                kl_c = kl_c.sum(dim=[1, 2, 3, 4]).mean()
                loss = nll + beta * (kl_m + kl_c)
                elbo = -loss

                val_elbo += elbo
                val_kl_m += kl_m
                val_kl_c += kl_c
                val_nll += nll
                if i == 0:
                    """
                    batch, *xdims = x.shape
                    row = 8
                    n = min(x.shape[0], row)
                    comparison = torch.cat([x[:n], recon[:n]])
                    grid = make_grid(comparison.detach().cpu().float(),
                                     nrow=row)
                    writer.add_image('val/reconstruction', grid,
                                     engine.state.iteration)
                    """

                    cat = []
                    max_col = args.layers + 2
                    for x1, mu1, mu1_k in zip(x_processed, recon_processed,
                                              recon_k_processed):
                        cat.extend([x1, mu1])
                        cat.extend(mu1_k)
                        if len(cat) > (max_col * 10):
                            break
                    cat = torch.stack(cat)
                    #if cat.shape[0] > max_col * 3:
                    #    cat = cat[:max_col * 3]
                    cat = q.postprocess(cat)
                    writer.add_image(
                        'val/layers', make_grid(cat.detach().cpu(),
                                                nrow=max_col),
                        engine.state.iteration)
                    cat2 = []
                    for l in log_ms_k:
                        cat2.extend(l.exp())
                        if len(cat2) > (args.layers * 10):
                            break
                    cat2 = torch.stack(cat2)
                    writer.add_image(
                        'val/masks',
                        make_grid(cat2.detach().cpu(), nrow=args.layers),
                        engine.state.iteration)
            val_elbo /= len(test_loader)
            val_kl_m /= len(test_loader)
            val_kl_c /= len(test_loader)
            val_nll /= len(test_loader)
            writer.add_scalar('val/elbo', val_elbo.item(),
                              engine.state.iteration)
            writer.add_scalar('val/beta', beta.item(), engine.state.iteration)
            writer.add_scalar('val/kl_m', val_kl_m.item(),
                              engine.state.iteration)
            writer.add_scalar('val/kl_c', val_kl_c.item(),
                              engine.state.iteration)
            writer.add_scalar('val/nll', val_nll.item(), engine.state.iteration)
            print('{:3d} /{:3d} : ELBO: {:.4f}, KL-M: {:.4f}, '
                  'KL-C: {:.4f} NLL: {:.4f}'.format(engine.state.epoch,
                                                    engine.state.max_epochs,
                                                    val_elbo, val_kl_m,
                                                    val_kl_c, val_nll))

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
