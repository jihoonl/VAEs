import logging
import os
import time

from .cuda import num_gpus

formatter = logging.Formatter('[%(levelname)s] %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p')
ch = logging.StreamHandler()
ch.setFormatter(formatter)

logger = logging.getLogger('vaes')
logger.setLevel(logging.INFO)
logger.addHandler(ch)


def set_debug(debug=True):
    global logger

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def get_logdir_name(args, param):
    t = time.strftime('%b%d-%H%M')
    c = dict(model=args.model,
             batch_size=args.batch_size,
             epoch=args.epoch,
             num_gpu=num_gpus,
             lr=args.learning_rate)
    c.update(param)

    config = '_'.join(['{}{}'.format(k, str(v)) for k, v in c.items()])
    if args.postfix:
        config += '_{}'.format(args.postfix)

    dirpath = os.path.join(args.log_root_dir, '{}_vae'.format(args.dataset),
                           config, t)
    return dirpath
