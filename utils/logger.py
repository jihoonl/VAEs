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


def get_logdir_name(args):
    t = time.strftime('%b%d-%H%M')
    c = dict(model=args.model,
             batch_size=args.batch_size,
             num_gpu=num_gpus,
             lr=args.learning_rate)
    config = '_'.join(['{}{}'.format(k, str(v)) for k, v in c.items()])
    dirpath = os.path.join(args.log_root_dir, config, t)
    return dirpath
