from utils import num_gpus

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint

def log_metrics(engine, writer):
    for k, v in engine.state.metrics.items():
        writer.add_scalar('training/{}'.format(k), v, engine.state.iteration)


def add_events(trainer, model, writer, logdir):
    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED,
                              handler=log_metrics,
                              writer=writer)

    checkpoint_handler = ModelCheckpoint(logdir + '/checkpoints',
                                         'epoch',
                                         save_interval=1,
                                         n_saved=100,
                                         require_empty=False,
                                         save_as_state_dict=False)
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={
            'model':
            model.module.state_dict() if num_gpus > 1 else model.state_dict()
        })

