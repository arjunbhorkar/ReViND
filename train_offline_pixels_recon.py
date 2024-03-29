#! /usr/bin/env python
import os

import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from flax.training import checkpoints
from ml_collections import config_flags

from jaxrl2.agents import PixelIQLLearner
from jaxrl2.data.dataset_utils import ReconImageDataset
from jaxrl2.evaluation_recon import evaluate

tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_string('exp_dir', 'experiments', 'Environment name.')
flags.DEFINE_string('save_dir', './tensorboard/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 300, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 15000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(2000000), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    'configs/offline_pixels_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    name = f'IQL_grassy{FLAGS.config.actor_lr}_expectile{FLAGS.config.expectile}_Ascale{FLAGS.config.A_scaling}'
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.exp_dir, FLAGS.save_dir, name))

    images_file_path = f'./{FLAGS.exp_dir}/recon_graphs_waypoints/' + name + '/'

    new_dir = "./datasets/train.pkl"
    print("Using training dataset: ", new_dir)

    dataset = ReconImageDataset(new_dir, isher=True)

    sample = dataset.sample(1, isher=True)

    kwargs = dict(FLAGS.config)
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = FLAGS.max_steps
    agent = PixelIQLLearner(FLAGS.seed, sample.observations[0][np.newaxis],
                            sample.image_observations[0][np.newaxis],
                            sample.actions[0][np.newaxis], **kwargs)

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):

        batch = dataset.sample(FLAGS.batch_size, saveimg=False, isher=True)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.scalar(f'training/{k}', v, i)
                else:
                    summary_writer.histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0 or i == 100 or i == 20000:
            eval_batch = dataset.sample(10, isher=True)

            eval_info = evaluate(agent, eval_batch,
                                 images_file_path + f'step{i}', True)

            for k, v in eval_info.items():
                summary_writer.scalar(f'evaluation/{k}', v, i)
            policy_folder = os.path.join(f'{FLAGS.exp_dir}/model_checkpoints',
                                         name)
            os.makedirs(policy_folder, exist_ok=True)

            param_dict = {
                "actor": agent._actor,
                "critic": agent._critic,
                "value": agent._value,
                "target_critic_params": agent._target_critic_params
            }
            checkpoints.save_checkpoint(policy_folder,
                                        param_dict,
                                        step=i,
                                        keep=1000)


if __name__ == '__main__':
    app.run(main)
