#! /usr/bin/env python
import random
import numpy as np
import time
from os.path import isfile, join
from evaluation_recon import evaluate
from jaxrl2.dataset_utils import MujImageDataset, ImageBatch, ConcatImgDataset, ReconImageDataset, format_filter
from jaxrl2.agents import PixelIQLLearner
import pickle
import os
import tensorflow as tf
import tqdm
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags
from flax.training import checkpoints
# from collect_data import collect
import gc

tf.config.experimental.set_visible_devices([], "GPU")


fine_tine_iterations = 3

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'mujoco_car_dcist', 'Environment name.')
flags.DEFINE_string('save_dir', './grace_recon3/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 300, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 15000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 512, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(2000000), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e3),
                     'Number of training steps to start training.')
flags.DEFINE_integer('data_pull_interval', int(2000000),
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('image_size', 64, 'Image size.')
flags.DEFINE_integer('num_stack', 3, 'Stack frames.')
flags.DEFINE_integer('replay_buffer_size', None,
                     'Number of training steps to start training.')
flags.DEFINE_integer(
    'action_repeat', None,
    'Action repeat, if None, uses 2 or PlaNet default values.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'configs/offline_pixels_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    name = f'IQL_collision_herfinal_updatedn50_Bimp_HER_lr{FLAGS.config.actor_lr}_expectile{FLAGS.config.expectile}_Ascale{FLAGS.config.A_scaling}'
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, name))

    images_file_path = './recon_graphs_waypoints/' + name + '/'

    # bag_dir = "/nfs/kun2/users/arjun/pklcollision2"
    # Good traj ac space newer
    # bag_dir = "/home/arjunbhorkar/pkl_bright1"

    # vel ac space
    bag_dir = "/nfs/kun2/users/arjun/pkl_bright_newacs_avg"

    bag_list = [str(bag_dir + '/' + f) for f in os.listdir(bag_dir)
                if isfile(join(bag_dir, f)) and "traj" in f]
    # new_dir = random.choice(bag_list)
    new_dir = "/nfs/kun2/users/arjun/recon_dataset/recon_datavis/pklher3/traj_train.pkl"
    print("Using training dataset: ", new_dir)

    # FC

    # t = format_filter(new_dir, True)
    # dataset1 = ReconImageDataset(t, isdir=False)

    # FC
    # notfc

    dataset1 = ReconImageDataset(new_dir, isher=True)

    # NOTFC

    samp_a = dataset1.sample(1, isher=True)

    print(samp_a.observations[0][np.newaxis].shape)
    print(samp_a.image_observations[0][np.newaxis].shape)
    print(samp_a.actions[0][np.newaxis].shape)

    kwargs = dict(FLAGS.config)
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = FLAGS.max_steps
    agent = PixelIQLLearner(FLAGS.seed, samp_a.observations[0][np.newaxis],
                            samp_a.image_observations[0][np.newaxis],
                            samp_a.actions[0][np.newaxis], **kwargs)

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):

        # , not_rand=True, saveimg=True)
        batch = dataset1.sample(FLAGS.batch_size, saveimg=False, isher=True)
        update_info = agent.update(batch)

        if i % FLAGS.data_pull_interval == 0:
            new_dir = random.choice(bag_list)
            print("Using dataset: ", new_dir)

            dataset1 = ReconImageDataset(new_dir)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.scalar(f'training/{k}', v,
                                          i)
                else:
                    summary_writer.histogram(f'training/{k}', v,
                                             i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0 or i == 100 or i == 20000:
            eval_batch = dataset1.sample(7000, isher=True)

            eval_info = evaluate(
                agent, eval_batch, images_file_path+f'step{i}', True)

            # running_reward += rewards
            for k, v in eval_info.items():
                summary_writer.scalar(f'evaluation/{k}', v, i)
            # summary_writer.scalar(f'evaluation/running rewards average', np.mean(running_reward), i)
            # summary_writer.histogram(f'evaluation/running rewards', running_reward, i)

            policy_folder = os.path.join('policies_off_exp_5', name)
            os.makedirs(policy_folder, exist_ok=True)

            param_dict = {"actor": agent._actor, "critic": agent._critic,
                          "value": agent._value, "target_critic_params": agent._target_critic_params}
            checkpoints.save_checkpoint(
                policy_folder, param_dict, step=i, keep=1000)


if __name__ == '__main__':
    app.run(main)
