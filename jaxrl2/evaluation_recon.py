import os
import warnings
from typing import Dict

import cv2
import flax.linen as nn
import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image

import jaxrl2.np_utils as np_utils

warnings.filterwarnings("ignore", category=DeprecationWarning)


def rectify_and_resize(image, shape, rectify=False):
    if rectify:
        # jackal camera intrinsics
        fx, fy, cx, cy = 272.547000, 266.358000, 320.000000, 240.000000
        K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        D = np.array([[-0.038483, -0.010456, 0.003930, -0.001007]]).T
        balance = 0.5

        if len(image.shape) == 4:
            return np.array(
                [rectify_and_resize(im_i, shape) for im_i in image])

        image = np_utils.imrectify_fisheye(image, K, D, balance=balance)

    image = np_utils.imresize(image, shape)

    return image


def project_points(xy):
    batch_size, horizon, _ = xy.shape

    fx, fy, cx, cy = 272.547000, 266.358000, 320.000000, 220.000000

    # camera is ~0.35m above ground
    xyz = np.concatenate([xy, -0.95 * np.ones(list(xy.shape[:-1]) + [1])],
                         axis=-1)  # -0.35
    rvec = tvec = (0, 0, 0)
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    k1, k2, p1, p2 = [-0.038483, -0.010456, 0.003930, -0.001007]
    k3 = k4 = k5 = k6 = 0.0
    dist_coeffs = (k1, k2, p1, p2, k3, k4, k5, k6)

    xyz[..., 0] += 0.45
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(xyz_cv.reshape(batch_size * horizon, 3), rvec,
                              tvec, camera_matrix, dist_coeffs)
    uv = uv.reshape(batch_size, horizon, 2)

    return uv


def evaluate(agent, batch, dir, istraj=True) -> Dict[str, float]:
    stats = {'norm diff': []}
    norms = []

    for i in range(len(batch.observations)):
        obs = batch.observations[i]
        img = batch.image_observations[i]

        pred_action = agent.eval_actions(obs, img)

        actual_waypoints = batch.actions[i]

        if istraj:
            pred = []
            actual = []
            for j in range(0, len(pred_action), 2):
                pred.append([pred_action[j + 0], pred_action[j + 1]])
                actual.append(
                    [actual_waypoints[j + 0], actual_waypoints[j + 1]])
            pts = np.array([pred, actual])
            print("horizon shape is", pts.shape)
            uv = project_points(pts)
            pred = uv[0]
            real = uv[1]

            img = rectify_and_resize(img, (480, 640, 3))
            plt.clf()
            x, y = zip(*pred)
            x1, y1 = zip(*real)
            plt.plot(x, y, color='red')
            plt.plot(x1, y1, color='green')
            plt.imshow(img)
            name = f'traj_im{i}.png'
            os.makedirs(dir, exist_ok=True)
            plt.savefig(os.path.join(dir, name))

            plt.clf()
            pred = pts[0]
            real = pts[1]
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            x, y = zip(*pred)
            x1, y1 = zip(*real)
            plt.plot(x, y, color='red')
            plt.plot(x1, y1, color='green')
            name = f'traj{i}.png'
            os.makedirs(dir, exist_ok=True)
            plt.savefig(os.path.join(dir, name))

        norms.append(np.linalg.norm(actual_waypoints - pred_action))

    stats['norm diff'] = np.mean(norms)

    return stats
