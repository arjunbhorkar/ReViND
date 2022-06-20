import collections
from matplotlib import pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import pickle
import time
import os
import cv2

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

ImageBatch = collections.namedtuple('ImageBatch', [
    'observations', 'image_observations', 'actions', 'rewards', 'masks',
    'next_observations', 'next_image_observations'
])


def sunny_detector(img):
    low_val = np.array([0, 0, 100])
    high_val = np.array([255, 255, 255])
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_sunny = cv2.inRange(img_hsv, low_val, high_val)
    # make top half of img_sunny 0
    img_sunny[:int(img_sunny.shape[0] * 2. / 3), :] = 0

    # make left third of img_sunny 0
    img_sunny[:, :int(img_sunny.shape[1] * 1. / 3)] = 0
    # make right third of img_sunny 0
    img_sunny[:, int(img_sunny.shape[1] * 2. / 3):] = 0

    mask = img_sunny > 0
    # create new image with img_sunny and img
    img_out = np.zeros(img.shape, dtype=np.uint8)
    img_out[:, :] = img[:, :]
    for i in range(3):
        img_out[mask, i] = img_sunny[mask]
    # return true if number of non zero mask elements greater than half
    pred = np.sum(mask) > int(mask.size / 27.)
    return float(pred)


def grass_detector(img):
    low_val = np.array([28, 50, 0])
    high_val = np.array([86, 255, 255])
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_grass = cv2.inRange(img_hsv, low_val, high_val)
    # make top half of img_grass 0
    img_grass[:int(img_grass.shape[0] * 2. / 3), :] = 0

    # make left third of img_grass 0
    img_grass[:, :int(img_grass.shape[1] * 1. / 3)] = 0
    # make right third of img_grass 0
    img_grass[:, int(img_grass.shape[1] * 2. / 3):] = 0

    mask = img_grass > 0
    # create new image with img_grass and img
    img_out = np.zeros(img.shape, dtype=np.uint8)
    img_out[:, :] = img[:, :]
    img_out[mask, 1] = img_grass[mask]
    # return true if number of non zero mask elements greater than half
    pred = np.sum(mask) > int(mask.size / 27.)
    return float(pred)


def euc2polar(curr, goal, rot):

    curr = np.array(curr[:2])
    goal = np.array(goal[:2])

    total = np.array([curr, goal, goal])
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]).T
    currac = np.dot(R, total.T).T
    currac = currac - currac[0]

    [x, y] = currac[1][0], currac[1][1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return np.array([np.sin(phi), np.cos(phi), rho])


class Dataset(object):

    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        pass

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class ImageDataset(Dataset):

    def __init__(
        self,
        dataset,
        image_observations: np.ndarray,
        next_image_observations: np.ndarray,
        dones_float: np.ndarray,
    ):
        super().__init__(None,
                         actions=None,
                         rewards=None,
                         masks=None,
                         dones_float=None,
                         next_observations=None,
                         size=None)

    def sample(self,
               batch_size: int,
               isher=True,
               not_rand=False,
               saveimg=False) -> ImageBatch:
        indx = np.random.randint(self.size, size=batch_size)

        if not_rand:
            indx = []
            for i in range(self.size):
                indx.append(i)
            indx = np.array(indx)
            print("indx len is", len(indx))

        currobs = []
        nextobs = []
        reward = []
        mask = []

        if isher:
            for i in indx:
                if self.dones_float[i] == 1:
                    r = 0
                    currobs.append(self.observations[i])
                    nextobs.append(self.next_observations[i])
                    reward.append(r)
                    mask.append(0)

                else:
                    traji = self.traj_index[i]
                    traj = self.trajs[traji[0]][0]

                    rot = self.trajs[traji[0]][1][traji[1]]
                    nextrot = self.trajs[traji[0]][1][traji[1] + 1]

                    end = min(
                        len(traj) - 3,
                        random.randint(traji[1] + 10, traji[1] + 50))

                    if end == traji[1] + 1:
                        mask.append(0)
                    else:
                        mask.append(1)

                    goalpt = traj[end]
                    currpt = traj[traji[1]]
                    nextpt = traj[traji[1] + 1]

                    r = -1 + (0.75 *
                              sunny_detector(self.image_observations[i]))
                    currpolar = euc2polar(currpt, goalpt, rot)
                    nextpolar = euc2polar(nextpt, goalpt, nextrot)

                    currobs.append(np.array(currpolar))
                    nextobs.append(np.array(nextpolar))
                    reward.append(r)

            return ImageBatch(
                observations=np.array(currobs),
                image_observations=self.image_observations[indx],
                actions=self.actions[indx],
                rewards=np.array(reward),
                masks=np.array(mask),
                next_observations=np.array(nextobs),
                next_image_observations=self.next_image_observations[indx])
        else:
            if saveimg:
                for i in indx:
                    bval = round(self.brightreward[i], 3)
                    dval = round(self.distreward[i], 3)
                    rval = round(self.rewards[i], 3)
                    plt.clf()
                    # plt.text(-3.5, -3.5, currname, c='black')
                    plt.text(
                        -1.5,
                        -1.5,
                        f"reward {rval} dist component {dval}, brightness component {bval}",
                        c='green')
                    plt.imshow(self.image_observations[i])
                    name = f'traj_im{i}.png'
                    os.makedirs('/home/arjunbhorkar/datacaps', exist_ok=True)
                    plt.savefig(
                        os.path.join('/home/arjunbhorkar/datacaps', name))

            return ImageBatch(
                observations=self.observations[indx],
                image_observations=self.image_observations[indx],
                actions=self.actions[indx],
                rewards=self.rewards[indx],
                masks=self.masks[indx],
                next_observations=self.next_observations[indx],
                next_image_observations=self.next_image_observations[indx])


# Delete all the angular changes from the actions
def transform_action(actions):
    if len(actions[0]) == 10:
        return actions
    print(f'Prev action shape {len(actions)}, {len(actions[0])}')
    for i in range(len(actions)):
        actions[i] = np.delete(actions[i], [2, 5, 8, 11, 14])
    print(f'New action shape {len(actions)}, {len(actions[0])}')
    return actions


class ReconImageDataset(ImageDataset):

    def __init__(self,
                 dir,
                 half: bool = False,
                 isdir: bool = True,
                 isher: bool = False,
                 filter: bool = False):

        if isdir:
            print("Loading data from dir")
            with open(dir, 'rb') as f:
                trajectories = pickle.load(f)
            print("loaded data")

        else:
            trajectories = dir

        dataset = {}
        dataset['observations'] = trajectories["states"]
        dataset['next_observations'] = trajectories["next_states"]
        dataset['image_observations'] = trajectories["image"]
        dataset['next_image_observations'] = trajectories["next_image"]
        dataset['actions'] = transform_action(trajectories["actions"])

        dataset['terminals'] = trajectories["dones"]
        dataset['rewards'] = []

        self.distreward = []
        self.brightreward = []

        for val in range(len(dataset['terminals'])):
            if dataset['terminals'][val] == 1:
                self.distreward.append(-1)
                self.brightreward.append(
                    (0.5 * grass_detector(trajectories["image"][val])))

                dataset['rewards'].append(0)
            else:
                self.distreward.append(-1)
                self.brightreward.append(
                    (0.5 * grass_detector(trajectories["image"][val])))

                r = -1 + (0.75 * grass_detector(trajectories["image"][val]))
                dataset['rewards'].append(r)

        if isher:
            self.iscoll = trajectories["iscoll"]
            self.traj_index = trajectories["traj_index"]

            self.meana = trajectories['collamean']
            self.meand = trajectories['colldmean']
            self.stda = trajectories['collastd']
            self.stdd = trajectories['colldstd']
            self.trajs = trajectories['trajectory']

        print(len(trajectories['rewards']))

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])
            print(np.shape(dataset[key]))

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if dataset['terminals'][i] == 1:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        self.observations = dataset['observations']
        self.actions = dataset['actions'].astype(np.float16)
        self.rewards = dataset['rewards'].astype(np.float16)
        self.masks = 1.0 - dataset['terminals'].astype(np.float16)
        self.dones_float = dones_float
        self.next_observations = dataset['next_observations']
        self.size = len(dataset['observations'])

        self.image_observations = dataset['image_observations']
        self.next_image_observations = dataset['next_image_observations']

        super().__init__(None,
                         image_observations=None,
                         next_image_observations=None,
                         dones_float=None)
