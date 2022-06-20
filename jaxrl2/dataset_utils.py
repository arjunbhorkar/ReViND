import collections
from typing import Optional
from matplotlib import pyplot as plt
# import d4rl
# import gym
import numpy as np
import sys
# np.set_printoptions(threshold=sys.maxsize)
from recordclass import recordclass
import random
from tqdm import tqdm
import math
import pickle
import time
import os
from dm_control.utils.transformations import quat_to_euler


import cv2

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

ImageBatch = collections.namedtuple(
    'ImageBatch',
    ['observations', 'image_observations', 'actions', 'rewards', 'masks', 'next_observations', 'next_image_observations'])

def sunny_detector(img):
    low_val = np.array([0, 0, 100])
    high_val = np.array([255, 255, 255])
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_sunny = cv2.inRange(img_hsv, low_val, high_val)
    # make top half of img_sunny 0
    img_sunny[:int(img_sunny.shape[0]*2./3), :] = 0
    
    # make left third of img_sunny 0
    img_sunny[:, :int(img_sunny.shape[1]*1./3)] = 0
    # make right third of img_sunny 0
    img_sunny[:, int(img_sunny.shape[1]*2./3):] = 0

    mask = img_sunny > 0
    # create new image with img_sunny and img
    img_out = np.zeros(img.shape, dtype=np.uint8)
    img_out[:, :] = img[:, :]
    for i in range(3):
        img_out[mask, i] = img_sunny[mask]
    # return true if number of non zero mask elements greater than half
    pred = np.sum(mask) > int(mask.size/27.)
    return float(pred)

def grass_detector(img):
    low_val = np.array([28, 50, 0])
    high_val = np.array([86, 255, 255])
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_grass = cv2.inRange(img_hsv, low_val, high_val)
    # make top half of img_grass 0
    img_grass[:int(img_grass.shape[0]*2./3), :] = 0
    
    # make left third of img_grass 0
    img_grass[:, :int(img_grass.shape[1]*1./3)] = 0
    # make right third of img_grass 0
    img_grass[:, int(img_grass.shape[1]*2./3):] = 0

    mask = img_grass > 0
    # create new image with img_grass and img
    img_out = np.zeros(img.shape, dtype=np.uint8)
    img_out[:, :] = img[:, :]
    img_out[mask, 1] = img_grass[mask]
    # return true if number of non zero mask elements greater than half
    pred = np.sum(mask) > int(mask.size/27.)
    return  float(pred)

def reward_b(b):
    return 1/(1+np.exp(87-b))

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
   
    # angles.append(phi)
    # distances.append(rho)

    return np.array([np.sin(phi), np.cos(phi), rho])

def sample_goal_norm(meand, meana, stdd, stda):

    distsamp = max(np.random.normal(meand, stdd), 1)

    return [np.random.normal(meana, stda), distsamp]

def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)

def resize_image(image, scale_down):
    curr_shape = image.shape
    new_shape = (int(curr_shape[0]/scale_down), int(curr_shape[1]/scale_down))
    new_img = cv2.resize(image, new_shape, interpolation = cv2.INTER_AREA)

    return new_img

def rewardf(pos, goal, collision = False):
        pos[2] = 0
        goal[2] = 0
        a = 0
        if collision:
            a = -100/(1-0.99)
        return -np.linalg.norm(pos - goal)

class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        # self.observations = observations
        # self.actions = actions
        # self.rewards = rewards
        # self.masks = masks
        # self.dones_float = dones_float
        # self.next_observations = next_observations
        # self.size = size
        print("done formatting3")
    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])

class ImageDataset(Dataset):
    def __init__(self, dataset, image_observations: np.ndarray,
                    next_image_observations: np.ndarray,
                    dones_float: np.ndarray,):
        # self.image_observations = dataset['image_observations'].astype(np.float16)
        # self.next_image_observations = dataset['next_image_observations'].astype(np.float16)
        print("done formatting2")
        super().__init__(None,
                         actions=None,
                         rewards=None,
                         masks=None,
                         dones_float=None,
                         next_observations=None,
                         size=None)


    def sample(self, batch_size: int, isher=True, not_rand = False, saveimg = False) -> ImageBatch:
        indx = np.random.randint(self.size, size=batch_size)

        # print(indx)
        if not_rand:
            indx = []

            for i in range(self.size):
                indx.append(i)
                # if self.iscoll[i]:
                #     for j in range(8):
                #         indx.append(i-j)

            indx = np.array(indx)

            print("indx len is", len(indx))


        currobs = []
        nextobs = []
        reward = []
        mask = []

        if isher:
            # print('gotnew')
            for i in indx:
                # print(isher)
                if self.iscoll[i]:
                    goal_angle, goal_dist = sample_goal_norm(self.meand, self.meana, self.stdd, self.stda)
                    r = -1-50
                    # print(r)
                    currpolar = np.array([np.sin(goal_angle), np.cos(goal_angle), goal_dist])
                    currobs.append(np.array(currpolar))
                    nextobs.append(np.array(currpolar))
                    reward.append(r)
                    mask.append(0)

                else:
                    traji = self.traj_index[i]
                    traj = self.trajs[traji[0]][0]

                    rot = self.trajs[traji[0]][1][traji[1]]
                    nextrot = self.trajs[traji[0]][1][traji[1]+1]


                    end = min(len(traj)-3, random.randint(traji[1]+1, traji[1]+50))

                    if end == traji[1]+1:
                        mask.append(0)
                    else:
                        mask.append(1)

                    goalpt = traj[end]
                    currpt = traj[traji[1]]
                    # print(currpt.shape)
                    nextpt = traj[traji[1]+1]

                    r = -1
                    currpolar = euc2polar(currpt, goalpt, rot)
                    nextpolar = euc2polar(nextpt, goalpt, nextrot)

                    currobs.append(np.array(currpolar))
                    nextobs.append(np.array(nextpolar))
                    reward.append(r)
                

            return ImageBatch(observations=np.array(currobs),
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
                    plt.text(-1.5, -1.5, f"reward {rval} dist component {dval}, brightness component {bval}", c='green')
                    plt.imshow(self.image_observations[i])
                    name = f'traj_im{i}.png'
                    os.makedirs('/home/arjunbhorkar/datacaps', exist_ok = True)
                    plt.savefig(os.path.join('/home/arjunbhorkar/datacaps', name))

            return ImageBatch(observations=self.observations[indx],
                        image_observations=self.image_observations[indx],
                        actions=self.actions[indx],
                        rewards=self.rewards[indx],
                        masks=self.masks[indx],
                        next_observations=self.next_observations[indx],
                        next_image_observations=self.next_image_observations[indx])
        

class MujDataset(Dataset):
    def __init__(self,
                 dir: str):

        print("Loading data")
        with open(dir, 'rb') as f:
            trajectories = pickle.load(f)
        print("loaded data")

        dataset = {}
        dataset['observations'] = []
        dataset['next_observations'] = []
        dataset['actions'] = []
        dataset['rewards'] = []
        dataset['terminals'] = []

        counter = 0
        print(len(trajectories['rewards']))
        for t in range(1, len(trajectories['rewards'])):
            if trajectories['rewards'][t] != None:
                dataset['observations'].append(np.concatenate((trajectories['images_embed'][t],
                                                                np.array(trajectories['states'][t][()]['walker/body_rotation'][0]), 
                                                                trajectories['actions'][t-1], 
                                                                trajectories['spawngoal'][t][1])).reshape((9, )))
                dataset['next_observations'].append(np.concatenate((trajectories['next_images_embed'][t],
                                                                    np.array(trajectories['next_states'][t][()]['walker/body_rotation'][0]), 
                                                                    trajectories['actions'][t],
                                                                    trajectories['spawngoal'][t][1])).reshape((9, )))
                dataset['actions'].append(trajectories['actions'][t].reshape((2, )))
                dataset['rewards'].append(trajectories['rewards'][t])
                dataset['terminals'].append(int(trajectories['dones'][t]))

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

        super().__init__(dataset['observations'].astype(np.float16),
                         actions=dataset['actions'].astype(np.float16),
                         rewards=dataset['rewards'].astype(np.float16),
                         masks=1.0 - dataset['terminals'].astype(np.float16),
                         dones_float=dones_float.astype(np.float16),
                         next_observations=dataset['next_observations'].astype(np.float16),
                         size=len(dataset['observations']))

class MujImageDataset(ImageDataset):
    def __init__(self,
                 dir, 
                 half: bool = False,
                 isdir:bool = True):

        if isdir:
            print("Loading data from dir")
            with open(dir, 'rb') as f:
                trajectories = pickle.load(f)
            print("loaded data")

        else:
            trajectories = dir

        dataset = {}
        dataset['observations'] = []
        dataset['next_observations'] = []
        dataset['image_observations'] = []
        dataset['next_image_observations'] = []
        dataset['actions'] = []
        dataset['rewards'] = []
        dataset['terminals'] = []

        counter = 0

        n = len(trajectories['rewards'])

        if half:
            n = n*0.2

        print(len(trajectories['rewards']))
        for t in range(1, int(n)):
            if trajectories['rewards'][t] != None:

                polarcoors = euc2polar( trajectories['states'][t][()]['walker/body_position'][0], trajectories['spawngoal'][t][1])
                dataset['observations'].append(np.concatenate((np.array(trajectories['states'][t][()]['walker/body_rotation'][0]), 
                                                                # trajectories['actions'][t-1], 
                                                                polarcoors)))
                                                                # np.array([np.linalg.norm(trajectories['spawngoal'][t][1] - trajectories['states'][t][()]['walker/body_position'][0])]),
                                                                # trajectories['spawngoal'][t][1])))

                polarcoorsnext = euc2polar( trajectories['next_states'][t][()]['walker/body_position'][0], trajectories['spawngoal'][t][1])
                dataset['next_observations'].append(np.concatenate((np.array(trajectories['next_states'][t][()]['walker/body_rotation'][0]), 
                                                                    # trajectories['actions'][t],
                                                                    polarcoorsnext))) 
                                                                    # np.array([np.linalg.norm(trajectories['spawngoal'][t][1] - trajectories['next_states'][t][()]['walker/body_position'][0])]),
                                                                    # trajectories['spawngoal'][t][1] )))
                dataset['image_observations'].append(np.array(trajectories['states'][t][()]['walker/realsense_camera'][0]))
                dataset['next_image_observations'].append(np.array(trajectories['next_states'][t][()]['walker/realsense_camera'][0]))
                dataset['actions'].append(trajectories['actions'][t])
                dataset['rewards'].append(trajectories['rewards'][t])
                dataset['terminals'].append(int(trajectories['dones'][t]))

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

        print("done formatting")

        self.observations = dataset['observations']
        self.actions = dataset['actions'].astype(np.float16)
        self.rewards = dataset['rewards'].astype(np.float16)
        self.masks = 1.0 - dataset['terminals'].astype(np.float16)
        self.dones_float = dones_float
        self.next_observations = dataset['next_observations']
        self.size = len(dataset['observations'])

        self.image_observations = dataset['image_observations']
        self.next_image_observations = dataset['next_image_observations']

        print("done formatting1.5")
        super().__init__(None,
                         image_observations=None,
                         next_image_observations=None,
                         dones_float=None)

# Delete all the angular changes from the actions
def transform_action(actions):
    if len(actions[0]) == 10:
        return actions
    print(f'Prev action shape {len(actions)}, {len(actions[0])}')
    for i in range(len(actions)):
        actions[i] = np.delete(actions[i], [2, 5, 8, 11, 14])
    print(f'New action shape {len(actions)}, {len(actions[0])}')
    return actions

def filtertraj(trajs, r):
    print(f'last is {trajs["dones"][-1]}')
    rewards = []
    currsum = 0
    currct = 0

    trewards = []
    for i in range(len(trajs["dones"])):
        if trajs["dones"][i] == 1:
            print('got')
            temp = currsum/currct
            rewards.append(temp)
            l = [temp]*currct
            trewards +=l

            currsum = 0
            currct = 0

        currsum += r[i]
        currct += 1
    
    
    threshold = np.percentile(rewards, 100 - 20)
    print(f'thesh is {threshold}')
    toremove = []
    res = {}

    print(f'tlen is {len(trewards)} {trewards[0]} {trewards[-1]}, total len is {len(trajs["dones"])}')

    for i in range(len(trewards)):
        if trewards[i] < threshold:
            toremove.append(i)

    for k in trajs.keys():
        print(k)
        if len(np.array(trajs[k])) < len(np.array(trajs["dones"])):
            res[k] = trajs[k]
            continue
        print(np.shape(np.array(trajs[k])))
        res[k] = np.delete(np.array(trajs[k]), toremove, 0)
        print(np.shape(res[k]))

    return res


def format_filter(dir, isdir):
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

    # dataset['actions'] = trajectories["th_actions"]

    # dataset['rewards'] = trajectories["rewards"]
    dataset['terminals'] = trajectories["dones"]
    dataset['rewards'] = []
 
    distreward = []
    brightreward = []

    for val in range(len(dataset['terminals'])):
        if dataset['terminals'][val] == 1:
            distreward.append(-1)
            brightreward.append((0.5 * grass_detector(trajectories["image"][val])))
            
            dataset['rewards'].append(0)
        else:
            distreward.append(-1)
            brightreward.append((0.5 * grass_detector(trajectories["image"][val])))

            r = -1 + (0.75 * grass_detector(trajectories["image"][val]))
            dataset['rewards'].append(r)

    if filter:
        trajectories = filtertraj(trajectories, dataset['rewards'])
        return trajectories


class ReconImageDataset(ImageDataset):
    def __init__(self,
                 dir, 
                 half: bool = False,
                 isdir:bool = True, 
                 isher:bool = False,
                 filter:bool = False):

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

        # dataset['actions'] = trajectories["th_actions"]

        # dataset['rewards'] = trajectories["rewards"]
        dataset['terminals'] = trajectories["dones"]
        dataset['rewards'] = []
        

        #Old reward scheme
        # maxr = -1000000.0
        self.distreward = []
        self.brightreward = []
        # for val in range(len(dataset['terminals'])):
        #     if trajectories["dones"][val] == 0:
        #         dataset['rewards'].append(-trajectories["n_dist"][val]+0.99*trajectories["n_dist"][val+1])
        #         maxr = max(-trajectories["n_dist"][val]+0.99*trajectories["n_dist"][val+1], maxr)
        #         # print(-trajectories["n_dist"][val]+0.99*trajectories["n_dist"][val+1])
        #     else:
        #         # print(-trajectories["n_dist"][val])
        #         dataset['rewards'].append(0)
        #         maxr = max(0, maxr)

        # print("MAXR", maxr)

        # for val in range(len(dataset['rewards'])):
        #     # print(maxr)
        #     self.distreward.append(dataset['rewards'][val] / maxr)
        #     self.brightreward.append(reward_b(trajectories["bright_lvl"][val]))
        #     dataset['rewards'][val] = (dataset['rewards'][val] / maxr) + grass_detector(trajectories["image"][val]) #reward_b(trajectories["bright_lvl"][val]) #+ (dataset['rewards'][val] / maxr) #+ reward_b(trajectories["bright_lvl"][val])
        #     # print(trajectories["image"][val].shape)


        for val in range(len(dataset['terminals'])):
            if dataset['terminals'][val] == 1:
                self.distreward.append(-1)
                self.brightreward.append((0.5 * grass_detector(trajectories["image"][val])))
                
                dataset['rewards'].append(0)
            else:
                self.distreward.append(-1)
                self.brightreward.append((0.5 * grass_detector(trajectories["image"][val])))

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

        print("done formatting")

        self.observations = dataset['observations']
        self.actions = dataset['actions'].astype(np.float16)
        self.rewards = dataset['rewards'].astype(np.float16)
        self.masks = 1.0 - dataset['terminals'].astype(np.float16)
        self.dones_float = dones_float
        self.next_observations = dataset['next_observations']
        self.size = len(dataset['observations'])

        self.image_observations = dataset['image_observations']
        self.next_image_observations = dataset['next_image_observations']

        print("done formatting1.5")
        super().__init__(None,
                         image_observations=None,
                         next_image_observations=None,
                         dones_float=None)

class ConcatImgDataset(ImageDataset):
    def __init__(self,
                 dataset1, 
                 dataset2):
        self.observations = np.concatenate([dataset1.observations, dataset2.observations]) 
        self.actions = np.concatenate([dataset1.actions, dataset2.actions]) 
        self.rewards = np.concatenate([dataset1.rewards, dataset2.rewards]) 
        self.masks = np.concatenate([dataset1.masks, dataset2.masks]) 
        self.dones_float = np.concatenate([dataset1.dones_float, dataset2.dones_float]) 
        self.next_observations = np.concatenate([dataset1.next_observations, dataset2.next_observations]) 
        self.image_observations = np.concatenate([dataset1.image_observations, dataset2.image_observations]) 
        self.next_image_observations = np.concatenate([dataset1.next_image_observations, dataset2.next_image_observations]) 
        self.size = dataset1.size + dataset2.size

class PKLDataset(ImageDataset):
    def __init__(self,
                 dir: str):

        with open(dir, 'rb') as f:
            trajectories = pickle.load(f)

        dataset = {}
        dataset['observations'] = []
        dataset['next_observations'] = []
        dataset['image_observations'] = []
        dataset['next_image_observations'] = []
        dataset['actions'] = []
        dataset['rewards'] = []
        dataset['terminals'] = []

        for t in trajectories:
            # print(len(t['actions']))
            dataset['observations'].append(t['obervations'])
            dataset['next_observations'].append(t['next_observations'])
            dataset['image_observations'].append(t['image_observations'])
            dataset['next_image_observations'].append(t['next_image_observations'])
            dataset['actions'].append(t['actions'])
            dataset['rewards'].append(t['rewards'])
            dataset['terminals'].append(t['terminals'])

        for key in dataset.keys():
            dataset[key] = np.concatenate(dataset[key], axis=0)
            # print(key + " dimention is")
            # print(np.shape(dataset[key]))

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if dataset['terminals'][i] == 1:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset,
                         image_observations=None,
                         next_image_observations=None,
                         dones_float=dones_float.astype(np.float16))

class ImageReplayBuffer(MujImageDataset):
    # def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
    #              capacity: int):

    #     observations = np.empty((capacity, *observation_space.shape),
    #                             dtype=observation_space.dtype)
    #     actions = np.empty((capacity, action_dim), dtype=np.float16)
    #     rewards = np.empty((capacity, ), dtype=np.float16)
    #     masks = np.empty((capacity, ), dtype=np.float16)
    #     dones_float = np.empty((capacity, ), dtype=np.float16)
    #     next_observations = np.empty((capacity, *observation_space.shape),
    #                                  dtype=observation_space.dtype)

                            
    #     super().__init__(observations=observations,
    #                      actions=actions,
    #                      rewards=rewards,
    #                      masks=masks,
    #                      dones_float=dones_float,
    #                      next_observations=next_observations,
    #                      size=0)

    #     self.size = 0

    #     self.insert_index = 0
    #     self.capacity = capacity

    def __init__(self, data, capacity = 512):
        self.insert_index = 0
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        super().__init__(data, False, False)

        num_samples = len(self.observations)
        print(f'Replay buffer initialized with size {num_samples}')

        self.insert_index = num_samples-1
        self.size = num_samples-1
        self.capacity = num_samples-1

    def insert(self, timesteps):

        polarcoors = euc2polar( timesteps['states'][0][()]['walker/body_position'][0], timesteps['spawngoal'][0][1])
        polarcoorsnext = euc2polar( timesteps['next_states'][0][()]['walker/body_position'][0], timesteps['spawngoal'][0][1])

        self.observations[self.insert_index] = np.concatenate((np.array(timesteps['states'][0][()]['walker/body_rotation'][0]), 
                                                        # trajectories['actions'][t-1], 
                                                        polarcoors))
        self.actions[self.insert_index] = timesteps['actions'][0]
        self.rewards[self.insert_index] = timesteps['rewards'][0]
        self.masks[self.insert_index] = 1.0 - float(timesteps['dones'][0])
        self.dones_float[self.insert_index] = int(timesteps['dones'][0])
        self.next_observations[self.insert_index] = np.concatenate((np.array(timesteps['next_states'][0][()]['walker/body_rotation'][0]), 
                                                            # trajectories['actions'][t],
                                                            polarcoorsnext))
        self.image_observations[self.insert_index] = np.array(timesteps['states'][0][()]['walker/realsense_camera'][0])
        self.next_image_observations[self.insert_index] = np.array(timesteps['next_states'][0][()]['walker/realsense_camera'][0])
        
        # print(self.insert_index)
        # # print(self.image_observations)

        # plt.imshow(np.array(timesteps['states'][0][()]['walker/realsense_camera'][0]), interpolation='nearest')
        # plt.show()

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
