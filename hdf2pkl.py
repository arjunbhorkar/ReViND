from cgitb import small
from turtle import color
import h5py
import pickle
import argparse
from PIL import Image
import numpy as np
import np_utils
import time
import os
import matplotlib.pyplot as plt
import cv2
import tqdm
import random
from recon_datavis.utils import get_files_ending_with, bytes2im


FOLDER = "../recon_release/"
save_folder = 'pklher3'
hdf5_fnames = get_files_ending_with(FOLDER, '.hdf5')
random.shuffle(hdf5_fnames)

names1 = []
names2 = []

mid = int(len(hdf5_fnames) * 0.75)

for s1 in range(mid):
    names1.append(hdf5_fnames[s1])

for s2 in range(mid, len(hdf5_fnames)):
    names2.append(hdf5_fnames[s2])

STATE_TOPICS = []
image_topics = []
angles = []
distances = []

pkl_count = 0
collision_horizon = 2

def reward(pos, goal, collision = False):
        pos[2] = 0
        goal[2] = 0
        a = 0
        if collision:
            a = -100/(1-0.99)
        return -np.linalg.norm(pos - goal)

def rectify_and_resize(image, shape, rectify=True):
    if rectify:
        fx, fy, cx, cy = 272.547000, 266.358000, 320.000000, 240.000000
        K = np.array([[fx, 0., cx],
                            [0., fy, cy],
                            [0., 0., 1.]])
        D = np.array([[-0.038483, -0.010456, 0.003930, -0.001007]]).T
        balance = 0.5
        if len(image.shape) == 4:
            return np.array([rectify_and_resize(im_i, shape) for im_i in image])
        image = np_utils.imrectify_fisheye(image, K, D, balance=balance)
    image = np_utils.imresize(image, shape)
    return image

def rot(theta, pt):
    x, y, _ = pt
    x_new = np.cos(theta)*x - np.sin(theta)*y
    y_new = np.sin(theta)*x + np.cos(theta)*y
    return [x_new, y_new]

def pos2actions(pos, rotations):
    actions = []
    counter = 0
    for i in range(len(pos)):
        if counter > 0:
            break
        counter += 1
        currrot = []
        currac = []
        curr = []
        if i+6 > len(pos):
            currac += pos[i:len(pos)]
            currrot += rotations[i:len(pos)]
            rem = 6 - (len(pos) - i)
            currac += [pos[-1]]*rem
            currrot += [rotations[-1]]*rem
        else:
            currac += pos[i:i+6]
            currrot += rotations[i:i+6]
        currpt = currac[0]
        currac = np.array(currac)
        currac = np.delete(currac, 2, 1)
        temp = currac[0]
        R = np.array([[np.cos(currrot[0]), -np.sin(currrot[0])], [np.sin(currrot[0]), np.cos(currrot[0])]]).T
        currac = np.dot(R, currac.T).T
        currac = currac - currac[0]    
        if(currac[0][0] != 0 and currac[0][1] != 0):
            print("err")
        for i in range(1, len(currac)):
            curr.append(currac[i][0] + np.random.normal(0, 0.007))
            curr.append(currac[i][1] + np.random.normal(0, 0.007))
            curr.append(currrot[0] - currrot[i] + np.random.normal(0, 0.007))
        actions.append(curr)
    return actions

def project_points(xy):
    """
    :param xy: [batch_size, horizon, 2]
    :return: [batch_size, horizon, 2]
    """
    batch_size, horizon, _ = xy.shape
    fx, fy, cx, cy = 272.547000, 266.358000, 320.000000, 220.000000
    # camera is ~0.35m above ground
    xyz = np.concatenate(
        [xy, -0.95 * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    ) 
    rvec = tvec = (0, 0, 0)
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    k1, k2, p1, p2 = [-0.038483, -0.010456, 0.003930, -0.001007]
    k3 = k4 = k5 = k6 = 0.0
    dist_coeffs = (k1, k2, p1, p2, k3, k4, k5, k6)
    xyz[..., 0] += 0.45 
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(batch_size, horizon, 2)
    return uv


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
    angles.append(phi)
    distances.append(rho)
    return np.array([np.sin(phi), np.cos(phi), rho])

def sample_goal_norm(angles, dist):
    meand = np.mean(dist)
    meana = np.mean(angles)
    vard = np.sqrt(np.var(dist))
    vara = np.sqrt(np.var(angles))
    distsamp = max(np.random.normal(meand, vard), 1)
    return [np.random.normal(meana, vara), distsamp]

def sample_many(mean, std, name):
    l = []
    for i in range(10000):
        l.append(np.random.normal(mean, std))
    plt.clf()
    plt.hist(l, bins=100)
    plt.savefig(f"./distribution/curr{name}_{len(l)}.png")

data = dict(states=[],
            actions=[],
            next_states=[],
            trajectory=[],
            traj_index=[],
            iscoll=[],
            rewards=[],
            dones=[],
            image=[],
            next_image=[],
            spawngoal=[],
            colldmean=0.0,
            colldstd=0.0,
            collamean=0.0,
            collastd=0.0)

traj_index = 0

for i in tqdm.tqdm(range(100)):
    print("Using file:", names1[i])
    curr_hdf5 = None
    try:
        curr_hdf5 = h5py.File(names1[i], 'r')
    except OSError:
        print("Coulndt open this hdf5")
        continue
    full_len = len(curr_hdf5['collision/any'])

    if full_len < 30:
        continue
    print("Using full lenght :", full_len)

    points = curr_hdf5['jackal/position']
    rotations = curr_hdf5['jackal/yaw']
    images = curr_hdf5['images/rgb_left']
    collisions = curr_hdf5['collision/physical']

    data['trajectory'].append([points[:full_len], rotations[:full_len]])

    for j in range(0, full_len-1):
        goal_step = np.random.randint(0, 30)
        end = min(j+goal_step, full_len-1)

        startpt = points[j]
        startrot = rotations[j]
        goalpt = points[end]
        goalyaw = rotations[end]

        actions = []

        if any([collisions[k] for k in range(j, min(j+collision_horizon, full_len))]):
            goal_angle, goal_dist = sample_goal_norm(angles, distances)
            r = -goal_dist - (100/(1-0.99))
            
            currpolar = np.array([np.sin(goal_angle), np.cos(goal_angle), goal_dist])
            nextpolar = np.array([np.sin(goal_angle), np.cos(goal_angle), goal_dist])
            
            data["states"].append(np.array(currpolar))
            data["next_states"].append(np.array(nextpolar))
            data["rewards"].append(r)
            data["dones"].append(1)
            data["traj_index"].append([traj_index, j])
            print(traj_index)
            data["iscoll"].append(True)
            data["image"].append(rectify_and_resize(bytes2im(images[j]), (48, 64, 3), False))
            data["next_image"].append(rectify_and_resize(bytes2im(images[j]), (48, 64, 3), False))

        else:
            r = reward(points[j], goalpt)
            currpolar = euc2polar(points[j], goalpt, rotations[j])
            nextpolar = euc2polar(points[j+1], goalpt, rotations[j+1])
            
            data["states"].append(np.array(currpolar))
            data["next_states"].append(np.array(nextpolar))
            data["rewards"].append(r)
            if goal_step < 2:
                dns = 1
            else:
                dns = 0
            data["dones"].append(dns)
            data["traj_index"].append([traj_index, j])
            data["iscoll"].append(False)
            data["image"].append(rectify_and_resize(bytes2im(images[j]), (48, 64, 3), False))
            data["next_image"].append(rectify_and_resize(bytes2im(images[j+1]), (48, 64, 3), False))

        actions = pos2actions(list(points[j:end+1]), list(rotations[j:end+1]))

        for l in range(len(actions)):

            data['actions'].append(np.copy(actions[l]))

    traj_index += 1

    print(traj_index, len(data['trajectory']))

data['colldmean'] = np.mean(distances)
data['colldstd'] = np.sqrt(np.var(distances))
data['collamean'] = np.mean(angles)
data['collastd'] = np.sqrt(np.var(angles))
        
print("Total len is", len(data["states"]))
os.makedirs(save_folder, exist_ok=True)

save_file = os.path.join(save_folder, 'traj_train' + '.pkl')
with open(save_file, 'wb') as f:
    pickle.dump(data, f)



data = dict(states=[],
            actions=[],
            next_states=[],
            trajectory=[],
            traj_index=[],
            iscoll=[],
            rewards=[],
            dones=[],
            image=[],
            next_image=[],
            spawngoal=[],
            colldmean=0.0,
            colldstd=0.0,
            collamean=0.0,
            collastd=0.0)

traj_index = 0

for i in tqdm.tqdm(range(len(names2))):
    print("Using file:", names2[i])
    curr_hdf5 = None
    try:
        curr_hdf5 = h5py.File(names2[i], 'r')
    except OSError:
        print("Coulndt open this hdf5")
        continue
    full_len = len(curr_hdf5['collision/any'])

    if full_len < 30:
        continue
    print("Using full lenght :", full_len)

    points = curr_hdf5['jackal/position']
    rotations = curr_hdf5['jackal/yaw']
    images = curr_hdf5['images/rgb_left']
    collisions = curr_hdf5['collision/physical']

    data['trajectory'].append([points[:full_len], rotations[:full_len]])

    for j in range(0, full_len-1):
        goal_step = np.random.randint(0, 30)
        end = min(j+goal_step, full_len-1)

        startpt = points[j]
        startrot = rotations[j]
        goalpt = points[end]
        goalyaw = rotations[end]

        actions = []

        if any([collisions[k] for k in range(j, min(j+collision_horizon, full_len))]):
            goal_angle, goal_dist = sample_goal_norm(angles, distances)
            r = -goal_dist - (100/(1-0.99))
            
            currpolar = np.array([np.sin(goal_angle), np.cos(goal_angle), goal_dist])
            nextpolar = np.array([np.sin(goal_angle), np.cos(goal_angle), goal_dist])
            
            data["states"].append(np.array(currpolar))
            data["next_states"].append(np.array(nextpolar))
            data["rewards"].append(r)
            data["dones"].append(1)
            data["traj_index"].append([traj_index, j])
            print(traj_index)
            data["iscoll"].append(True)
            data["image"].append(rectify_and_resize(bytes2im(images[j]), (48, 64, 3), False))
            data["next_image"].append(rectify_and_resize(bytes2im(images[j]), (48, 64, 3), False))

        else:
            r = reward(points[j], goalpt)
            currpolar = euc2polar(points[j], goalpt, rotations[j])
            nextpolar = euc2polar(points[j+1], goalpt, rotations[j+1])
            
            data["states"].append(np.array(currpolar))
            data["next_states"].append(np.array(nextpolar))
            data["rewards"].append(r)
            if goal_step < 2:
                dns = 1
            else:
                dns = 0
            data["dones"].append(dns)
            data["traj_index"].append([traj_index, j])
            data["iscoll"].append(False)
            data["image"].append(rectify_and_resize(bytes2im(images[j]), (48, 64, 3), False))
            data["next_image"].append(rectify_and_resize(bytes2im(images[j+1]), (48, 64, 3), False))

        actions = pos2actions(list(points[j:end+1]), list(rotations[j:end+1]))
        for l in range(len(actions)):
            data['actions'].append(np.copy(actions[l]))

    traj_index += 1

    print(traj_index, len(data['trajectory']))

data['colldmean'] = np.mean(distances)
data['colldstd'] = np.sqrt(np.var(distances))
data['collamean'] = np.mean(angles)
data['collastd'] = np.sqrt(np.var(angles))
        
print("Total len is", len(data["states"]))
os.makedirs(save_folder, exist_ok=True)

save_file = os.path.join(save_folder, 'traj_val' + '.pkl')
with open(save_file, 'wb') as f:
    pickle.dump(data, f)