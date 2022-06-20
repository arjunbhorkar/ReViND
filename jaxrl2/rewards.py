import cv2
import numpy as np


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
    return img_out, mask, pred

def grass_detector(img):
    low_val = np.array([16, 50, 0])
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
    return img_out, mask, pred