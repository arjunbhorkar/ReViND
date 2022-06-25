# ReViND (Submission for CORL 2022)

This is the code submission for the corresponding CoRL 2022 submission 'ReViND'.

## Installation

Run
```bash
# Install all the requirements from requirements.txt

# To process raw RECON data into required pkl format - put raw recon data in the relative folder ../recon_release
python hdf2pkl.py

# To train the model - edit new_dir to the directory of your stored pkl file
python train_offline_recon.py
```

## Code used for the various reward schemes ##

(The following code can be found in `dataset_utils.py` in the jaxrl2 folder)

<details><summary> Sunny Detector </summary>
<p>

**Description:** Assigns a float representing how "sunny" an image is based on the bottom middle third of the image. To allow for this, we first convert the image from RGB to HSV. We then check which pixels are in a certain _value_ range.

```python
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
```

</p>
</details>


<details><summary> Grassy Detector </summary>
<p>

**Description:** Assigns a float representing how "grassy" an image is based on the bottom middle third of the image. To allow for this, we first convert the image from RGB to HSV. We then check which pixels are in a certain _hue_ range.

```python
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
```
</p>
</details>

## Processing the dataset for training ##

(The following code can be found in `dataset_utils.py` in the jaxrl2 folder)


Our method for sampling goals from the generated pkl and labelling the rewards

<details><summary> Goal sampling and reward labelling </summary>
<p>

```python
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
```
</p>
</details>





