# ReViND

This is the code submission for the corresponding CORL submission 'ReViND'.

## Code used for the various reward schemes ##

### Sunny Detector ###

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

## How Datasets are created for training ##

Pickled dataset (converted from native hdf5 files) of trajectories is loaded in (dataset of trajectories picked for specific reward scheme)

<details><summary> Loading pickled trajectories </summary>
<p>

```python
if isdir:
    print("Loading data from dir")
    with open(dir, 'rb') as f:
        trajectories = pickle.load(f)
    print("loaded data")

    else:
        trajectories = dir
```
</p>
</details>

Example of assignment of the rewards to trajectories (grassy example):

**Description:** (Fill in)

<details><summary> Assignment of rewards for grassy example </summary>
<p>

```python
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
```
</p>
</details>




