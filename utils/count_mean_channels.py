import cv2
import os
import numpy as np
from tqdm import tqdm

def count_mean_channels(dir):
    channels_values = np.zeros(3)
    img_names = os.listdir(dir)
    cnt = 0
    for i, img_name in tqdm(enumerate(img_names)):
        if i %  10 != 0:
            continue
        img = cv2.imread(os.path.join(dir, img_name))
        channels_values += img.mean(axis=0).mean(axis=0)
        cnt += 1
    channels_values /= cnt
    channels_values /= 255
    return channels_values


if __name__ == '__main__':
    print(count_mean_channels('/data/mapillary/training/images'))
