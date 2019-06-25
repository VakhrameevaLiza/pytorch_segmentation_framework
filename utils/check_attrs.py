import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch import nn
import matplotlib.pyplot as plt


def get_colors(num_colors, transparent_coef=1., white_bg=False):
    cmap = plt.cm.get_cmap("hsv", num_colors)
    colors = np.array([cmap(i) for i in range(num_colors)])
    if white_bg:
        colors[0] = np.array([1.,1.,1.,1.])
    else:
        colors[0] = np.array([0.,0.,0.,1.])
    colors[:,3] = transparent_coef
    return colors



dataset_path = '/home/vakhrameevaliza/fashion2019'
marking_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))

imgs_names = os.listdir(os.path.join(dataset_path, 'train'))

for img_name in imgs_names:
    img_encode_df = marking_df[marking_df.ImageId == img_name]

    h = int(img_encode_df['Height'].values[0])
    w = int(img_encode_df['Width'].values[0])

    gt = np.zeros((h * w, 2))

    for idx in img_encode_df.index:

        ClassId = img_encode_df.ClassId[idx]

        cat = ClassId.split('_')[0]
        attrs = ClassId.split('_')[1:]

        cat = int(cat)
        attrs = np.array([int(attr) for attr in attrs], dtype='int')

        starts = img_encode_df['EncodedPixels'][idx].split(' ')[::2]
        starts = [int(start) for start in starts]

        lengths = img_encode_df['EncodedPixels'][idx].split(' ')[1::2]
        lengths = [int(length) for length in lengths]

        for start, length in zip(starts, lengths):

            gt[start - 1: start - 1 + length, 0] = 1 + cat

            for attr in attrs:
                gt[start - 1: start - 1 + length, 1] = 1 + attr

    gt = gt.reshape((w, h, 2)).transpose((1, 0, 2))
    gt = gt[:, :, 1:]

    uniq0, cnt0 = np.unique(gt, return_counts=True)

    resized_gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST)
    resized_gt = resized_gt.astype('int')
    uniq1,cnt1 = np.unique(resized_gt, return_counts=True)

    max_pool_resized = cv2.resize(gt, (2048, 2048), interpolation=cv2.INTER_NEAREST)[:,:, np.newaxis]
    max_pool_resized = max_pool_resized.transpose((2,0,1))[np.newaxis]
    t = torch.from_numpy(max_pool_resized)
    op = nn.MaxPool2d(kernel_size=2)
    max_pool_resized = op(op(t)).data.numpy()
    max_pool_resized = max_pool_resized.astype('int')
    uniq2, cnt2 = np.unique(max_pool_resized, return_counts=True)

    if len(uniq0) != len(uniq1) or len(uniq0) != len(uniq2):
        print(uniq0, cnt0)
        print(uniq1, cnt1)
        print(uniq2, cnt2)
        print('\n')
    elif len(uniq0)>1:
        print('Allright')
        # print(np.unique(attrs, return_counts=True))
        # col = get_colors(92, transparent_coef=1., white_bg=False)
        # col_lbl = col[gt.astype('int')][:,:,0,:]
        # print(col_lbl.shape)
        # plt.imshow(col_lbl)
        # plt.show()


