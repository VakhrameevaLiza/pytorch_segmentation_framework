import torch
import torch.utils.data as data
import os
import json
import numpy as np
import pandas as pd
import cv2
from glob import glob
from torch.utils.data import DataLoader
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


class FashionDataset(data.Dataset):
    def __init__(self, dataset_path, data_type='train', train_pct=1.0,
                 valid_pct=None, w=512, h=512, mean=0,):

        self.dataset_path = dataset_path
        self.data_type = data_type

        if self.data_type == 'training' or self.data_type == 'validation':
            self.marking_df = pd.read_csv(os.path.join(self.dataset_path, "train.csv"))
        else:
            self.marking_df = None

        self.train_pct = train_pct
        if valid_pct is None:
            self.valid_pct = 1 - train_pct
        else:
            self.valid_pct = valid_pct

        self.w = w
        self.h = h
        self.mean = mean

        self.paths = self.get_img_paths()

        self.num_cat = 46+1
        self.num_attr = 0

        self.num_classes = self.num_cat + self.num_attr

        self.cat_colors = get_colors(self.num_cat, transparent_coef=0.5, white_bg=True)
        self.attr_colors = get_colors(self.num_attr+1, transparent_coef=0.5)

        self.ignore_index = -100

    def __len__(self):
        return len(self.paths)

    def get_img_paths(self):
        if self.data_type == 'training' or self.data_type == 'validation':

            imgs_names = os.listdir(os.path.join(self.dataset_path, 'train'))

            self.marking_df["fine_grained"] = self.marking_df["ClassId"].apply(lambda x: len(x.split("_"))) > 1

            imgs_paths = [os.path.join(self.dataset_path, "train", img_name) for img_name in imgs_names]
            num_imgs = len(imgs_paths)
            num_train_imgs = int(self.train_pct * num_imgs)

            np.random.seed(42)
            all_ind = np.arange(num_imgs)
            np.random.shuffle(all_ind)

            train_ind = all_ind[:num_train_imgs]
            valid_ind = all_ind[num_train_imgs:]

            if self.data_type == 'training':
                return [imgs_paths[i] for i in train_ind]
            else:
                return [imgs_paths[i] for i in valid_ind]
        else:
            imgs_names = os.listdir((os.path.join(self.dataset_path, "test")))
            imgs_paths = [os.path.join(self.dataset_path, "test", img_name) for img_name in imgs_names]
            return imgs_paths


    def get_gt(self, img_name):
        img_encode_df = self.marking_df[self.marking_df.ImageId == img_name]
        h = int(img_encode_df['Height'].values[0])
        w = int(img_encode_df['Width'].values[0])

        gt = np.zeros((h * w, 1 + self.num_attr))

        for idx in img_encode_df.index:
            ClassId = img_encode_df.ClassId[idx]

            cat = ClassId.split('_')[0]
            attrs = ClassId.split('_')[1:]

            cat = int(cat)
            attrs = [int(attr) for attr in attrs]

            starts = img_encode_df['EncodedPixels'][idx].split(' ')[::2]
            starts = [int(start) for start in starts]

            lengths = img_encode_df['EncodedPixels'][idx].split(' ')[1::2]
            lengths = [int(length) for length in lengths]

            for start, length in zip(starts, lengths):

                gt[start-1: start-1 + length, 0] = 1 + cat

                # for attr in attrs:
                #     #attr=1
                #     gt[start-1: start-1 + length, 1 + attr] = 1

        gt = gt.reshape((w, h, 1 + self.num_attr)).transpose((1,0,2))
        return gt

    def __getitem__(self, item):
        img_path = self.paths[item]
        img_name = img_path.split('/')[-1]

        img = cv2.imread(img_path)[:, :, ::-1]

        if self.data_type == 'training' or self.data_type == 'validation':
            gt = self.get_gt(img_name)
        else:
            gt = None

        img = self.preprocess_img(img, self.mean, self.h, self.w)
        if gt is not None:
            lbl = self.preprocess_lbl(gt, self.ignore_index, self.h, self.w)
        else:
            lbl = None

        img = torch.from_numpy(np.ascontiguousarray(img)).float()
        lbl = torch.from_numpy(np.ascontiguousarray(lbl)).long()

        sample = dict(
            fn=img_name,
            data=img,
            label=lbl,
        )
        return sample

    def get_by_ind(self, ind):
        output_dict = self.__getitem__(ind)
        output_dict['data'] = output_dict['data'].unsqueeze(0)
        output_dict['label'] = output_dict['label'].unsqueeze(0)
        return output_dict

    @staticmethod
    def preprocess_img(img, mean, new_h, new_w):
        h,w,_ = img.shape
        img = cv2.resize(img, (new_w, new_h))
        img = np.array(img).astype('float32') / 255.0
        img -= mean
        img = img.transpose((2, 0, 1))
        return img

    @staticmethod
    def preprocess_lbl(lbl, ignore_index, new_h, new_w):
        h,w,_ = lbl.shape
        preproc_lbl = cv2.resize(lbl, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if len(preproc_lbl.shape) == 2:
            preproc_lbl = preproc_lbl[:,:,np.newaxis]
        preproc_lbl = preproc_lbl.astype('int')
        preproc_lbl = preproc_lbl.transpose(2, 0, 1)
        return preproc_lbl

    def cat_color(self, lbl):
        color_lbl = self.cat_colors[lbl]
        return color_lbl

    def attr_color(self, attrs):
        h, w = attrs.shape[1], attrs.shape[2]
        color_attrs = np.zeros((h,w,4))

        for i in range(self.num_attr):
            attr = attrs[i]
            color_attrs += self.attr_colors[(i+1)*attr]
        color_attrs[:, :, 3] /= self.num_attr
        color_attrs[:,:,:3]=np.minimum(1., color_attrs[:,:,:3])
        return color_attrs


if __name__ == '__main__':
    train_dataset = FashionDataset('/home/vakhrameevaliza/fashion2019', data_type='train',
                                   h=512, w=512)
    train_loader = DataLoader(train_dataset, batch_size=1)

    for batch in train_loader:
        img = batch['data'].numpy()[0]
        label = batch['label'].numpy()[0]

        if label[1:].sum() > 0:
            color_label = train_dataset.color_label(label[0])
            color_attrs = train_dataset.color_attr(label[1:])

            plt.figure(figsize=(30,10))
            plt.subplot(1,3,1)
            plt.imshow(img[:,:,::-1])
            plt.subplot(1,3,2)
            plt.imshow(color_label)
            plt.subplot(1,3,3)
            plt.imshow(color_attrs)
            plt.title(batch['fn'][0], fontsize=14)
            plt.show()
            plt.close()
