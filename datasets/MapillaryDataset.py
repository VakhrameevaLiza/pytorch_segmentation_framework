import torch
import torch.utils.data as data
import os
import json
import numpy as np
import cv2


class MapillaryDataset(data.Dataset):

    def __init__(self, dataset_path, data_type, h=1024, w=2048, mean=np.zeros(3), crop=True, seed=42):
        super(MapillaryDataset, self).__init__()

        np.random.seed(seed)

        self.data_type = data_type
        self.dataset_path = dataset_path

        self.h = h
        self.w = w
        self.mean = mean

        self.crop = crop

        if data_type == 'testing':
            self.paths = self.get_img_paths()
        else:
            self.paths = self.get_img_lbl_paths()

        t = self.get_label_color_maps()
        self.colors = t[0]
        self.class_names = t[1]
        self.color_label_map = t[2]
        self.ignore_index = t[3]
        self.num_classes = t[4]

    def __len__(self):
        return len(self.paths)

    # def len(self):
    #     return len(self.paths)

    def __getitem__(self, index):
        img_path, lbl_path = self.paths[index]
        item_name = img_path.split('/')[-1][:-4]

        img, lbl = self.fetch_data(img_path, lbl_path)

        if self.crop:

            if self.data_type == 'training':

                if img.shape[0] > self.h:
                    h1 = np.random.randint(0, img.shape[0] - self.h)
                else:
                    h1 = 0

                if img.shape[1] > self.w:
                    w1 = np.random.randint(0, img.shape[1] - self.w)
                else:
                    w1 = 0
            else:
                h1 = max(0, img.shape[0] // 2 - self.h // 2)
                w1 = max(0, img.shape[1] // 2 - self.w // 2)
        else:
            h1 = None
            w1 = None

        img = self.preprocess_img(img, self.mean, self.h, self.w, h1=h1, w1=w1)
        lbl = self.preprocess_lbl(lbl, self.ignore_index, self.h, self.w, h1=h1, w1=w1)

        img = torch.from_numpy(np.ascontiguousarray(img)).float()
        lbl = torch.from_numpy(np.ascontiguousarray(lbl)).long()

        output_dict = dict(data=img, label=lbl, fn=str(item_name))

        return output_dict

    def get_by_ind(self, ind):
        output_dict = self.__getitem__(ind)
        output_dict['data'] = output_dict['data'].unsqueeze(0)
        output_dict['label'] = output_dict['label'].unsqueeze(0)
        return output_dict

    @staticmethod
    def fetch_data(img_path, lbl_path):
        img = cv2.imread(img_path)
        if lbl_path:
            lbl = cv2.imread(lbl_path)
        else:
            lbl = None
        return img, lbl

    @staticmethod
    def preprocess_img(img, mean, h, w, h1=None, w1=None):

        if h1 and w1:
            img = img[h1:min(h1+h, img.shape[0]), w1:min(w1+w, img.shape[1])]
        #else:
        img = cv2.resize(img, (w, h))

        img = np.array(img).astype('float32') / 255
        img = img[:, :, ::-1]
        img -= mean
        img = img.transpose(2, 0, 1)
        return img

    # @staticmethod
    # def reverse_preprocess_img(img, mean):
    #     img = img.transpose(1, 2, 0)
    #
    #     img += mean
    #     img = (img * 255).astype('uint8')
    #     return img

    @staticmethod
    def preprocess_lbl(lbl, ignore_index, h, w, h1=None, w1=None):
        lbl = lbl[:, :, 0]
        if h1 and w1:
            lbl = lbl[h1:min(h1+h, lbl.shape[0]), w1:min(w1+w, lbl.shape[1])]
        #else:
        preproc_lbl = cv2.resize(lbl, (w, h), interpolation=cv2.INTER_NEAREST)

        preproc_lbl = preproc_lbl.astype('int')
        preproc_lbl[preproc_lbl == ignore_index] = -100
        return preproc_lbl

    def color_label(self, lbl):
        lbl[lbl == -100] = self.ignore_index
        color_lbl = self.colors[lbl]
        lbl[lbl == self.ignore_index] = -100
        return color_lbl


    def get_img_lbl_paths(self):
        imgs_dir = os.path.join(self.dataset_path, self.data_type, 'images')
        lbls_dir = os.path.join(self.dataset_path, self.data_type, 'instances')

        imgs_names = sorted(os.listdir(imgs_dir), key=lambda s: s[:-4])
        lbls_names = sorted(os.listdir(lbls_dir), key=lambda s: s[:-4])

        assert len(imgs_names) == len(lbls_names)

        file_names = []

        for img_name, lbl_name in zip(imgs_names, lbls_names):
            assert img_name[:-4] == lbl_name[:-4]
            file_names.append((os.path.join(imgs_dir, img_name),
                               os.path.join(lbls_dir, lbl_name)))

        return file_names

    def get_img_paths(self):
        imgs_dir = os.path.join(self.dataset_path, self.data_type, 'images')
        imgs_names = sorted(os.listdir(imgs_dir), key=lambda s: s[:-4])

        file_names = []
        for img_name in imgs_names:
            file_names.append((os.path.join(imgs_dir, img_name),
                               None))
        return file_names


    def get_label_color_maps(self):
        dataset_config_path = os.path.join(self.dataset_path, 'config.json')
        with open(dataset_config_path) as f:
            cfg = json.load(f)

        labels = cfg['labels']

        colors = []
        class_names = []
        color_label_map = {}

        for i, label in enumerate(labels):
            color = label['color']
            name = label['readable']

            colors.append(color)
            class_names.append(name)

            color_label_map[tuple(color)] = i

        num_classes = len(labels)
        ignore_index = num_classes - 1
        colors = np.array(colors)

        return colors, class_names, color_label_map, ignore_index, num_classes - 1


import matplotlib.pyplot as plt
if __name__ == "__main__":
    dataset = MapillaryDataset('/home/liza/PycharmProjects/data', 'training', h=512, w=512)
    out = dataset.__getitem__(2)
    img = out['data']
    lbl = out['label']
    fn = out['fn']

    img = dataset.reverse_preprocess_img(img.numpy(), np.zeros(3))

    plt.imshow(img)
    plt.show()
    plt.imshow(lbl)
    plt.show()