import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

def save_examples(num_cat, num_attr,
                  names_scores, model, get_by_ind_func,
                  color_cat_func, color_attr_func, save_path, config):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    for ind, score in names_scores:

        batch = get_by_ind_func(ind)
        data = batch['data']
        label = batch['label']
        name = batch['fn']

        cuda = torch.cuda.is_available()
        if cuda:
            data = batch['data'].cuda(non_blocking=True)
            label = batch['label'].cuda(non_blocking=True)

        output = model(data)
        if isinstance(output, tuple):
            output = output[-1]

        softmax = torch.nn.Softmax(dim=1)
        cat_probs = softmax(output[:, :num_cat])

        sigmoid = torch.nn.Sigmoid()
        attr_probs = sigmoid(output[:, num_cat:])

        # _, _, h, w = output.shape
        # cat_pred = np.zeros((h, w, 3))

        if cuda:
            img = data.cpu().numpy()
            label = label.cpu().numpy()
            cat_probs = cat_probs.data.cpu().numpy()
            attr_probs = attr_probs.data.cpu().numpy()
        else:
            img = data.numpy()
            label = label.numpy()
            cat_probs = cat_probs.data.numpy()
            attr_probs = attr_probs.data.numpy()

        img = img[0]
        label = label[0].astype('int64')
        cat_probs = cat_probs[0]
        attr_probs = attr_probs[0]

        img = img.transpose(1, 2, 0)
        if img.shape[2] == 1:
            img = img[:,:,0]
        img += config.mean
        img = (img * 255).astype('uint8')

        fontsize = 30
        plt.figure(figsize=(30, 30))

        if num_cat > 0 and num_attr > 0:
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.title("Source", fontsize=fontsize)

            plt.subplot(2, 2, 2)
            plt.imshow(color_cat_func(label[0]))
            plt.title("Ground Truth", fontsize=fontsize)

            plt.subplot(2, 2, 3)
            plt.imshow(color_cat_func(cat_probs.argmax(axis=0)))

            plt.subplot(2, 2, 4)
            _, h, w = attr_probs.shape
            attr_labels = np.zeros((num_attr, h, w), dtype='int64')

            labels_cnt = []
            for i in range(num_attr):
                pos_probs = attr_probs[i][np.newaxis]
                neg_probs = 1. - pos_probs
                probs = np.vstack((neg_probs, pos_probs))
                attr_labels[i] = probs.argmax(axis=0)

            plt.imshow(color_attr_func(attr_labels))

        elif num_cat > 0 and num_attr == 0:

            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title("Source", fontsize=fontsize)

            plt.subplot(1, 3, 2)
            plt.imshow(color_cat_func(label[0]))
            plt.title("Ground Truth", fontsize=fontsize)

            plt.subplot(1, 3, 3)
            plt.title("Prediction\niou = {:.2f}".format(score), fontsize=fontsize)
            plt.imshow(color_cat_func(cat_probs.argmax(axis=0)))

        elif num_cat == 0 and num_attr > 0:
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title("Source", fontsize=fontsize)

            plt.subplot(1, 3, 2)
            plt.imshow(color_attr_func(label))
            plt.title("Ground Truth", fontsize=fontsize)

            plt.subplot(1, 3, 3)
            plt.title("Prediction\niou = {:.2f}".format(score), fontsize=fontsize)

            _, h, w = attr_probs.shape
            attr_labels = np.zeros((num_attr, h, w), dtype='int64')
            for i in range(num_attr):
                pos_probs = attr_probs[i][np.newaxis]
                neg_probs = 1. - pos_probs
                probs = np.vstack((neg_probs, pos_probs))
                attr_labels[i] = probs.argmax(axis=0)

            plt.imshow(color_attr_func(attr_labels))

        plt.savefig(os.path.join(save_path, name))
        plt.close()
