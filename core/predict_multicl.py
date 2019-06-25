import torch
from tqdm import tqdm
import sys
import numpy as np
import cv2
import os
from utils.ensure_dir import ensure_dir_exists_and_empty


def predict(model, data_loader, save_path, out_h, out_w):
    ensure_dir_exists_and_empty(save_path)

    model.eval()
    niters = len(data_loader.dataset) // data_loader.batch_size

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(niters), file=sys.stdout, bar_format=bar_format)

    dataloader = iter(data_loader)
    cuda = torch.cuda.is_available()

    for idx in pbar:
        batch = dataloader.next()
        data = batch['data']
        name = batch['fn']
        if cuda:
            data = batch['data'].cuda(non_blocking=True)
            label = batch['label'].cuda(non_blocking=True)

        output = model(data)
        probs = torch.sigmoid(output)

        pbar_desc = "Predict: {}; Iter {:04d}/{:04d};".format(13*' ', idx + 1, niters)
        pbar.set_description(pbar_desc, refresh=False)

        if cuda:
            probs = probs.cpu().data.numpy()
        else:
            probs = probs.data.numpy()

        num_classes = probs.shape[1]

        for i in range(len(name)):

            predictions = []

            for j in range(num_classes):
                cl_probs = probs[i, j]
                bg_probs = 1 - cl_probs
                cl_predictions = np.where(cl_probs>bg_probs, np.ones_like(cl_probs), np.zeros_like(cl_probs))
                predictions.append(cl_predictions)

            np.save(os.path.join(save_path, name[i]), np.array(predictions))
