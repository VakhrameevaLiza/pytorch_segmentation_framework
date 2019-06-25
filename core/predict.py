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
        preds = torch.nn.Softmax(dim=1)(output)

        pbar_desc = "Predict: {}; Iter {:04d}/{:04d};".format(13*' ', idx + 1, niters)
        pbar.set_description(pbar_desc, refresh=False)

        if cuda:
            preds = preds.cpu().data.numpy()
        else:
            preds = preds.data.numpy()

        for i in range(len(name)):
            pred = preds[i][:, :, np.newaxis]
            pred = cv2.resize(pred, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(save_path, name[i]), pred)
