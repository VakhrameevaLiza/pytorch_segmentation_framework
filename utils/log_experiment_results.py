import time
import datetime
import os


def log_results(d, filename):
    sep = '\t'
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            pass

    with open(filename, "a") as f:
        currentDT = datetime.datetime.now()
        s = str(currentDT)+sep
        for k in ['name', 'model', 'loss', 'cat_weights', 'attr_weights',
                  'lr_schedule', 'num_epochs', 'train_time_min']:
            s += str(d[k]) + sep

        for mode in ['train', 'valid']:
            for metric in ['iou', 'loss', 'cat_iou', 'attr_iou', 'cat_dice', 'attr_dice']:
                k = mode + '_' + metric
                s += str(d[k]) + sep

        s += '\n'
        f.write(s)
