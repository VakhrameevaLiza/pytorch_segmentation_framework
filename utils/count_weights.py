import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


def count_weigths(dataset, save_path):
    dataloader = DataLoader(dataset, batch_size=1,
                             num_workers=4, shuffle=False, pin_memory=True)

    num_cat = dataset.num_cat
    num_attr = dataset.num_attr

    cat_pct = np.zeros(num_cat)
    attr_pct = np.zeros(num_attr)

    cat_presented = np.zeros(num_cat)
    attr_presented = np.zeros(num_attr)

    cnt = 0
    for i, batch in tqdm(enumerate(iter(dataloader))):
        label = batch['label'][0]
        size = label.shape[1] * label.shape[2]

        cat_label = label[0]

        unique_cats, unique_cats_cnt = np.unique(cat_label, return_counts=True)
        cat_pct[unique_cats] += unique_cats_cnt/size
        cat_presented[unique_cats] = 1.

        attr_label = label[1]
        unique_attrs, unique_attrs_cnt = np.unique(attr_label, return_counts=True)
        unique_attrs = unique_attrs[1:]
        unique_attrs_cnt = unique_attrs_cnt[1:]

        attr_pct[unique_attrs-1] += unique_attrs_cnt/size
        attr_presented[unique_attrs-1] = 1.

        cnt += 1

    cat_total_pct =  cat_pct / cnt
    assert np.allclose(cat_total_pct.sum(),  1.)
    median = sorted(cat_total_pct)[num_cat//2]
    cat_weights = median / cat_total_pct

    attr_total_pct = attr_pct / cnt
    attr_total_pct = attr_total_pct[:, np.newaxis] # pct-0.1, w=0.1/0.1=1.
    bg_total_pct = 1. - attr_total_pct # pct-0.9, w =0.1/0.9=1/9

    attr_weights = np.hstack((bg_total_pct/bg_total_pct,
                              bg_total_pct/attr_total_pct))

    np.save(os.path.join(save_path, 'cat_weights_2'), cat_weights)
    np.save(os.path.join(save_path, 'attr_weights_2'), attr_weights)

    np.save(os.path.join(save_path, 'cat_presented'), cat_presented)
    np.save(os.path.join(save_path, 'attr_presented'), attr_presented)



