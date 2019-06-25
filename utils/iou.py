import torch
import numpy as np


def iou_func(predict, target, num_classes, ignore_bg=False):

    batch_size = predict.shape[0]

    inter_arr = np.zeros((batch_size, num_classes))
    union_arr = np.zeros((batch_size, num_classes))

    predict = predict.view((batch_size, -1))
    target = target.view((batch_size, -1))

    for i in range(num_classes):
        inter = ((predict == i) & (target == i)).sum(dim=1)
        union = (predict == i).sum(dim=1) + (target == i).sum(dim=1) - inter

        inter = inter.float()
        union = union.float()

        if torch.cuda.is_available():
            inter_arr[:, i] = inter.data.cpu().numpy()
            union_arr[:, i] = union.data.cpu().numpy()
        else:
            inter_arr[:, i] = inter.data.numpy()
            union_arr[:, i] = union.data.numpy()

    with np.errstate(divide='ignore', invalid='ignore'):
        ious = inter_arr / union_arr

        if ignore_bg:
            print("\nIGNORE BG!!!!!!!")
            inter_arr = inter_arr[:, 1:]
            union_arr = union_arr[:, 1:]
            ious = ious[:, 1:]

        mask = 1.0 * (union_arr > 0)
        by_classes = np.nansum(ious, axis=0) / np.sum(mask, axis=0)
        mean = np.nansum(ious) / np.sum(mask)
        return mean, by_classes


# def count_iou(input, target, num_cat=None, num_attr=None, num_cat_with_bg=False):
#     if len(target.shape) == 3:
#         num_classes = input.shape[1]
#         return iou_func(input.argmax(dim=1), target, num_classes)
#     elif num_cat is not None:
#             if not num_cat_with_bg:
#                 num_cat += 1
#             cat_iou = iou_func(input[:num_cat].argmax(dim=1), target[:,0], num_cat)
#             ious = []
#             for i in range(num_attr):
#                 cl_probs = torch.sigmoid(input[:, num_cat+i:num_cat+(i+1)])
#                 bg_probs = 1. - cl_probs
#                 probs = torch.cat([bg_probs, cl_probs], dim=1)
#                 iou = iou_func(probs.argmax(dim=1), target[:, num_cat+i], 2)[0]
#                 ious.append(iou)
#             attr_iou = np.mean(ious)
#             return cat_iou, attr_iou
#     else:
#         num_classes = target.shape[1]
#         ious = []
#         for i in range(num_classes):
#             cl_probs = torch.sigmoid(input[:,i:(i+1)])
#             bg_probs = 1. - cl_probs
#             probs = torch.cat([bg_probs, cl_probs], dim=1)
#             iou = iou_func(probs.argmax(dim=1), target[:, i], 2)[0]
#             ious.append(iou)
#         return np.mean(ious), ious


if __name__ == "__main__":
    predict = torch.from_numpy(np.array([[[0, 1, 2], [0, 1, 2]] ]))
    target =  torch.from_numpy(np.array([[[0, 1, 3], [1, 1, 0]] ]))
