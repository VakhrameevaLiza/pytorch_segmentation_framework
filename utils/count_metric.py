import torch
import numpy as np

def count_metric(input, target, metric_func, num_cat, num_attr, pass_probs=False):
    if len(target.shape) == 3:
        num_classes = input.shape[1]
        return metric_func(input.argmax(dim=1), target, num_classes)

    else:
        if num_cat > 0:
            if pass_probs:
                softmax = torch.nn.Softmax(dim=1)
                cat_metric = metric_func(softmax(input[:, :num_cat]), target[:,0], num_cat)[0]
            else:
                cat_metric = metric_func(input[:, :num_cat].argmax(dim=1), target[:, 0], num_cat)[0]
        else:
            cat_metric = 0

        metrics = []
        if num_attr > 0:
            for i in range(num_attr):
                bs, ch, h, w = target.shape
                #cur_target = torch.zeros((bs, h, w)).type_as(target)
                #cur_target[target[:, 1] == i + 1] = 1.
                if num_cat > 0:
                    cur_target = target[:, 1+i]
                else:
                    cur_target = target[:, i]

                sigmoid = torch.nn.Sigmoid()

                if pass_probs:
                    metric = metric_func(sigmoid(input[:, num_cat+i]), cur_target, 2)[0]
                else:
                    cl_probs = sigmoid(input[:, num_cat+i])[:, np.newaxis]
                    bg_probs = 1. - cl_probs
                    probs = torch.cat([bg_probs, cl_probs], dim=1)
                    metric = metric_func(probs.argmax(dim=1), cur_target, 2)[0]
                metrics.append(metric)
            attr_metric = np.mean(metrics)
        else:
            attr_metric = 0

        return cat_metric, attr_metric, metrics

