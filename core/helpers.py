import torch
import numpy as np

def set_lr(optimizer, lr_schedule, global_step):
    lr = lr_schedule.get_lr(global_step)
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


def get_grad_norm(model):
    sum_grad_norm = 0
    for p in model.parameters():
        grad = p.grad.data
        if torch.cuda.is_available():
            grad = grad.cpu().numpy()
        else:
            grad = grad.numpy()
        sum_grad_norm += np.linalg.norm(grad)
    return sum_grad_norm


def decoupled_weight_decay(model, lr, wd):
    for name, tensor in model.named_parameters():
        if 'bias' in name:
            continue
        tensor.data.add_(-wd * lr * tensor.data)


def get_mean_iou(num_cat, num_attr, metrics):
    if num_cat > 0 and num_attr > 0:
        iou = 0.5 * metrics['cat_iou'] + 0.5 * metrics['attr_iou']
    elif num_cat == 0:
        iou = metrics["attr_iou"]
    else:
        iou = metrics["cat_iou"]
    return iou

from utils.lr_schedules import ComplexSchedule, PolynomialSchedule
def get_lr_schedule(config, niters_per_epoch, num_epochs):
    if config.lr_schedule_type == 'polynomial':
        lr_schedule = PolynomialSchedule(config.lr, config.power, niters_per_epoch * num_epochs)
    elif config.lr_schedule_type == 'complex':
        lr_schedule = ComplexSchedule(niters_per_epoch * config.cycle_len_in_epochs,
                                      config.lr_min, config.lr_max, config.lr)
    else:
        raise NotImplementedError("Lr schedule type is unknown")
    return  lr_schedule


def process_metrics(metrics):
    metrics["attr_by_cl_iou"] = np.vstack(metrics["attr_by_cl_iou"]).mean(axis=0)
    metrics["attr_by_cl_dice"] = np.vstack(metrics["attr_by_cl_dice"]).mean(axis=0)
    metrics["loss"] = np.mean(metrics["loss"])

    metrics["cat_iou"] = np.mean(metrics["cat_iou"])
    metrics["cat_dice"] = np.mean(metrics["cat_dice"])
    metrics["attr_iou"] = np.mean(metrics["attr_iou"])
    metrics["attr_dice"] = np.mean(metrics["attr_dice"])

    metrics["iou"] = np.mean(metrics["iou"])


def write_logs(sw, mode, metrics, epoch):
    sw.add_scalar(mode + '_loss', metrics["loss"], epoch)
    sw.add_scalar(mode + '_iou', metrics["iou"], epoch)

    sw.add_scalar(mode + '_cat_iou', metrics["cat_iou"], epoch)
    sw.add_scalar(mode + '_attr_iou', metrics["attr_iou"], epoch)

    sw.add_scalar(mode + '_cat_dice', metrics["cat_dice"], epoch)
    sw.add_scalar(mode + '_attr_dice', metrics["attr_dice"], epoch)

    #
    # if num_attr <= 3:
    #     class_names = dataloader.dataset.get_class_names()
    #
    #     for cl, v in zip(class_names, train_metrics["attr_by_cl_iou"]):
    #         sw.add_scalar('train_iou_{}'.format(cl), v, epoch)
    #
    #     for cl, v in zip(class_names, train_metrics["attr_by_cl_dice"]):
    #         sw.add_scalar('train_dice_{}'.format(cl), v, epoch)


def create_state_dict(model, optimizer, epoch, global_step, val_metrics, config):
    state_dict = {}
    state_dict['model'] = model.state_dict()
    state_dict['optim'] = optimizer.state_dict()
    state_dict['epoch'] = epoch
    state_dict['global_step'] = global_step
    if val_metrics is not None:
        state_dict['val_cat_iou'] = val_metrics['cat_iou']
        state_dict['val_attr_iou'] = val_metrics['attr_iou']
    else:
        state_dict['val_cat_iou'] = -1
        state_dict['val_attr_iou'] = -1

    state_dict['config'] = config
    return state_dict
