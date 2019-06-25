import torch

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
