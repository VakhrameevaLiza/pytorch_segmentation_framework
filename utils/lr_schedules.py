import numpy as np


class PolynomialSchedule(object):
    def __init__(self, start_lr, power, total_iters):
        self.start_lr = start_lr
        self.power = power
        self.total_iters = total_iters

    def get_lr(self, cur_iter):
        return self.start_lr * ((1 - float(cur_iter) / self.total_iters) ** self.power)


class ComplexSchedule(object):

    def __init__(self, cycle_len, lr_min, lr_max, const_lr):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.const_lr = const_lr
        self.val_stats = []
        self.cycle_stats = []
        self.strategy = 'cycle'
        self.cycle_len = cycle_len
        self.global_step = 0

    def update_val_stats(self, v):

        self.val_stats.append(v)
        self.update_strategy()

    def update_strategy(self):
        num_epochs = len(self.val_stats)

        if self.strategy == 'const' and num_epochs >= 2:

            rel_imprv_pct = 100 * (self.val_stats[-1] - self.val_stats[-2]) / self.val_stats[-2]
            if rel_imprv_pct < 1:
                self.const_lr *= 0.9

        elif self.strategy == 'cycle':
            if (self.global_step + 1) % self.cycle_len == 0:
                self.cycle_stats.append(self.val_stats[-1])
                if len(self.cycle_stats) >= 2:
                    rel_imprv_pct = 100 * (self.cycle_stats[-1] - self.cycle_stats[-2]) / self.cycle_stats[-2]
                    if rel_imprv_pct < 5:
                        self.strategy = 'const'

    def get_cycle_lr(self, global_step):

        t = global_step / self.cycle_len - global_step // self.cycle_len
        lr = self.lr_min + np.arccos(np.cos(2.0 * np.pi * t)) / np.pi * (self.lr_max - self.lr_min)
        return lr

    def get_lr(self, global_step):
        self.global_step = global_step

        if self.strategy == 'cycle':
            return self.get_cycle_lr(global_step)
        else:
            return self.const_lr

