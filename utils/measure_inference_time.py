import torch
import time
import numpy as np

from models.Unet import Unet


def measure_inference_time(model, input_size):
    torch.backends.cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size).cuda()

    # warming
    for _ in range(10):
        out = model(input)

    num_iters = 10

    ts = []
    torch.cuda.synchronize()
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        out = model(input)
        end.record()

        torch.cuda.synchronize()

        t = start.elapsed_time(end)
        ts.append(t)

    print('{:.1f} ms'.format(np.mean(ts)))


if __name__ == '__main__':
    net = Unet(65)
    input_size = (1, 3, 1024, 2048)

    measure_inference(net, input_size)