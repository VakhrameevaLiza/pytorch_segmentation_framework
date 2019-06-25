import time
import torch


def compute_speed(model, input_size, iteration=500):
    torch.backends.cudnn.benchmark = True

    model.train()
    model = model.cuda()

    input = torch.randn(*input_size).cuda()

    for _ in range(10):
        model(input)

    torch.cuda.synchronize()
    torch.cuda.synchronize()

    t_start = time.time()
    for _ in range(iteration):
        model(input)

    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_in_ms = elapsed_time / iteration * 1000

    return speed_in_ms
