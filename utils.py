import numpy as np
import time
import torch
from ptflops import get_model_complexity_info


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr
    # return lr


def fast_hist(a, b, n):
    '''
    a and b are label and prediction respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

#FLOPS and params

def get_model_stats(model, input_shape=(3, 512, 1024), device='cuda'):
    model.eval().to(device)
    with torch.cuda.amp.autocast(enabled=False):  # Disable AMP for analysis
        macs, params = get_model_complexity_info(
            model,
            input_shape,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False
        )
    print("\n - FLOPs: {macs}\n - Params: {params}")
    return macs, params


def measure_latency(model, input_shape=(3, 512, 1024), device='cuda', warmup=2, runs=10):
    model.eval().to(device)
    dummy_input = torch.randn(1, *input_shape).to(device)

    # Warm-up
    for _ in range(warmup):
        _ = model(dummy_input)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    end = time.time()

    avg_latency = (end - start) / runs
    print(f"Avg Latency: {avg_latency * 1000:.2f} ms per image")
    return avg_latency