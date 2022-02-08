import argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from time import time
from typing import Callable, Any
from skimage import morphology

from models import ConvNet, ResNet, UNet


def measure_inference_time_cpu(model: nn.Module,
                               filter_func: Callable[[Any], np.array],
                               in_channels: int,
                               img_size: int,
                               batch_size: int = 1,
                               num_iter: int = 1000,
                               warm_up: int = 30) -> (float, float):

    """
    Function for measuring inference time on CPU of classic algorithm and its neural approximation.

    :param model: model of neural network
    :param filter_func: lambda function for classic algorithm
    :param in_channels: number of input channels of the model
    :param img_size: size of image
    :param batch_size: size of batch
    :param num_iter: number of iterations for performance measuring
    :param warm_up: number of iterations for performance measuring ising for warm-up
    :return: mean time of inference of the model (ms), mean time of inference of the algorithm (ms)
    """

    time_nn = []
    time_algo = []

    dummy_input_nn = torch.randn(batch_size, in_channels, img_size, img_size, dtype=torch.float).cpu()
    dummy_input_algo = np.random.randn(img_size, img_size)

    model = model.cpu()
    model.eval()
    with torch.no_grad():
        for i in range(warm_up+num_iter):
            # neural network
            start = time()
            _ = model(dummy_input_nn)
            time_nn.append(time() - start)

        for i in range(warm_up + num_iter):
            # algorithm
            start = time()
            _ = filter_func(dummy_input_algo)
            time_algo.append(time() - start)

    return np.mean(time_nn[warm_up:])*1000, np.mean(time_algo[warm_up:])*1000


def measure_inference_time_gpu(model: nn.Module,
                               in_channels: int,
                               img_size: int,
                               batch_size: int = 1,
                               num_iter: int = 1000,
                               warm_up: int = 30) -> float:
    """
    Function for measuring inference time on GPU of the neural network.

    :param model: model of neural network
    :param in_channels: number of input channels of the model
    :param img_size: size of image
    :param batch_size: size of batch
    :param num_iter: number of iterations for performance measuring
    :param warm_up: number of iterations for performance measuring ising for warm-up
    :return: mean time of inference of the model (ms)
    """

    time_nn = []

    dummy_input = torch.randn(batch_size, in_channels, img_size, img_size, dtype=torch.float).cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    model = model.cuda()
    model.eval()
    # GPU-WARM-UP
    for _ in range(warm_up):
        _ = model(dummy_input)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for i in range(num_iter):
            starter.record()
            _ = model(dummy_input)
            ender.record()

            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # ms
            time_nn.append(curr_time)
    return np.mean(time_nn)


def compare(model: nn.Module,
            in_channels: int,
            img_size: int,
            forms: list,
            sizes: list,
            num_iter: int,
            warm_up: int) -> pd.DataFrame:

    """
    Function for comparing inference time of classic algorithm and its neural approximation

    :param model: model of neural network
    :param in_channels: number of input channels of the model
    :param img_size: size of image
    :param forms: forms of dilation kernel
    :param sizes: sizes of dilation kernel
    :param num_iter: number of iterations for performance measuring
    :param warm_up: number of iterations for performance measuring using for warm-up
    :return: comparison table
    """
    torch.set_num_threads(args.num_threads)
    print('Number of threads: %i\n' %torch.get_num_threads())

    idx = pd.IndexSlice
    mux_cols = pd.MultiIndex.from_product([['CPU', 'GPU'], ['Network', 'Algorithm']])
    mux_rows = pd.MultiIndex.from_product([forms, sizes])
    comp_table = pd.DataFrame(columns=mux_cols, index=mux_rows)

    for form in forms:
        for size in sizes:
            if form == 'disk':
                filt = lambda x: morphology.dilation(x, selem=morphology.disk(radius=size))
            elif form == 'square':
                filt = lambda x: morphology.dilation(x, selem=morphology.square(width=size))

            print('Processing', form, size)
            # cpu
            time_nn, time_algo = measure_inference_time_cpu(model=model,
                                                            filter_func=filt,
                                                            in_channels=in_channels,
                                                            img_size=img_size,
                                                            batch_size=1,
                                                            num_iter=num_iter,
                                                            warm_up=warm_up)

            comp_table.loc[idx[form, size], idx['CPU', 'Network']] = "{:.4f}".format(time_nn)
            comp_table.loc[idx[form, size], idx['CPU', 'Algorithm']] = "{:.4f}".format(time_algo)

            # gpu
            time_nn = measure_inference_time_gpu(model=model,
                                                 in_channels=in_channels,
                                                 img_size=img_size,
                                                 batch_size=1,
                                                 num_iter=num_iter,
                                                 warm_up=warm_up)
            
            comp_table.loc[idx[form, size], idx['GPU', 'Network']] = "{:.4f}".format(time_nn)

    return comp_table


def parse():
    parser = argparse.ArgumentParser(description='Measure inference time')
    parser.add_argument("model", type=str, help='Type of model')
    parser.add_argument("in_channels", type=int, help='Number of input channels')
    parser.add_argument("mid_channels", type=int, help='Number of middle channels')
    parser.add_argument("num_threads", type=int, help='Number of threads')
    parser.add_argument("--img_size", type=int, default=128, help='Size of image')
    parser.add_argument("--num_iter", type=int, default=100, help='Number of iterations')
    parser.add_argument("--warm_up", type=int, default=30, help='Number of iterations for warm-uo')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    if args.model == 'ConvNet':
        model = ConvNet(filter_name='dilation disk',
                        in_channels=args.in_channels,
                        out_channels=[args.mid_channels] * 12 + [1],
                        kernel_size=[3] * 13,
                        stride=[1] * 13)

    elif args.model == 'ResNet':
        model = ResNet(filter_name='dilation disk',
                       in_channels=args.in_channels,
                       mid_channels=args.mid_channels)

    elif args.model == 'UNet':
        model = UNet(filter_name='dilation disk',
                     in_channels=1,
                     mid_channels=4)

    comp_table = compare(model=model,
                         in_channels=args.in_channels,
                         img_size=args.img_size,
                         forms=['disk', 'square'],
                         sizes=[3, 5, 7, 10, 12, 15, 17, 20],
                         num_iter=args.num_iter,
                         warm_up=args.warm_up)

    print('\nFor %s:' % args.model)
    print(comp_table)