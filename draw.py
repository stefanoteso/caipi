#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mojito import load

def pad(array, length):
    assert array.ndim == 1
    full = np.zeros(length, dtype=array.dtype)
    full[:len(array)] = array
    return full

def prettify(ax):
#        xtick = 5 if max_iters <= 50 else 10
#        xticks = np.hstack([[1], np.arange(xtick, max_iters + 1, xtick)])
#        reg_ax.set_xticks(xticks)
#
#        ax.xaxis.label.set_fontsize(18)
#        ax.yaxis.label.set_fontsize(18)
#        ax.grid(True)
#        for line in ax.get_xgridlines() + ax.get_ygridlines():
#            line.set_linestyle('-.')
    pass

def draw(args):
    plt.style.use('ggplot')

    data = np.array([load(path) for path in args.pickles])
    # data indices: [file][fold][iteration][measure]

    num_files = data.shape[0]
    num_perfs = data.shape[-1]
    fig, axes = plt.subplots(1, num_perfs)

    # Draw all performance measures
    for p in range(num_perfs):
        perf_data = data[:,:,:,p]

        # TODO x axis should be the % of labelled examples

        for f in range(num_files):
            file_perf_data = perf_data[f,:,:]

            xs = np.arange(1, file_perf_data.shape[1] + 1)
            ys = np.mean(file_perf_data, axis=0)
            yerrs = np.std(file_perf_data, axis=0) / np.sqrt(file_perf_data.shape[0])

            axes[p].plot(xs, ys, linewidth=2, label='perf {}'.format(p))
            axes[p].fill_between(xs, ys - yerrs, ys + yerrs, linewidth=0,
                                 alpha=0.35)

            axes[p].set_title(args.title)
            axes[p].set_xlabel('% labels')
            axes[p].set_ylabel('performance')
            prettify(axes[p])

    fig.set_size_inches(12, 4)
    fig.savefig(args.png_basename + '.png', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('png_basename', type=str,
                        help='basename of the loss/time PNG plots')
    parser.add_argument('pickles', type=str, nargs='+',
                        help='comma-separated list of pickled results')
    parser.add_argument('-T', '--title', type=str, default='Title',
                        help='plot title')
    args = parser.parse_args()

    draw(args)
