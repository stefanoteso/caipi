#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mojito import load


PERF_TO_TITLE = ['pr', 'rc', 'F1', 'expl. rc', '# fixed predictions']


class Tango:
    # From light to dark
    YELLOW  = ("#fce94f", "#edd400", "#c4a000")
    ORANGE  = ("#fcaf3e", "#f57900", "#ce5c00")
    BROWN   = ("#e9b96e", "#c17d11", "#8f5902")
    GREEN   = ("#8ae234", "#73d216", "#4e9a06")
    BLUE    = ("#729fcf", "#3465a4", "#204a87")
    VIOLET  = ("#ad7fa8", "#75507b", "#5c3566")
    RED     = ("#ef2929", "#cc0000", "#a40000")
    WHITE   = ("#eeeeec", "#d3d7cf", "#babdb6")
    BLACK   = ("#888a85", "#555753", "#2e3436")


def get_style(args):
    color_type = {
        ('svm', 'random'): Tango.BLUE,
        ('svm', 'least-confident'): Tango.RED,
        ('svm', 'least-margin'): Tango.ORANGE,
        ('gp', 'random'): Tango.VIOLET,
    }[(args.learner, args.strategy)]
    color = color_type[0 if args.improve_explanations else -1]

    linestyle = {
        False: '-',
        True: '--',
    }[args.improve_explanations]

    label = args.strategy
    if args.improve_explanations:
        label += ' EI'

    return label, color


def draw(args):
    plt.style.use('ggplot')

    trace_args, traces = [], []
    for path in args.pickles:
        data = load(path)
        trace_args.append(data['args'])
        traces.append(data['traces'])
        num_examples = data['num_examples']

    traces = np.array(traces)
    # traces indices: [file, fold, iteration, measure]

    num_files = traces.shape[0]
    num_perfs = traces.shape[-1]
    fig, axes = plt.subplots(1, num_perfs)

    FACTOR = 100 / num_examples

    # For each performance measure
    for p in range(num_perfs):
        traces_p = traces[:,:,:,p]

        # For each results file
        x0, ymin, ymax = None, np.inf, -np.inf
        for f in range(num_files):
            traces_f_p = traces_p[f,:,:]

            label, color = get_style(trace_args[f])

            x0f = round(trace_args[f].perc_known)
            if x0 is None:
                x0 = x0f
            # assert x0 == x0f, 'perc_known mismatch'

            xs = x0 + np.arange(0, traces_f_p.shape[1]) * FACTOR
            ys = np.mean(traces_f_p, axis=0)
            yerrs = np.std(traces_f_p, axis=0) / np.sqrt(traces_f_p.shape[0])

            axes[p].plot(xs, ys, linewidth=2, label=label, color=color)
            axes[p].fill_between(xs, ys - yerrs, ys + yerrs, linewidth=0,
                                 alpha=0.35, color=color)

            # Draw a vertical line where explanations start
            xlinef = x0 + (trace_args[f].start_explaining_at + 1) * FACTOR
            axes[p].axvline(x=xlinef, ymin=0, ymax=1, linewidth=2,
                            linestyle=':', color=Tango.BLACK[1])

            ymin = min(ys.min(), ymin)
            ymax = max(ys.max(), ymax)

        axes[p].set_title(PERF_TO_TITLE[p])
        axes[p].set_xlim([xs[0], xs[-1]])
        axes[p].set_xlabel('% labels')
        axes[p].set_ylim([max(0, ymin * 0.9), ymax * 1.1])

    legend = axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                            ncol=3, shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-large')
    for line in legend.get_lines():
        line.set_linewidth(2)

    fig.set_size_inches(12, 4)
    fig.savefig(args.png_basename + '.png', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('png_basename', type=str,
                        help='basename of the loss/time PNG plots')
    parser.add_argument('pickles', type=str, nargs='+',
                        help='comma-separated list of pickled results')
    args = parser.parse_args()

    draw(args)
