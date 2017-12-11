#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mojito import load


PERF_TO_TITLE = ['pr', 'rc', 'F1', 'expl. rc', 'expl. fit', '# fixed predictions']


class Tango:
    # From light to dark
    YELLOW = ("#fce94f", "#edd400", "#c4a000")
    ORANGE = ("#fcaf3e", "#f57900", "#ce5c00")
    BROWN = ("#e9b96e", "#c17d11", "#8f5902")
    GREEN = ("#8ae234", "#73d216", "#4e9a06")
    BLUE = ("#729fcf", "#3465a4", "#204a87")
    VIOLET = ("#ad7fa8", "#75507b", "#5c3566")
    RED = ("#ef2929", "#cc0000", "#a40000")
    WHITE = ("#eeeeec", "#d3d7cf", "#babdb6")
    BLACK = ("#888a85", "#555753", "#2e3436")


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


def homogenize_fold_lengths(stat_array):
    """
    Transform one loaded list of arrays of tuples of arrays into a
    homogeneous numpy array by adding cells up to the largest number of samples in a folder
    """
    #
    # assuming first index is for files, second for # folds
    n_files = len(stat_array)
    n_folds = len(stat_array[0])
    n_stats = stat_array[0][0].shape[-1]
    print('# files:{} # folds:{} # stats:{}'.format(n_files,
                                                    n_folds,
                                                    n_stats))

    fold_lengths = [len(stat_array[f][s]) for f in range(n_files) for s in range(n_folds)]
    max_fold_length = max(fold_lengths)

    for f in range(n_files):
        for s in range(n_folds):
            max_size = list(stat_array[f][s].shape)
            max_size[0] = max_fold_length
            new_stats = np.zeros(max_size, dtype=stat_array[f][s].dtype)
            n_samples = len(stat_array[f][s])
            new_stats[:n_samples, :] = stat_array[f][s]
            stat_array[f][s] = new_stats

    stat_array = np.array(stat_array)
    print('homogenized shape {}'.format(stat_array.shape))
    return stat_array


def draw(args):
    plt.style.use('ggplot')

    trace_args, traces, explanation_perfs = [], [], []
    for path in args.pickles:
        data = load(path)
        trace_args.append(data['args'])
        traces.append(data['traces'])
        explanation_perfs.append(data['explanation_perfs'])
        num_examples = data['num_examples']

    # traces = np.array(traces)
    print('\nChecking traces shapes')
    traces = homogenize_fold_lengths(traces)
    # traces indices: [file, fold, iteration, metric]

    # explanation_perfs = np.array(explanation_perfs)
    print('\nChecking exp perfs shapes')
    explanation_perfs = homogenize_fold_lengths(explanation_perfs)
    # shape: [file, fold, num evaluations, test example, measure]

    num_files = traces.shape[0]
    num_perfs = traces.shape[-1]
    fig, axes = plt.subplots(1, num_perfs)

    FACTOR = 100 / num_examples

    # For each performance metric
    for p in range(num_perfs):
        print(traces.shape)
        traces_p = traces[:, :, :, p]

        # For each results file
        x0, ymin, ymax = None, np.inf, -np.inf
        for f in range(num_files):
            traces_f_p = traces_p[f, :, :]

            label, color = get_style(trace_args[f])

            x0f = round(trace_args[f].perc_known)
            if x0 is None:
                x0 = x0f
            # assert x0 == x0f, 'perc_known mismatch'

            xs = x0 + np.arange(traces_f_p.shape[1]) * FACTOR
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

    fig.set_size_inches(4 * num_perfs, 4)
    fig.savefig(args.png_basename + '.png', bbox_inches='tight', pad_inches=0)

    num_perfs = explanation_perfs.shape[-1]
    num_files = explanation_perfs.shape[0]

    fig, axes = plt.subplots(1, num_perfs)

    # For each performance metric
    for p in range(num_perfs):

        # For each result file
        for f in range(num_files):
            perfs = np.mean(explanation_perfs[f, :, :, :, p], axis=2)

            label, color = get_style(trace_args[f])

            xs = np.arange(perfs.shape[1]) * trace_args[f].eval_explanations_every
            ys = np.mean(perfs, axis=0)
            yerrs = np.std(perfs, axis=0) / np.sqrt(perfs.shape[0])

            axes[p].plot(xs, ys, linewidth=2, label=label, color=color)
            axes[p].fill_between(xs, ys - yerrs, ys + yerrs, linewidth=0,
                                 alpha=0.35, color=color)

    legend = axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                            ncol=3, shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-large')
    for line in legend.get_lines():
        line.set_linewidth(2)

    fig.set_size_inches(4 * num_perfs, 4)
    fig.savefig(args.png_basename + '-explanations.png',
                bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('png_basename', type=str,
                        help='basename of the loss/time PNG plots')
    parser.add_argument('pickles', type=str, nargs='+',
                        help='comma-separated list of pickled results')
    args = parser.parse_args()

    draw(args)
