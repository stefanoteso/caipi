#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from caipi import load


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

    label = args.learner + ' ' + args.strategy
    if args.improve_expl:
        label += ' (EI)'
    else:
        label += ' (NO EI)'

    base_color = {
        'svm': Tango.RED,
        'l1svm': Tango.VIOLET,
        'lr': Tango.GREEN,
        'gp': Tango.BLUE,
    }[args.learner]

    shade = {
        'random': 0,
        'least-confident': 2,
        'most-variance': 2,
        'least-margin': 1,
    }[args.strategy]
    color = base_color[shade]

    style, marker = {
        True: ('-', 's'),
        False: ('-.', '*'),
    }[args.improve_expl]

    return label, color, style, marker


def draw(args):
    plt.style.use('ggplot')
    np.set_printoptions(precision=2, linewidth=80, threshold=np.nan)

    pickle_data, pickle_args = [], []
    for path in args.pickles:
        data = load(path)
        pickle_data.append(data['perfs'])
        pickle_args.append(data['args'])

    min_folds = min(list(len(datum) for datum in pickle_data))
    perfs = np.array([datum[:min_folds] for datum in pickle_data])

    # traces has shape: [n_pickles, n_folds, n_iters, n_measures]
    TO_TITLE = [
        'Pred. Pr', 'Pred. Rc', 'Pred. F1',
        'Expl. Pr', 'Expl. Rc', 'Expl. F1',
        '# of corrections',
    ]

    for i_measure in range(perfs.shape[-1]):

        print(TO_TITLE[i_measure])
        print(perfs[:, :, :, i_measure])

        fig, ax = plt.subplots(1, 1)
        ax.set_title(TO_TITLE[i_measure])
        ax.set_xlabel('# iterations')

        for i_pickle in range(perfs.shape[0]):
            perf = perfs[i_pickle, :, :, i_measure]

            y = np.mean(perf, axis=0)
            yerr = np.std(perf, axis=0) / np.sqrt(perf.shape[0])
            if -1 in y:
                yerr = yerr[y != -1]
                y = y[y != -1]
            x = np.arange(y.shape[0])

            label, color, style, marker = get_style(pickle_args[i_pickle])
            ax.plot(x, y, label=label, color=color,
                    linestyle=style, linewidth=2)
            ax.fill_between(x, y - yerr, y + yerr, color=color,
                            alpha=0.35, linewidth=0)

        legend = ax.legend(loc='upper center',
                           bbox_to_anchor=(0.5, 1.25),
                           ncol=3,
                           shadow=False)

        fig.savefig(args.basename + '_{}.png'.format(i_measure),
                    bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('basename', type=str,
                        help='basename of the loss/time PNG plots')
    parser.add_argument('pickles', type=str, nargs='+',
                        help='comma-separated list of pickled results')
    args = parser.parse_args()

    draw(args)
