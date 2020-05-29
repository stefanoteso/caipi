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

    label = {
        'svm': 'SVM',
        'l1svm': 'L1 SVM',
        'lr': 'LR',
    }[args.learner]

    if args.start_expl_at >= 0:
        label += ' + Corr.'

    base_color = {
        'svm': Tango.RED,
        'l1svm': Tango.VIOLET,
        'lr': Tango.GREEN,
    }[args.learner]

    shade = 0 if args.start_expl_at >= 0 else 2
    color = base_color[shade]

    style, marker = {
        True: ('-', 's'),
        False: (':', '*'),
    }[args.start_expl_at >= 0]

    return label, color, style, marker


def draw(args):
    plt.style.use('ggplot')
    np.set_printoptions(precision=2, linewidth=80, threshold=np.inf)

    pickle_data, instant_data, pickle_args = [], [], []
    for path in args.pickles:
        data = load(path)
        pickle_data.append(np.array(data['perfs']))
        try:
            instant_data.append(np.array(data['instant_perfs']))
        except KeyError:
            pass
        pickle_args.append(data['args'])

    min_folds = min(list(len(datum) for datum in pickle_data))
    print('# folds =', min_folds)
    min_measures = min(datum.shape[-1] for datum in pickle_data)
    perfs = np.array([datum[:min_folds,:,:min_measures] for datum in pickle_data])
    instant_perfs = np.array([datum[:min_folds] for datum in instant_data])
    max_iters = pickle_args[0].max_iters

    # perfs have shape: [n_pickles, n_folds, n_iters, n_measures]
    if perfs.shape[-1] == 3:
        to_title = [
            'Predictive F1',
            'Confictive Rec.',
            '# Corrections',
        ]
    else:
        to_title = [
            'Predictive Pr', 'Predictive Rc', 'Predictive F1',
            'Explanatory Pr', 'Explanatory Rc', 'Explanatory F1',
            '# Corrections',
        ]

    to_title_inst = [
        'Inst. Predictive Pr', 'Inst. Predictive Rc', 'Inst. Predictive F1',
        'Inst. Explanatory Pr', 'Inst. Explanatory Rc', 'Inst. Explanatory F1',
    ]

    for i_measure in range(perfs.shape[-1]):

        #print(to_title[i_measure])
        #print(perfs[:, :, :, i_measure])

        fig, ax = plt.subplots(1, 1)
        ax.set_title(to_title[i_measure], fontsize=16)
        ax.set_xlabel('Iterations', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        if to_title[i_measure].startswith('Predictive'):
            ax.set_ylim(args.min_pred_val, args.max_pred_val)
        else:
            n_ticks = len(ax.get_xticklabels())
            eval_iters = max_iters // n_ticks
            labels = list(range(0, max_iters, eval_iters))
            ax.set_xticklabels(['dunno'] + [str(l) for l in labels])

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

        legend = ax.legend(loc='lower right',
                           fontsize=16,
                           shadow=False)

        fig.savefig(args.basename + '_{}.png'.format(i_measure),
                    bbox_inches='tight', pad_inches=0)

    if not len(instant_perfs):
        print('Your pickle file does not have instant perfs, skipped.')
        return

    for i_measure in range(instant_perfs.shape[-1]):

        #print(to_title[i_measure])
        #print(perfs[:, :, :, i_measure])

        fig, ax = plt.subplots(1, 1)
        ax.set_title(to_title_inst[i_measure], fontsize=16)
        ax.set_xlabel('Iterations', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        if to_title[i_measure].startswith('Predictive'):
            ax.set_ylim(args.min_inst_pred_val, args.max_inst_pred_val)
        elif eval_iters > 0:
            pass

        for i_pickle in range(instant_perfs.shape[0]):
            perf = instant_perfs[i_pickle, :, :, i_measure]

            y = np.cumsum(perf, axis=1)
            y = np.mean(y, axis=0)
            y *= 1 / np.arange(1, len(y) + 1)
            yerr = np.std(y, axis=0) / np.sqrt(y.shape[0])
            if -1 in y:
                yerr = yerr[y != -1]
                y = y[y != -1]
            x = np.arange(y.shape[0])

            label, color, style, marker = get_style(pickle_args[i_pickle])
            ax.plot(x, y, label=label, color=color,
                    linestyle=style, linewidth=2)
            ax.fill_between(x, y - yerr, y + yerr, color=color,
                            alpha=0.35, linewidth=0)

        legend = ax.legend(loc='lower right',
                           fontsize=16,
                           shadow=False)

        fig.savefig(args.basename + '_instant_{}.png'.format(i_measure),
                    bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('basename', type=str,
                        help='basename of the loss/time PNG plots')
    parser.add_argument('pickles', type=str, nargs='+',
                        help='comma-separated list of pickled results')
    parser.add_argument('--min-pred-val', type=float, default=0,
                        help='minimum pred. score')
    parser.add_argument('--max-pred-val', type=float, default=1.05,
                        help='minimum pred. score')
    parser.add_argument('--min-inst-pred-val', type=float, default=0,
                        help='minimum instantaneous pred. score')
    parser.add_argument('--max-inst-pred-val', type=float, default=1.05,
                        help='minimum instantaneous pred. score')
    parser.add_argument('--legend', action='store_true',
                        help='whether to draw the legend')
    args = parser.parse_args()

    draw(args)
