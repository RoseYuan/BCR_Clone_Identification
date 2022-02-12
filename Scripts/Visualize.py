from Utils import *
import matplotlib.pyplot as plt

def plot_nearest_dis_style1(d_to_nearest_all_sample, singletons, non_singletons, d_to_nearest_all_neg,
                            title, x_label, figname, loc_min, binwidth=0.02, cutoff=None, smoothed_curve=None):
    # __ visualize __
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, hspace=0.2, wspace=0)
    fig.suptitle(title)
    fig.supxlabel(x_label)
    fig.supylabel('Frequency')
    (ax), (ax2) = gs.subplots(sharex=True, sharey=True)

    if smoothed_curve is not None:
        ax.plot(smoothed_curve[0, :], smoothed_curve[1, :], label='smoothed', color='grey')

    data1 = d_to_nearest_all_sample[singletons]
    data2 = d_to_nearest_all_sample[non_singletons]
    bins = np.concatenate(
        (np.arange(min(data2), loc_min, binwidth), np.arange(loc_min, max(data1) + binwidth, binwidth)))

    ax.hist(d_to_nearest_all_sample[singletons], bins=bins, alpha=0.5, label='singletons')
    ax.hist(d_to_nearest_all_sample[non_singletons], bins=bins, alpha=0.5, label='non-singletons')
    ax.set_yscale('log')
    ax.set_title("Sample sequences")

    ax.axvline(loc_min, ymin=-0.002, linestyle='--', c='grey')
    ax.text(loc_min - 0.05, 2, "%1f" % loc_min, color='red')

    ax.legend()

    data = d_to_nearest_all_neg
    bins = np.arange(min(data), max(data) + binwidth, binwidth)
    ax2.hist(d_to_nearest_all_neg, bins=bins, color='green', alpha=0.5, label='negation')
    ax2.set_yscale('log')
    ax2.set_title("Negation sequences")

    if cutoff is not None:
        ax2.axvline(cutoff, ymin=-0.002, linestyle='--', c='grey')
        ax2.text(cutoff - 0.05, 2, "%1f" % cutoff, color='red')
        # ax2.text(cutoff + 0.02, 1000, "tolerance = %s" % tolerance)

    ax2.legend()

    fig.savefig(figname)

def plot_nearest_dis_style2(d_to_nearest_all_sample, singletons, non_singletons, d_to_nearest_all_neg,
                            title, x_label, figname, binwidth=0.02, loc_min=None, cutoff=None, annotext=None,
                            smoothed_curve=None):
    # __ visualize __
    fig = plt.figure(figsize=(8, 8))
    fig.supxlabel(x_label)
    fig.supylabel('Frequency')
    ax = fig.add_subplot(111)

    data4 = smoothed_curve
    label4 = data4
    if data4 is not None:
        label4 = 'smoothed'

    data1 = d_to_nearest_all_sample[singletons]
    label1 = 'singletons'
    data2 = d_to_nearest_all_sample[non_singletons]
    label2 = 'non-singletons'

    # negation distribution
    data = d_to_nearest_all_neg
    bins = np.arange(min(data), max(data) + binwidth, binwidth)
    heights, bins = np.histogram(d_to_nearest_all_neg, bins=bins)  # get positions and heights of bars

    bin_width = np.diff(bins)[0]
    bin_pos = bins[:-1] + bin_width / 2
    data3 = np.array([bin_pos, -1 * heights])
    label3 = 'negation'

    ax = plot_nearest_dis_sub(ax, data1, data2, data3, label1, label2, label3, binwidth, title, data4, label4,
                              loc_min, cutoff, annotext)

    fig.savefig(figname)




def plot_nearest_dis_sub(ax, data1, data2, data3, label1, label2, label3, binwidth, title, data4=None, label4=None,
                         loc_min=None, cutoff=None, annotext=None):
    bins = np.arange(min(data1), max(data1) + binwidth, binwidth)
    ax.hist(data1, bins=bins, alpha=0.5, label=label1)
    bins = np.arange(min(data2), max(data2) + binwidth, binwidth)
    ax.hist(data2, bins=bins, alpha=0.5, label=label2)
    ax.set_title(title)

    if data4 is not None:
        ax.plot(data4[0, :], data4[1, :], label=label4, color='grey')

    if loc_min is not None:
        ax.axvline(loc_min, ymin=-0.002, linestyle='--', c='grey')
        ax.text(loc_min - 0.05, 2, "%1f" % loc_min, color='red')

    # plot data
    ax.bar(data3[0, :], data3[1, :], width=np.diff(data3[0, :])[0], alpha=0.5, color='green', label=label3)
    ax.set_yscale('symlog')
    ax.set_yticklabels([abs(x) for x in ax.get_yticks()])

    if cutoff is not None:
        ax.axvline(cutoff, ymin=-0.002, linestyle='--', c='grey')
        ax.axhline(0, xmin=-0.002, c='black', linewidth=0.8)
        ax.text(cutoff, -1, "%s" % str(cutoff)[:6], color='red')
        # ax.text(cutoff, -1, "%s" % str(cutoff)[:5], color='red')
    if annotext is not None:
        ax.annotate(annotext, xy=(cutoff, -5), xytext=(cutoff - 0.1, -100), arrowprops=dict(facecolor='red',
                                                                                            shrink=0.05))
    ax.legend()
    return ax