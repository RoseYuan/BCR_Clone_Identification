from Utils import *
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


"""
Detecting the threshold:
"""


def outlier_based_cutoff(d_to_nearest_all, visualize=True, x_label ="Normalized Hamming distance to nearest neighbor",figname=None):
    bins = 30
    cov = 1.5
    # smoothing
    freq = np.histogram(d_to_nearest_all, bins=bins)
    freq_pos = [f if f > 0 else f + 1 for f in freq[0]]  # mask zero, otw log-scale on y axis wouldn't work
    smoothed_freq0 = gaussian_filter1d(np.log(freq_pos), cov)

    # ___ detection of local minimums and maximums ___
    min_ind = (np.diff(np.sign(np.diff(smoothed_freq0))) > 0).nonzero()[0] + 1
    try:
        min_ind = min_ind[0:1]
    except:
        pass
    # local min
    loc_min = freq[1][min_ind]

    if visualize:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Distance-to-nearest neighbor distribution", fontsize=14)
        # the histogram
        ax[0].hist(d_to_nearest_all, bins=bins, label="frequency")
        ax[0].set_yscale('log')
        ax[0].set_xlabel(x_label, fontsize=12)
        ax[0].set_ylabel("Frequency", fontsize=12)
        ax[0].plot(freq[1][:-1], np.exp(smoothed_freq0), label='smoothed')
        ax[0].legend(fontsize=12)
        # the density estimate
        ax[1].plot(freq[1][:-1], np.exp(smoothed_freq0), label='smoothed', c='orange')
        ax[1].set_yscale('log')
        ax[1].set_xlabel(x_label, fontsize=12)
        ax[1].set_ylabel("Frequency", fontsize=12)
        ax[1].set_ylim(ax[0].get_ylim())
        # the local minimum
        plt.axvline(loc_min, ymin=-0.002, linestyle='--', c='grey')
        plt.text(loc_min - 0.05, 1, "%1f" % loc_min[0], color='red')
        plt.grid()
        ax[1].legend(fontsize=12)
        # save fig
        if figname is not None:
            plt.savefig(figname)
        else:
            plt.show()

    return loc_min


def negation_based_cutoff(d_to_nearest_cp, tolerance, visualize=True, figname=None):
    """
    :param d_to_nearest_cp: the distribution of distances between negation sequences and their closest
    counterpart in the repertoire
    :param tolerance: the fraction of the distances to negation sequences that are allowed within the
    cluster (false-positive rate)
    :return: the cutoff value
    """
    d_threshold = np.quantile(d_to_nearest_cp, tolerance)
    if visualize:
        bins = 30
        plt.figure(figsize=(5,5))
        plt.hist(d_to_nearest_cp, bins=bins, label="negation sequences")
        plt.axvline(d_threshold, ymin=-0.002, linestyle='--', c='grey')
        plt.text(d_threshold - 0.05, 1, "%1f" % d_threshold, color='red')
        plt.grid()
        plt.legend()
        if figname is not None:
            plt.savefig(figname)
        else:
            plt.show()
    return d_threshold