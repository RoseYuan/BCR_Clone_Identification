import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib
import copy
import colorcet
from math import ceil

def main():
    M_dist = 1 - np.load("MI.npy")
    d_threshold = 0.8
    cluster_HAC(M_dist, d_threshold)


def cluster_HAC(Mat_dist, d_threshold,fig1=None,fig2=None,fig3=None):
    # tuto = https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

    """MI is a distance matrix n*n between each element
       d_treshold is the maximum distance of elements within cluster"""

    M_dist = copy.deepcopy(Mat_dist)
    f = M_dist.shape[0]

    # Making the matrix fully symetric (if its not the case already)
    for fi in range(0, f):
        for fj in range(0, f):
            M_dist[fi, fj] = M_dist[fj, fi]

    # Setting diagonal to zero (mandatory, important for the squareform)
    for fi in range(0, f):
        M_dist[fi, fi] = 0

    # ---------------------------------- Plot distance matrix ----------------------------------#
    plt.figure(figsize=(10, 10))
    plt.imshow(M_dist, origin='lower', cmap=plt.cm.get_cmap('jet'))
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.title('Distance matrix')
    cbar = plt.colorbar()
    cbar.set_label('Distance')  # , rotation=270)
    plt.clim(0,ceil(np.max(M_dist)*10)/10)

    if fig1 is not None:
        plt.savefig(fig1)
    else:
        plt.show()

    M_dist_ss = squareform(M_dist)  # the HAC algorithm need the squareform as an input

    d_result = []

    # --------------------------- HAC clustering ----------------------------#
    print("\n")
    print("  Calculating information clusters with hierarchical clustering...")

    Z = linkage(M_dist_ss, 'complete')
    clusters = fcluster(Z, d_threshold, criterion='distance')  # fcluster(Z, k, criterion='maxclust')

    # plot dendrogram
    n_clust = int(np.max(clusters))
    d_result.append(n_clust)
    print("    For a threshold distance of d =", d_threshold, "there is", n_clust, "clusters")

    colors = colorcet.glasbey_light
    set_link_color_palette(colors)
    plt.figure(figsize=(16, 4))
    fancy_dendrogram(Z, p=840, truncate_mode='lastp', annotate_above=20, no_labels=True, max_d=d_threshold)
    if fig2 is not None:
        plt.savefig(fig2)
    else:
        plt.show()

    # Visualizing clusters
    if n_clust <= len(colors):
        cMI = np.zeros((f, f))
        for fi in range(0, f):
            for fj in range(0, f):
                if clusters[fi] == clusters[fj]:
                    cMI[fi, fj] = clusters[fi]
        plt.figure(figsize=(10, 10))
        # plt.rcParams["figure.figsize"] = [10, 10]
        ncol = 0
        for n in range(0, n_clust):
            xs, ys = np.where(cMI == n + 1)
            plt.scatter(ys, xs, marker='s', s=1, color=colors[ncol])
            if np.count_nonzero(
                    clusters == n + 1) == 1:  # the dendrogram skip the color when only 1 element, and we do the same
                ncol -= 1
            ncol += 1
        plt.xlim(0, f)
        plt.ylim(0, f)
        plt.ylabel('Index')
        plt.title("Clusters")
        if fig3 is not None:
            plt.savefig(fig3)
        else:
            plt.show()

    return clusters


def fancy_dendrogram(*args, **kwargs):
    # a build in fucntion to plot better dendrogram
    with plt.rc_context({'lines.linewidth': 0.5}):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram')
            # plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
    return ddata


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 18})
    main()
