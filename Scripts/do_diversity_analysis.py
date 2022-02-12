from libBiodiversity import *
import pandas as pd
from collections import Counter
from scipy import stats
import itertools

# calculate the diversity index, save to files;
# calculate spearman correlation coefficient
# visualize

def diversity_index(path_data,samples,div_file):

    div_indexes = [richness, richness_chao, Shannon_entropy, Shannon_entropy_Chao, Simpson_index, eveness, eveness_chao,
                   dominance]
    c_name = ['junction-based', 'v-j-junction-based', 'align-free']

    df_div = pd.DataFrame(columns=[i.__name__ for i in div_indexes])

    for sample in samples:
        clusters0 = np.load(path_data + "clustering/clusters_%s_junction.npy" % sample)
        clusters1 = np.load(path_data + "clustering/clusters_%s_V-J-junction.npy" % sample)
        clusters2 = np.load(path_data + "clustering/clusters_%s_a-free.npy" % sample)

        # __ get the clustering result as a count dictionary
        c_base = dict(Counter(clusters0))
        c_align = dict(Counter(clusters1))
        c_a_free = dict(Counter(clusters2))

        clusterings = [c_base, c_align, c_a_free]

        for ci in range(len(c_name)):
            l = len(df_div)
            df_div.loc[l, 'sample'] = sample
            df_div.loc[l, 'clustering method'] = c_name[ci]
            for index in div_indexes:
                df_div.loc[l, index.__name__] = index(clusterings[ci])

    print('Calculate diversity index of clustering for each sample and each clonotyping methods.')
    df_div.to_csv(div_file,sep='\t',index=False)
    return

def consistency_index_real(path_data, div_file):
    c_name = ['junction-based', 'v-j-junction-based', 'align-free']

    div_indexes = [richness, richness_chao, Shannon_entropy, Shannon_entropy_Chao, Simpson_index, eveness, eveness_chao,
                   dominance]
    df_div = pd.read_csv(div_file, sep='\t')
    method_comb = []
    c_name.sort()
    for method1, method2 in itertools.combinations(c_name, r=2):
        method_comb.append('%s vs %s' % (method1, method2))
    df_spearman = pd.DataFrame(columns=['indices', 'combinations', 'correlation', 'p_value'])

    for i in div_indexes:
        count = -1
        df_index = df_div.pivot(index='sample', columns='clustering method', values=i.__name__)
        index = df_index.values
        rho, pval = stats.spearmanr(index)
        for m in range(len(c_name)):
            for n in range(len(c_name)):
                if m < n:
                    count += 1
                    df_spearman = df_spearman.append({'indices': i.__name__, 'combinations': method_comb[count],
                                                      'correlation': rho[m, n], 'p_value': pval[m, n]},
                                                     ignore_index=True)
    df_indices = df_spearman[["indices", "correlation"]].groupby('indices').mean()
    df_indices.sort_values('correlation', inplace=True)
    df_indices['indices'] = df_indices.index
    df_indices = df_indices.reset_index(drop=True)

    y_labels = df_indices['indices']

    # Plot the figure.
    plt.figure(figsize=(8, 5))
    ax = df_indices['correlation'].plot(kind='barh', zorder=3,
                                        color=['gold', 'tab:blue', 'tab:grey', 'tab:green', 'tab:orange', 'tab:purple',
                                               'tab:pink', 'tab:red'], fontsize=14)
    ax.set_title('Average Spearman correlation coefficient for measured diversity index', fontsize=14)
    ax.set_xlabel('Spearman correlation coefficient', fontsize=14)
    ax.set_ylabel('Diversity index', fontsize=14)
    ax.set_yticklabels(y_labels, fontsize=14)
    ax.set_xlim(0.4, 1.0)  # expand xlim to make labels easier to read

    rects = ax.patches

    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        ha = 'left'

        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            # Invert space to place label to the left
            space *= -1
            # Horizontally align label at right
            ha = 'right'

        # Use X value as label and format number with one decimal place
        label = "{:.3f}".format(x_value)

        # Create annotation
        plt.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(space, 0),  # Horizontally shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            va='center',  # Vertically center label
            ha=ha)  # Horizontally align label differently for
        # positive and negative values.
    plt.grid(linewidth=0.5, zorder=0, axis='x')
    plt.rcParams.update({'font.size': 14})
    print("Calculate the consistency of diversity index across different clonotyping methods.")
    plt.savefig(path_data + "fig/diversity_index_real.png", bbox_inches='tight', pad_inches=0.5)


    fig, axs = plt.subplots(figsize=(13, 5))
    i = div_indexes[2]
    fig.suptitle("Shannon entropy for each sample", fontsize=16)
    df_index = df_div.pivot(index='sample', columns='clustering method', values=i.__name__)
    axs.plot(df_index['v-j-junction-based'], label='V-J-junction based')
    axs.plot(df_index['align-free'], label='alignment free')
    axs.plot(df_index['junction-based'], label='junction based')
    axs.legend()
    axs.set(ylabel="Shannon entropy")
    axs.tick_params('x', labelrotation=-45)
    # set the spacing between subplots
    # wspace and hspace specify the space reserved between Matplotlib subplots.
    # They are the fractions of axis width and height, respectively.
    # left, right, top and bottom parameters specify four sides of the subplots’ positions.
    # They are the fractions of the width and height of the figure.
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2,
                        hspace=0.5)
    plt.rcParams.update({'font.size': 16})

    fig.savefig(path_data + "fig/Shannon_entropy_real.png", bbox_inches='tight', pad_inches=0.5)
    return

def diversity_profile(path_data,samples,div_profile, div_profile_fig):
    a = np.arange(-100, 101, 1) / 100
    y2 = transform_func(a)
    alphas = y2[y2 <= 150]

    df_div = pd.DataFrame(columns=['alpha=%s' % i for i in alphas])


    c_name = ['junction-based', 'v-j-junction-based', 'align-free']

    for sample in samples:
        clusters0 = np.load(path_data + "clustering/clusters_%s_junction.npy" % sample)
        clusters1 = np.load(path_data + "clustering/clusters_%s_V-J-junction.npy" % sample)
        clusters2 = np.load(path_data + "clustering/clusters_%s_a-free.npy" % sample)

        # __ get the clustering result as a count dictionary
        c_base = dict(Counter(clusters0))
        c_align = dict(Counter(clusters1))
        c_a_free = dict(Counter(clusters2))

        clusterings = [c_base, c_align, c_a_free]

        for ci in range(len(c_name)):
            l = len(df_div)
            df_div.loc[l, 'sample'] = sample
            df_div.loc[l, 'clustering method'] = c_name[ci]
            for alpha in alphas:
                df_div.loc[l, 'alpha=%s' % alpha] = Hill_diversity(clusterings[ci], alpha)

    df_div.to_csv(div_profile, sep='\t', index=False)

    show_alpha = np.append(np.arange(0, 11, 1) / 10, np.array([2, 4, 6, 8, 10, 100, np.inf]))
    alpha_axis = []
    for i in show_alpha:
        if i > 1 and i < 101:
            alpha_axis.append(int(i))
        else:
            alpha_axis.append(i)

    initial_axis = rev_transform(alpha_axis)

    print("Calculate diversity profiles.")
    plt.figure(figsize=(22, 9))
    for s, sample in enumerate(samples):
        index = np.logical_and(df_div['clustering method'] == 'align-free', df_div['sample'] == sample)
        y = df_div.loc[index, :].iloc[:, 0:-2].values.T
        y = np.append(y, y[-1])
        plt.plot(np.append(a[y2 <= 150], 1), y, label=sample)
        plt.xlabel("Alpha in Hill's diversity index", fontsize=26)
        plt.ylabel("Hill's diversity index", fontsize=26)
        plt.title('Diversity profiles of real data based on alignment-free method', fontsize=26)
        plt.xticks(initial_axis, alpha_axis, fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=15)
        plt.xlim(-1.1, 1.3)
    plt.yscale('log')
    plt.savefig(div_profile_fig)
    return

def consistency_profile(div_profile, cp_profile):
    a = np.arange(-100, 101, 1) / 100
    y2 = transform_func(a)
    alphas = y2[y2 <= 150]

    c_name = ['junction-based', 'v-j-junction-based', 'align-free']
    df_div = pd.read_csv(div_profile, sep='\t')

    method_comb = []
    c_name.sort()
    for method1, method2 in itertools.combinations(c_name, r=2):
        method_comb.append('%s vs %s' % (method1, method2))
    df_div = df_div[df_div['sample'] != 'sample76-99']
    df_spearman = pd.DataFrame(columns=['indices', 'combinations', 'correlation', 'p_value'])

    for a in alphas:
        count = -1
        df_index = df_div.pivot(index='sample', columns='clustering method', values='alpha=%s' % a)
        index = df_index.values
        rho, pval = stats.spearmanr(index)
        for m in range(len(c_name)):
            for n in range(len(c_name)):
                if m < n:
                    count += 1
                    df_spearman = df_spearman.append({'indices': a, 'combinations': method_comb[count],
                                                      'correlation': rho[m, n], 'p_value': pval[m, n]},
                                                     ignore_index=True)
    df_indices = df_spearman[["indices", "correlation"]].groupby('indices').mean()
    df_indices['indices'] = df_indices.index
    df_indices = df_indices.reset_index(drop=True)
    print('Calculate the consistency of diversity profiles across different clonotyping methods.')
    df_indices.to_csv(cp_profile, index=False)
    return

def consistency_profile_visualize(cp_1,cp_2,cp_fig):
    a = np.arange(-100, 101, 1) / 100
    y2 = transform_func(a)

    show_alpha = np.append(np.arange(0, 11, 1) / 10, np.array([2, 4, 6, 10, 100, np.inf]))
    alpha_axis = []
    for i in show_alpha:
        if i > 1 and i < 101:
            alpha_axis.append(int(i))
        else:
            alpha_axis.append(i)

    df_indices1 = pd.read_csv(cp_1)
    df_indices2 = pd.read_csv(cp_2)

    initial_axis = rev_transform(alpha_axis)
    plt.figure(figsize=(20, 6))
    plt.plot(np.append(a[y2 <= 150], 1),
             np.append(df_indices1['correlation'].values, df_indices1['correlation'].values[-1]), label='real data')
    plt.plot(np.append(a[y2 <= 150], 1),
             np.append(df_indices2['correlation'].values, df_indices2['correlation'].values[-1]),
             label='simulated data')
    plt.xticks(initial_axis, alpha_axis, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Alpha in Hill's diversity index", fontsize=16)
    plt.ylabel('Spearman correlation coefficient', fontsize=16)
    plt.title("Measuring consistency of Hill's diversity index", fontsize=16)
    plt.legend(fontsize=16)
    plt.rcParams.update({'font.size': 16})
    plt.savefig(cp_fig)
    return

if __name__ == '__main__':
    path_data = '/home/siyuan/thesis/Data/new_data/rerun/'
    samples = ["sample%s" % n for n in
                    [76, 77, 78, 79, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]
    div_file = path_data + 'diversity/diversity_index_real.csv'
    diversity_index(path_data,samples,div_file)
    consistency_index_real(path_data,div_file)

    div_profile = path_data + 'diversity/diversity_profile_real.csv'
    div_profile_fig = path_data + "fig/diversity_profile_real.png"
    diversity_profile(path_data,samples,div_profile,div_profile_fig)
    cp_profile = path_data + 'diversity/consistency_profile_real.csv'
    consistency_profile(div_profile,cp_profile)

    path_data = '/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/'
    samples = ["sampleMS2_%s" % (m + 1) for m in range(20)]
    div_profile = path_data + 'diversity/diversity_profile_sim.csv'
    div_profile_fig = path_data + "fig/diversity_profile_sim.png"
    diversity_profile(path_data,samples,div_profile,div_profile_fig)
    cp_profile = path_data + 'diversity/consistency_profile_sim.csv'
    consistency_profile(div_profile,cp_profile)

    cp_1 = '/home/siyuan/thesis/Data/new_data/rerun/' + 'diversity/consistency_profile_real.csv'
    cp_2 = '/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/' + 'diversity/consistency_profile_sim.csv'
    cp_fig = '/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/' + "fig/Hills_diversity_index_consistency.png"
    consistency_profile_visualize(cp_1, cp_2, cp_fig)