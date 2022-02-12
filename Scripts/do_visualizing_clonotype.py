from Visualize import *
from DetectThreshold import *
from sklearn.metrics.cluster import adjusted_rand_score
import seaborn as sns
from Clonotyping import *

# script, for both datasets
# compare clone identification results (heatmap)
# distance to nearest neighbor distribution

def compare_clustering_real(path_data,samples):

    df_ari = pd.DataFrame(columns=['rand index_0_1', 'rand index_0_2', 'rand index_1_2'])
    for sample in samples:
        clusters0 = np.load(path_data + "clustering/clusters_%s_junction.npy"%sample)
        clusters1 = np.load(path_data + "clustering/clusters_%s_V-J-junction.npy"%sample)
        clusters2 = np.load(path_data + "clustering/clusters_%s_a-free.npy"%sample)

        df_ari.loc[sample, :] = {'rand index_0_1': adjusted_rand_score(clusters1, clusters0),
                                 'rand index_0_2': adjusted_rand_score(clusters2, clusters0),
                                 'rand index_1_2': adjusted_rand_score(clusters1, clusters2)}
        print(
            "For sample %s:\n %d clusters using baseline method, %d clusters using alignment-based method, %d clusters using alignment-free method;\n rand index_0_1 :%f,rand index_0_2 :%f,rand index_1_2 :%f"%(sample, np.max(clusters0), np.max(clusters1), np.max(clusters2),
               adjusted_rand_score(clusters1, clusters0), adjusted_rand_score(clusters0, clusters2),
               adjusted_rand_score(clusters1, clusters2)))

    c_name = ["junction based", "V-D-junction based", "alignment free"]
    cluster_similarity = np.zeros((len(c_name), len(c_name)))

    for i in range(len(c_name)):
        for j in range(len(c_name)):
            if i + j - 1 < 3:
                cluster_similarity[i, j] = df_ari.iloc[:, i + j - 1].mean()

    fig = plt.figure(figsize=(10, 8.3))
    di = np.diag_indices(cluster_similarity.shape[0])
    cluster_similarity[di] = 1

    vmin = np.min(cluster_similarity)
    vmax = np.max(cluster_similarity)

    x_axis_labels = c_name
    y_axis_labels = c_name

    p = sns.heatmap(cluster_similarity, annot=True, cmap='BuGn', vmin=vmin, vmax=vmax,
                    cbar_kws={'label': 'Adjusted Rand Index'}, xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                    annot_kws={"size": 16})
    p.set_title("Similarity between clustering results", fontsize=16)
    p.set_yticklabels(y_axis_labels, va='center', rotation=90, position=(0, 0.28), fontsize=16)
    p.set_xticklabels(p.get_xmajorticklabels(), fontsize=16)
    p.figure.axes[-1].yaxis.label.set_size(16)
    cbar = p.collections[0].colorbar
    # here set the labelsize by 16
    cbar.ax.tick_params(labelsize=16)
    fig.savefig(path_data + "fig/cluster_similarity_real.png")

def singleton_distribution_real(path_data):
    print("Plot the distance to nearest neighbor distribution with Levenshtein distance.")
    dis_neg = np.load(path_data + 'distance/dis_negation_v-j-junction.npy')
    d_to_nearest_all_neg = dist_to_nearest(dis_neg)
    dis = np.load(path_data+'distance/dis_sample76-99_v-j-junction.npy')
    d_to_nearest_all = dist_to_nearest(dis)

    # __ smoothing __
    cov = 1.5
    freq = np.histogram(d_to_nearest_all[d_to_nearest_all < 1], bins=30)
    freq_pos = [f if f > 0 else f + 1 for f in freq[0]]  # mask zero
    smoothed_freq0 = gaussian_filter1d(np.log(freq_pos), cov)
    smoothed_curve = np.array([freq[1][:-1], np.exp(smoothed_freq0)])

    # __ detect local minimum __
    min_ind = (np.diff(np.sign(np.diff(smoothed_freq0))) > 0).nonzero()[0] + 1
    loc_min = freq[1][min_ind][0]

    title = "Sample 76-99"
    x_label = "Normalized Levenshtein distance"
    figname = path_data + "fig/d-to-nearest-v-j-junction.png"

    singletons = d_to_nearest_all >= loc_min
    non_singletons = d_to_nearest_all < loc_min

    binwidth = 0.02
    plt.rcParams.update({'font.size': 16})
    plot_nearest_dis_style2(d_to_nearest_all[d_to_nearest_all < 1], singletons[d_to_nearest_all < 1],
                            non_singletons[d_to_nearest_all < 1],
                            d_to_nearest_all_neg[d_to_nearest_all_neg < 1],
                            title, x_label, figname, binwidth=binwidth, cutoff=loc_min, annotext=None,
                            smoothed_curve=smoothed_curve)

    print("Plot the distance to nearest neighbor distribution with Cosine distance.")
    tolerance = 0.0003
    dis = np.load(path_data + "distance/dis_sample76-99_align-free.npy")
    dis_neg = np.load(path_data + "distance/dis_negation_align-free.npy")
    d_to_nearest_all_neg = dist_to_nearest(dis_neg)
    d_to_nearest_all = dist_to_nearest(dis)

    # d==1: excluded
    d_to_nearest_all_neg_1 = d_to_nearest_all_neg[d_to_nearest_all_neg < 1]
    # __ detect threshold by negation __
    cutoff = negation_based_cutoff(d_to_nearest_all_neg_1, tolerance)

    title = " Sample 76-99"
    x_label = "Normalized Cosine distance"
    figname = path_data + "fig/d-to-nearest-align-free.png"
    annotext = "tolerance = 0.03%"
    binwidth = 0.01
    # __ smoothing __
    cov = 1.5
    freq = np.histogram(d_to_nearest_all, bins=30)
    freq_pos = [f if f > 0 else f + 1 for f in freq[0]]  # mask zero
    smoothed_freq0 = gaussian_filter1d(np.log(freq_pos), cov)
    smoothed_curve = np.array([freq[1][:-1], np.exp(smoothed_freq0)])
    plt.rcParams.update({'font.size': 16})
    plot_nearest_dis_style2(d_to_nearest_all, singletons, non_singletons, d_to_nearest_all_neg_1,
                            title, x_label, figname, binwidth=binwidth, cutoff=cutoff, annotext=annotext,
                            smoothed_curve=smoothed_curve)
    return

def compare_clustering_simulated(path_data):
    samples = ["sampleMS2_%s"%(m + 1) for m in range(20)]
    clone_samples = ["2_vv%s"%(i+1) for i in range(20)]
    df_ari = pd.DataFrame(
        columns=['rand index_0_1', 'rand index_0_2', 'rand index_0_3', 'rand index_1_2', 'rand index_1_3',
                 'rand index_2_3'])
    for s, sample in enumerate(samples):
        clusters0 = np.load(path_data + "clustering/clusters_%s_junction.npy"%sample)
        clusters1 = np.load(path_data + "clustering/clusters_%s_V-J-junction.npy"%sample)
        clusters2 = np.load(path_data + "clustering/clusters_%s_a-free.npy"%sample)
        outfile = path_data + "datasets/" + sample + "_Nt_info.csv"
        df = pd.read_csv(outfile, sep='\t')
        df_clone = pd.read_csv(path_data + "datasets/clone_dataMS%s.csv"%clone_samples[s], sep=',', index_col=0)
        df_clone = df_clone.loc[df['Sequence ID'].values]
        clusters3 = df_clone['SEED_SEQUENCE_ID'].values

        df_ari.loc[sample, :] = {'rand index_0_1': adjusted_rand_score(clusters1, clusters0),
                                 'rand index_0_2': adjusted_rand_score(clusters2, clusters0),
                                 'rand index_1_2': adjusted_rand_score(clusters1, clusters2),
                                 'rand index_0_3': adjusted_rand_score(clusters3, clusters0),
                                 'rand index_1_3': adjusted_rand_score(clusters1, clusters3),
                                 'rand index_2_3': adjusted_rand_score(clusters2, clusters3), }
        print(
            "For sample %s:\n %d clusters using baseline method, %d clusters using alignment-based method, %d clusters using alignment-free method;\n rand index_0_1 :%f,rand index_0_2 :%f,rand index_1_2 :%f"%(sample, np.max(clusters0), np.max(clusters1), np.max(clusters2),
               adjusted_rand_score(clusters1, clusters0), adjusted_rand_score(clusters0, clusters2),
               adjusted_rand_score(clusters1, clusters2)))


    c_name = ["junction based","V-D-junction based","alignment free","groundtruth"]
    cluster_similarity = np.zeros((len(c_name), len(c_name)))
    cluster_similarity[0, 1] = cluster_similarity[1, 0] = df_ari.iloc[:, 0].mean()
    cluster_similarity[0, 2] = cluster_similarity[2, 0] = df_ari.iloc[:, 1].mean()
    cluster_similarity[0, 3] = cluster_similarity[3, 0] = df_ari.iloc[:, 2].mean()
    cluster_similarity[1, 2] = cluster_similarity[2, 1] = df_ari.iloc[:, 3].mean()
    cluster_similarity[1, 3] = cluster_similarity[3, 1] = df_ari.iloc[:, 4].mean()
    cluster_similarity[2, 3] = cluster_similarity[3, 2] = df_ari.iloc[:, 5].mean()

    fig = plt.figure(figsize=(12.8, 10.5))
    di = np.diag_indices(cluster_similarity.shape[0])
    cluster_similarity[di] = 1

    vmin = 0.32  # np.min(cluster_similarity)
    vmax = np.max(cluster_similarity)

    x_axis_labels = c_name
    y_axis_labels = c_name

    p = sns.heatmap(cluster_similarity, annot=True, cmap='BuGn', vmin=vmin, vmax=vmax,
                    cbar_kws={'label': 'Adjusted Rand Index'}, xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                    annot_kws={"size": 16})
    p.set_title("Similarity between clustering results", fontsize=16)
    p.set_yticklabels(y_axis_labels, va='center', rotation=90, position=(0, 0.28), fontsize=16)
    p.set_xticklabels(p.get_xmajorticklabels(), fontsize=16)

    p.figure.axes[-1].yaxis.label.set_size(16)

    cbar = p.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    fig.savefig(path_data + "fig/cluster_similarity_sim.png")

def get_true_singleton(path_data,clone_file,sample):
    df_clone = pd.read_csv(path_data + clone_file, sep=',', index_col=0)
    outfile = path_data + "datasets/%s_Nt_info.csv"%sample
    df = pd.read_csv(outfile, sep='\t')
    df_unique = df.drop_duplicates(subset="JUNCTION", ignore_index=True)
    df_clone = df_clone.loc[df_unique['Sequence ID'].values]
    df_clone['clone size'] = 0
    df_clone['clone size'] = df_clone.groupby(['SEED_SEQUENCE_ID']).transform('count')
    singletons = df_clone['clone size'] == 1
    singletons.to_csv(path_data + "datasets/singletons_%s_truth.csv"%sample,index=False)

def singleton_distribution_sim(path_data, sample,metric,style):

    outfile = path_data + 'datasets/' + sample + "_Nt_info.csv"
    df = pd.read_csv(outfile, sep='\t')
    df_neg = pd.read_csv("/home/siyuan/thesis/Data/new_data/rerun/datasets/negative_table_Nt_info.csv",
                         sep='\t').drop_duplicates(
        subset=['JUNCTION'], ignore_index=True)
    df_unique = df.drop_duplicates(subset=['JUNCTION'], ignore_index=True)

    if metric == 1 and style == 1:
        print('Calculate pairwise distance according to v-j-junction based distance.')
        dis, _ = v_j_junction_dis(df, field_group='JUNCTION', field_dis='JUNCTION',
                                  groupby=['V-GENE', 'J-GENE'], metric=Normalized_Levenshtein_dist,
                                  diag=1., default_dist=1.)
        d_to_nearest_all = dist_to_nearest(dis)
        dis_neg = cal_group_pdist(df_unique, fields='JUNCTION', groupby=['V-GENE', 'J-GENE'], df2 = df_neg, metric=Normalized_Levenshtein_dist,
                                  diag=1., default_dist=1.)
        d_to_nearest_all_neg = dist_to_nearest(dis_neg)

        # __ smoothing __
        cov = 1.5
        freq = np.histogram(d_to_nearest_all[d_to_nearest_all < 1], bins=30)
        freq_pos = [f if f > 0 else f + 1 for f in freq[0]]  # mask zero
        smoothed_freq0 = gaussian_filter1d(np.log(freq_pos), cov)
        smoothed_curve = np.array([freq[1][:-1], np.exp(smoothed_freq0)])

        # __ detect local minimum __
        min_ind = (np.diff(np.sign(np.diff(smoothed_freq0))) > 0).nonzero()[0] + 1
        loc_min = freq[1][min_ind][0]

        title = sample
        x_label = "Normalized Levenshtein distance"
        figname = path_data + "fig/d-to-nearest-v-j-junction.png"

        singletons = d_to_nearest_all >= loc_min
        non_singletons = d_to_nearest_all < loc_min

        binwidth = 0.02
        plt.rcParams.update({'font.size': 16})
        plot_nearest_dis_style2(d_to_nearest_all[d_to_nearest_all < 1], singletons[d_to_nearest_all < 1],
                                non_singletons[d_to_nearest_all < 1],
                                d_to_nearest_all_neg[d_to_nearest_all_neg < 1],
                                title, x_label, figname, binwidth=binwidth, cutoff=loc_min, annotext=None,
                                smoothed_curve=smoothed_curve)
        np.save(path_data+"dis_%s_v-j-junction.npy"%sample,dis)
        np.save(path_data+"dis_neg_%s_v-j-junction.npy"%sample, dis_neg)

    elif metric == 2 and style == 1:
        print("Plot the distance to nearest neighbor distribution with Cosine distance.")
        tolerance = 0.0003


        dis_cp, rep1, rep2 = cal_numeric_represent_pdis(df_unique, fields='Sequence', transform_func=tf_idf_BCR,
                                                        metric=Cosine_dist, diag=1., df2=df_neg, k=4, l=130)
        d_to_nearest_all_neg = dist_to_nearest(dis_cp)

        dis = align_free_dis(df, df_unique, rep1, 'JUNCTION', metric=Cosine_dist, diag=1.)
        d_to_nearest_all = dist_to_nearest(dis)

        # d==1: excluded
        d_to_nearest_all_neg_1 = d_to_nearest_all_neg[d_to_nearest_all_neg < 1]
        # __ detect threshold by negation __
        cutoff = negation_based_cutoff(d_to_nearest_all_neg_1, tolerance)
        singletons = d_to_nearest_all >= cutoff
        non_singletons = d_to_nearest_all < cutoff

        np.save(path_data+"dis_%s_a-free.npy"%sample,dis)
        np.save(path_data+"dis_neg_%s_a-free.npy"%sample, dis_cp)

        title = sample
        x_label = "Normalized Cosine distance"
        figname = path_data + "fig/d-to-nearest-align-free.png"
        annotext = "tolerance = 0.03%"
        binwidth = 0.01
        # __ smoothing __
        cov = 1.5
        freq = np.histogram(d_to_nearest_all, bins=30)
        freq_pos = [f if f > 0 else f + 1 for f in freq[0]]  # mask zero
        smoothed_freq0 = gaussian_filter1d(np.log(freq_pos), cov)
        smoothed_curve = np.array([freq[1][:-1], np.exp(smoothed_freq0)])
        plt.rcParams.update({'font.size': 16})
        print("Plot the singleton identified via threshold detection.")
        plot_nearest_dis_style2(d_to_nearest_all, singletons, non_singletons, d_to_nearest_all_neg_1,
                                title, x_label, figname, binwidth=binwidth, cutoff=cutoff, annotext=annotext,
                                smoothed_curve=smoothed_curve)

    elif metric == 1 and style == 2:
        print('Calculate pairwise distance according to v-j-junction based distance.')
        dis, _ = v_j_junction_dis(df, field_group='JUNCTION', field_dis='JUNCTION',
                                  groupby=['V-GENE', 'J-GENE'], metric=Normalized_Levenshtein_dist,
                                  diag=1., default_dist=1.)
        d_to_nearest_all = dist_to_nearest(dis)
        dis_neg = cal_group_pdist(df_unique, fields='JUNCTION', groupby=['V-GENE', 'J-GENE'], df2=df_neg,
                                  metric=Normalized_Levenshtein_dist,
                                  diag=1., default_dist=1.)
        d_to_nearest_all_neg = dist_to_nearest(dis_neg)

        # __ smoothing __
        cov = 1.5
        freq = np.histogram(d_to_nearest_all[d_to_nearest_all < 1], bins=30)
        freq_pos = [f if f > 0 else f + 1 for f in freq[0]]  # mask zero
        smoothed_freq0 = gaussian_filter1d(np.log(freq_pos), cov)
        smoothed_curve = np.array([freq[1][:-1], np.exp(smoothed_freq0)])

        # __ detect local minimum __
        min_ind = (np.diff(np.sign(np.diff(smoothed_freq0))) > 0).nonzero()[0] + 1
        loc_min = freq[1][min_ind][0]

        title = sample
        x_label = "Normalized Levenshtein distance"
        figname = path_data + "fig/d-to-nearest-v-j-junction_gt.png"

        singletons = pd.read_csv(path_data + "datasets/singletons_%s_truth.csv"%sample)["clone size"].values
        non_singletons = np.logical_not(singletons)

        binwidth = 0.02
        plt.rcParams.update({'font.size': 16})
        plot_nearest_dis_style2(d_to_nearest_all[d_to_nearest_all < 1], singletons[d_to_nearest_all < 1],
                                non_singletons[d_to_nearest_all < 1],
                                d_to_nearest_all_neg[d_to_nearest_all_neg < 1],
                                title, x_label, figname, binwidth=binwidth, cutoff=loc_min, annotext=None,
                                smoothed_curve=smoothed_curve)

        np.save(path_data + "dis_%s_v-j-junction.npy"%sample, dis)
        np.save(path_data + "dis_neg_%s_v-j-junction.npy"%sample, dis_neg)
    elif metric == 2 and style == 2:
        print("Plot the distance to nearest neighbor distribution with Cosine distance.")
        tolerance = 0.0003
        # dis_cp, rep1, rep2 = cal_numeric_represent_pdis(df_unique, fields='Sequence', transform_func=tf_idf_BCR,
        #                                                 metric=Cosine_dist, diag=1., df2=df_neg, k=4, l=130)

        # dis = align_free_dis(df, df_unique, rep1, 'JUNCTION', metric=Cosine_dist, diag=1.)

        dis = np.load(path_data + "dis_%s_a-free.npy"%sample)
        dis_cp = np.load(path_data + "dis_neg_%s_a-free.npy"%sample)
        d_to_nearest_all = dist_to_nearest(dis)
        d_to_nearest_all_neg = dist_to_nearest(dis_cp)

        # d==1: excluded
        d_to_nearest_all_neg_1 = d_to_nearest_all_neg[d_to_nearest_all_neg < 1]
        # __ detect threshold by negation __
        cutoff = negation_based_cutoff(d_to_nearest_all_neg_1, tolerance)
        singletons = pd.read_csv(path_data + "datasets/singletons_%s_truth.csv"%sample)["clone size"].values
        non_singletons = np.logical_not(singletons)

        title = sample
        x_label = "Normalized Cosine distance"
        figname = path_data + "fig/d-to-nearest-align-free_gt.png"
        annotext = "tolerance = 0.03%"
        binwidth = 0.01
        # __ smoothing __
        cov = 1.5
        freq = np.histogram(d_to_nearest_all, bins=30)
        freq_pos = [f if f > 0 else f + 1 for f in freq[0]]  # mask zero
        smoothed_freq0 = gaussian_filter1d(np.log(freq_pos), cov)
        smoothed_curve = np.array([freq[1][:-1], np.exp(smoothed_freq0)])
        plt.rcParams.update({'font.size': 16})
        print("Plot the singleton identified via threshold detection.")
        plot_nearest_dis_style2(d_to_nearest_all, singletons, non_singletons, d_to_nearest_all_neg_1,
                                title, x_label, figname, binwidth=binwidth, cutoff=cutoff, annotext=annotext,
                                smoothed_curve=smoothed_curve)

if __name__ == '__main__':
    path_data = '/home/siyuan/thesis/Data/new_data/rerun/'
    samples = ["sample%s"%n for n in
               [76, 77, 78, 79, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]

    compare_clustering_real(path_data, samples)
    singleton_distribution_real(path_data)

    path_data = '/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/'
    compare_clustering_simulated(path_data)

    sample = "sampleMS2_11"
    clone_file = "datasets/clone_dataMS2_vv11.csv"
    get_true_singleton(path_data,clone_file,sample)

    metric=1
    style=2
    singleton_distribution_sim(path_data, sample, metric, style)
    metric = 2
    style = 1
    singleton_distribution_sim(path_data, sample, metric, style)
    metric = 2
    style = 2
    singleton_distribution_sim(path_data, sample, metric, style)