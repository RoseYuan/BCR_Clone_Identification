import pandas as pd

from Utils import *
from HAC_clustering import *
from libBiodiversity import *
from Clonotyping import *
import argparse
# script, calculate the diversity profiles after subsampling
# visualize diversity profiles
# quantify the sensitivity of diversity profiles
# visualize

def align_free_dis_repertoire(samples, df, df_neg, field_group, field_dis, threshold, transform_func=tf_idf_BCR,
                               metric=Cosine_dist,diag=1.,k=4,l=130):
    '''
    Do alignment-free clonotyping for a whole repertoire (should learn the tf-idf representation
    of a whole repertoire instead of an individual sample).
    '''
    df_all = df.sample(frac=1).reset_index(drop=True)
    df_all_unique = df_all.drop_duplicates(subset=field_group, ignore_index=True)
    dis_cp, df_rep_all, rep2 = cal_numeric_represent_pdis(df_all_unique, field_dis, transform_func=transform_func,
                                                    metric=metric, diag=diag, df2=df_neg, k=k, l=l)
    print(">>> align-free method.")
    clusters_repertoire = []
    for sample in samples:
        sample_index = df_all_unique['sample'] == sample
        df_unique = df_all_unique[sample_index]
        df_rep = df_rep_all[sample_index]
        df = df_all[df_all['sample'] == sample]
        dis, representor = align_free_dis(df, df_unique, df_rep, field_group='JUNCTION', metric=Cosine_dist, diag=1.)
        clusters = HAC_cluster_by_distance_fields(representor, dis, threshold)
        clusters_repertoire.append(clusters)
    return clusters_repertoire


def clustering(df_nt, method, threshold=None):
    df = df_nt.sample(frac=1).reset_index(drop=True) # shuffle
    if method == 'junction-based':
        clusters = cluster_by_sequence(df.loc[:,'JUNCTION'].values)
    elif method == 'V-J-junction-based':
        dis, representor = v_j_junction_dis(df, field_group='JUNCTION', field_dis='JUNCTION',
                                            groupby=['V-GENE', 'J-GENE'], metric=Normalized_Levenshtein_dist,
                                            diag=1., default_dist=1.)
        clusters = HAC_cluster_by_distance_fields(representor, dis, threshold)
    else:
        raise ValueError('Unknown clonotyping method.')
    print('Clustering results: ', len(clusters))
    return clusters

def subsample_clustering_afree(frac, redo, threshold, samples, path_data):
    outfile = path_data + 'datasets/' + "sample76-99_Nt_info.csv"
    df_all = pd.read_csv(outfile, sep='\t')
    df_neg = pd.read_csv(path_data + "datasets/negative_table_Nt_info.csv", sep='\t').drop_duplicates(
        subset=['JUNCTION'], ignore_index=True)
    df_sub = df_all.sample(frac=frac, replace=False)
    df_sub = df_sub.reset_index()
    if redo:
        clusters_repo = align_free_dis_repertoire(samples, df_sub, df_neg, ['JUNCTION','sample'], 'Sequence', threshold, transform_func=tf_idf_BCR,
                              metric=Cosine_dist, diag=1., k=4, l=130)
        for i,sample in enumerate(samples):
            clusters = clusters_repo[i]
            np.save(path_data + "subsampling/clusters_%s_a-free_%s_redoT.npy" % (sample, frac), clusters)
    else:
        for sample in samples:
            nt_file = path_data + 'datasets/' + "%s_Nt_info.csv" % sample
            df_nt = pd.read_csv(nt_file, sep='\t')
            print("BCR sequences:", df_nt.shape)
            df_sub = df_nt.sample(frac=frac, replace=False)
            print("BCR sequences subset:", df_sub.shape)
            index_sub = df_sub.index
            clusters = np.load(path_data + "clustering/clusters_%s_a-free.npy" % sample)
            clusters = clusters[index_sub]
            np.save(path_data + "subsampling/clusters_%s_a-free_%s_redoF.npy" %(sample,frac),clusters)

def subsample_clustering(frac, redo, threshold, samples, path_data):
    for sample in samples:
        nt_file = path_data + 'datasets/' + "%s_Nt_info.csv"%sample
        df_nt = pd.read_csv(nt_file, sep='\t')
        print("BCR sequences:", df_nt.shape)
        df_sub = df_nt.sample(frac=frac, replace=False)
        print("BCR sequences subset:", df_sub.shape)
        index_sub = df_sub.index
        df_sub = df_sub.reset_index()

        if redo:
            # do clustering for subset BCRs:
            c1 = clustering(df_sub, method='junction-based')
            c2 = clustering(df_sub, method='V-J-junction-based', threshold=threshold)
            np.save(path_data + "subsampling/clusters_%s_junction_%s_redoT.npy" % (sample, frac), c1)
            np.save(path_data + "subsampling/clusters_%s_V-J-junction_%s_redoT.npy" % (sample, frac), c2)
        else:
            clusters = np.load(path_data + "clustering/clusters_%s_V-J-junction.npy" % sample)
            clusters = clusters[index_sub]
            np.save(path_data + "subsampling/clusters_%s_V-J-junction_%s_redoF.npy" % (sample, frac), clusters)
            clusters = np.load(path_data + "clustering/clusters_%s_junction.npy" % sample)
            clusters = clusters[index_sub]
            np.save(path_data + "subsampling/clusters_%s_junction_%s_redoF.npy" % (sample, frac), clusters)

    return


def index_in_diversity_profile(c,a):
    if a == 'richness_Chao':
        return richness_chao(c)
    elif a == 'shannon_entropy_Chao':
        return Shannon_entropy_Chao(c)
    else:
        return Hill_diversity(c, a)


def cal_diversity_profile(alphas, clustering):
    profile = {}
    c = {}
    unique, counts = np.unique(clustering, return_counts=True)
    for i, cluster in enumerate(unique):
        c[cluster] = counts[i]
    for a in alphas:
        profile['alpha=%s' % a] = index_in_diversity_profile(c, a)
    return profile


def subsample_profile(alphas, fracs, samples):
    df_div = pd.DataFrame(
        columns=['alpha=%s' % i for i in alphas] + ['subsampling fraction', 'clustering method', 're_clustering'])

    for sample in samples:
        for i, frac in enumerate(fracs):
            c1 = np.load(path_data + "subsampling/clusters_%s_junction_%s_redoF.npy" % (sample, frac))
            profile = cal_diversity_profile(alphas, c1)
            profile['clustering method'] = 'junction-based'
            profile['re_clustering'] = False
            profile['subsampling fraction'] = frac
            profile['sample'] = sample
            df_div = df_div.append(profile, ignore_index=True)

            c1 = np.load(path_data + "subsampling/clusters_%s_junction_%s_redoT.npy" % (sample, frac))
            profile = cal_diversity_profile(alphas, c1)
            profile['clustering method'] = 'junction-based'
            profile['re_clustering'] = True
            profile['subsampling fraction'] = frac
            profile['sample'] = sample
            df_div = df_div.append(profile, ignore_index=True)

            c2 = np.load(path_data + "subsampling/clusters_%s_V-J-junction_%s_redoF.npy" % (sample, frac))
            profile = cal_diversity_profile(alphas, c2)
            profile['clustering method'] = 'V-J-junction-based'
            profile['re_clustering'] = False
            profile['subsampling fraction'] = frac
            profile['sample'] = sample
            df_div = df_div.append(profile, ignore_index=True)

            c2 = np.load(path_data + "subsampling/clusters_%s_V-J-junction_%s_redoT.npy" % (sample, frac))
            profile = cal_diversity_profile(alphas, c2)
            profile['clustering method'] = 'V-J-junction-based'
            profile['re_clustering'] = True
            profile['subsampling fraction'] = frac
            profile['sample'] = sample
            df_div = df_div.append(profile, ignore_index=True)

            c3 = np.load(path_data + "subsampling/clusters_%s_a-free_%s_redoF.npy" % (sample, frac))
            profile = cal_diversity_profile(alphas, c3)
            profile['clustering method'] = 'align-free'
            profile['re_clustering'] = False
            profile['subsampling fraction'] = frac
            profile['sample'] = sample
            df_div = df_div.append(profile, ignore_index=True)

            c3 = np.load(path_data + "subsampling/clusters_%s_a-free_%s_redoT.npy" % (sample, frac))
            profile = cal_diversity_profile(alphas, c3)
            profile['clustering method'] = 'align-free'
            profile['re_clustering'] = True
            profile['subsampling fraction'] = frac
            profile['sample'] = sample
            df_div = df_div.append(profile, ignore_index=True)
    return df_div


def calculate_subsample_profile(alphas, threshold1, threshold2, n_repeat, fracs, profile_file):
    print('No. 1 subsampling experiments.')
    for frac in fracs:
        print('Subsample %s BCR sequence.'%frac)
        redo = False
        subsample_clustering_afree(frac, redo, threshold1, samples, path_data)
        subsample_clustering(frac, redo, threshold2, samples, path_data)
        redo = True
        subsample_clustering_afree(frac, redo, threshold1, samples, path_data)
        subsample_clustering(frac, redo, threshold2, samples, path_data)

    df_div = subsample_profile(alphas, fracs, samples)
    df_div.to_csv(path_data+'subsampling/%s'%profile_file, mode='a', header=True)

    for i in range(n_repeat-1):
        print('No. %s subsampling experiments.'%(i+2))
        for frac in fracs:
            print('Subsample %s BCR sequence.' % frac)
            redo = False
            subsample_clustering_afree(frac, redo, threshold1, samples, path_data)
            subsample_clustering(frac, redo, threshold2, samples, path_data)
            redo = True
            subsample_clustering_afree(frac, redo, threshold1, samples, path_data)
            subsample_clustering(frac, redo, threshold2, samples, path_data)
        df_div = subsample_profile(alphas, fracs, samples)
        df_div.to_csv(path_data+'subsampling/%s'%profile_file, mode='a', header=False)

def visualize_subsample_profile(sample, method, redo, fracs, div_file):
    df_div = pd.read_csv(div_file,sep=',',index_col=0)
    show_alpha = np.append(np.arange(0, 11, 1) / 10, np.array([2, 4, 6, 8, 10, 100, np.inf]))
    alpha_axis = []
    for i in show_alpha:
        if i > 1 and i < 101:
            alpha_axis.append(int(i))
        else:
            alpha_axis.append(i)
    initial_axis = rev_transform(alpha_axis)

    a = np.arange(-100, 101, 1) / 100
    y2 = transform_func(a)
    x = np.append(a[y2 <= 150], 1)
    # x = a[y2 <= 150]

    plt.figure(figsize=(20, 10))
    colors = ['tab:blue', 'tab:red', 'tab:orange', "tab:green", "tab:pink", "tab:grey", "tab:purple", 'gold', 'dark']
    for i, frac in enumerate(fracs):
        df_check = df_div.query(
            '`subsampling fraction` == %s and re_clustering == %s and `clustering method` == "%s" and sample == "%s"' % (
            frac, redo, method,sample))
        y_all = df_check.iloc[:, 0:-4].values
        mean_y = np.mean(y_all, axis=0)
        std_y = np.std(y_all, axis=0)
        mean_y = np.append(mean_y, mean_y[-1])
        std_y = np.append(std_y, std_y[-1])
        plt.plot(x, mean_y, '-', label="fraction = %s" % (frac), color=colors[i])
        plt.fill_between(x, mean_y - std_y, mean_y + std_y, alpha=0.2)

    plt.legend(fontsize=16)
    plt.xticks(initial_axis, alpha_axis, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Alpha in Hill's diversity index", fontsize=16)
    plt.ylabel("Hill's diversity index", fontsize=16)

    plt.title('Diversity profiles of %s based on %s method' % (sample, method), fontsize=16)
    plt.yscale('log')
    plt.rcParams.update({'font.size': 16})
    plt.savefig(path_data + "fig/subsampling_profiles_%s_%s_redo%s.png" % (sample, method, redo))

def cal_sensitivity(path_data,method,redo,samples):
    df_div = pd.read_csv(path_data + "subsampling/diversity_profile_all.csv", sep=',',index_col=0)
    df_chao = pd.read_csv(path_data + "subsampling/diversity_profile_chao_all.csv", sep=',',index_col=0)

    df_div = df_div.rename(columns={'subsampling fraction': 'subsampling_fraction','clustering method':'clustering_method'})
    df_chao = df_chao.rename(columns={'subsampling fraction': 'subsampling_fraction','clustering method':'clustering_method'})
    sensitivity = pd.DataFrame(columns=df_div.columns[:-4])
    sensitivity_chao = pd.DataFrame(columns=df_chao.columns[:-4])

    for sample in samples:
        # average over # subsampling experiments
        s1 = df_div.query('subsampling_fraction == 1 and re_clustering == %s and clustering_method == "%s" and sample == "%s" ' % (redo, method, sample)).mean()[:-2]
        s2 = df_div.query('subsampling_fraction == 0.1 and re_clustering == %s and clustering_method == "%s" and sample == "%s" ' % (redo, method, sample)).mean()[:-2]
        sens = np.log(s1) - np.log(s2)
        sensitivity = sensitivity.append(sens, ignore_index=True)


        s10 = df_chao.query('subsampling_fraction == 1 and re_clustering == %s and clustering_method == "%s" and sample == "%s" ' % (redo, method, sample)).mean()[:-2]
        s20 = df_chao.query('subsampling_fraction == 0.1 and re_clustering == %s and clustering_method == "%s" and sample == "%s" ' % ( redo, method, sample)).mean()[:-2]
        sens0 = np.log(s10) - np.log(s20)
        sensitivity_chao = sensitivity_chao.append(sens0, ignore_index=True)

    # average over samples

    sensitivity_mean = sensitivity.mean().values
    sensitivity_mean = np.append(sensitivity_mean, sensitivity_mean[-1])
    sensitivity_chao_mean = sensitivity_chao.mean().values

    # sensitivity_mean = sensitivity_mean[:-2]

    a = np.arange(-100, 101, 1) / 100
    y2 = transform_func(a)

    show_alpha = np.append(np.arange(0, 11, 1) / 10, np.array([2, 4, 6, 8, 10, 100, np.inf]))
    alpha_axis = []
    for i in show_alpha:
        if i > 1 and i < 101:
            alpha_axis.append(int(i))
        else:
            alpha_axis.append(i)


    initial_axis = rev_transform(alpha_axis)

    plt.figure(figsize=(20, 10))

    x=np.append(a[y2 <= 150], 1)
    # x=a[y2 <= 150]

    plt.plot(x,sensitivity_mean)  # ,label=labels[sampleID[s]-2] if s in [19,26,73] else '',color=colors[sampleID[s]-2],alpha=0.7)
    mean_y = sensitivity_mean
    # mean_y = np.append(mean_y, mean_y[-1])
    std_y = sensitivity.std().values
    std_y = np.append(std_y, std_y[-1])
    plt.fill_between(x, mean_y - std_y, mean_y + std_y, alpha=0.2)


    plt.plot(rev_transform(0), sensitivity_chao_mean[0], color='red', marker='.', markersize=15)
    plt.plot(rev_transform(1), sensitivity_chao_mean[1], color='red', marker='.', markersize=15)
    plt.text(rev_transform(0) - 0.08, sensitivity_chao_mean[0] - 0.03, 'Richness Chao', fontsize=16)
    plt.text(rev_transform(1) + 0.03, sensitivity_chao_mean[1] - 0.01, 'Shannon entropy Chao', fontsize=16)
    plt.errorbar(rev_transform(0), sensitivity_chao_mean[0], yerr=sensitivity_chao.std()[0], ecolor="black", capsize=5)
    plt.errorbar(rev_transform(1), sensitivity_chao_mean[1], yerr=sensitivity_chao.std()[1], ecolor="black", capsize=5)
    # plt.legend(fontsize=16)
    plt.xticks(initial_axis, alpha_axis, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Alpha in Hill's diversity index", fontsize=16)
    plt.ylabel("Sensitivity", fontsize=16)
    plt.title('Sensitivity to sequencing depth based on %s method'%method, fontsize=16)
    plt.savefig(path_data+"fig/sensitivity_to_subsampling_%s-redo%s.png"%(method,redo))

if __name__ == '__main__':


    path_data = '/home/siyuan/thesis/Data/new_data/rerun/'
    samples = ["sample%s" % n for n in
                    [76, 77, 78, 79, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]
    fracs = [0.01,0.02,0.05,0.1,0.25,0.50,0.85,1]
    n_repeat = 30
    threshold1 = 0.1111111
    threshold2 = 0.1508548650741577

    # alphas = np.arange(0,10,1)
    alphas = ['richness_Chao','shannon_entropy_Chao']
    parser = argparse.ArgumentParser(description='Plot Clustering result.')
    parser.add_argument('--file', required=True, type=str) # output file name
    args = parser.parse_args()
    profile_file = args.file
    calculate_subsample_profile(alphas, threshold1, threshold2, n_repeat, fracs, profile_file)

    div_file = path_data+'subsampling/diversity_profile_all.csv'
    redo = False
    method = 'align-free'
    sample = 'sample92'
    visualize_subsample_profile(sample, method, redo, fracs, div_file)
    cal_sensitivity(path_data,method, redo, samples)