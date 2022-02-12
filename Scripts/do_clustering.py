from Clonotyping import *
from SeqRepresent import *
from ReadIgBlast import *
from DetectThreshold import *

def cat_sequence_real():
    # save sequences to fasta files, and run igblastn
    path_data = '/home/siyuan/thesis/Data/new_data/rerun/datasets/'
    real_samples = ["sample%s" % n for n in
                    [76, 77, 78, 79, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]
    for sample in real_samples:
        print(sample)
        summary_file = path_data + sample + '/1_Summary.txt'
        seqfile = path_data + sample + '.fasta'

        df = pd.read_csv(summary_file,sep='\t')
        l,_ = df.shape
        df = df.dropna(subset=['Sequence'])
        l2,_ = df.shape
        print("Drop %s lines with NaN."%(l-l2))
        sequences = list(df.loc[:,'Sequence'].values)

        ofile = open(seqfile, "w")
        for i in range(len(sequences)):
            ofile.write(">%s"%i + "\n" + sequences[i] + "\n")
        ofile.close()

def clean_real():
    # clean the data
    path_data = '/home/siyuan/thesis/Data/new_data/rerun/datasets/'
    real_samples = ["sample%s" % n for n in
                    [76, 77, 78, 79, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]
    for sample in real_samples:
        infile = path_data + sample + '_IgBlast.txt'
        outfile = path_data + sample +"_Nt_info.csv"
        print("========== %s ==========="%sample)
        df_n = pd.read_csv(infile, sep='\t')
        df_nes = read_data(df_n)
        df_nes.to_csv(outfile, sep='\t', index=False)


def combine_all():
    path_data = '/home/siyuan/thesis/Data/new_data/rerun/'
    outfile = path_data + 'datasets/' + "sample76-99_Nt_info.csv"
    samples = ["sample%s" % n for n in
                    [76, 77, 78, 79, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]
    df_all = pd.DataFrame()
    for sample in samples:
        df = pd.read_csv(path_data + 'datasets/' + sample + "_Nt_info.csv", sep='\t')
        df['sample'] = sample
        df_all = pd.concat((df_all, df))
    df_all.to_csv(outfile, sep='\t', index=False)

def detect_threshold_real(tolerance):
    path_data = '/home/siyuan/thesis/Data/new_data/rerun/'
    outfile = path_data + 'datasets/' + "sample76-99_Nt_info.csv"
    df_all = pd.read_csv(outfile, sep='\t')
    df_all_unique = df_all.drop_duplicates(subset=['JUNCTION','sample'], ignore_index=True)

    print('Calculate pairwise distance according to v-j-junction based distance.')
    dis, _ = v_j_junction_dis(df_all, field_group=['JUNCTION','sample'], field_dis='JUNCTION',
                                        groupby=['V-GENE', 'J-GENE'], metric=Normalized_Levenshtein_dist,
                                        diag=1., default_dist=1.)
    d_to_nearest_all = dist_to_nearest(dis)
    d_to_nearest_all = d_to_nearest_all[d_to_nearest_all < 1]
    np.save(path_data+'distance/dis_sample76-99_v-j-junction.npy',dis)
    print('Detect the first threshold.')
    d_threshold1 = outlier_based_cutoff(d_to_nearest_all, visualize=True,
                                        x_label="Normalized Levenshtein distance to nearest neighbor",
                                        figname=path_data + 'fig/detect_threshold1_real.png')

    # clean the negation sequence
    print('Clean the negation sequence.')
    df_n = pd.read_csv(path_data + "datasets/negative_table_IgBlast.txt", sep='\t')
    df_neg = read_data(df_n)
    df_neg.to_csv(path_data + "datasets/negative_table_Nt_info.csv", sep='\t', index=False)

    print('Calculate distance between sample sequence and negation sequence using v-j-junction method.')
    df_neg = pd.read_csv(path_data + "datasets/negative_table_Nt_info.csv", sep='\t').drop_duplicates(
        subset=['JUNCTION'], ignore_index=True)

    dis_cp = cal_group_pdist(df_all_unique, fields='JUNCTION', groupby=['V-GENE','J-GENE'],
                             df2=df_neg, metric=Normalized_Levenshtein_dist, diag=1., default_dist=1.)
    np.save(path_data + 'distance/dis_negation_v-j-junction.npy', dis_cp)

    print("Calculate distance between sample sequence and negation sequence using align-free method.")
    df_neg = pd.read_csv(path_data + "datasets/negative_table_Nt_info.csv", sep='\t').drop_duplicates(subset=['JUNCTION'], ignore_index=True)
    df_all_unique = df_all.drop_duplicates(subset=['JUNCTION','sample'], ignore_index=True)
    dis_cp, rep1, rep2 = cal_numeric_represent_pdis(df_all_unique, fields='Sequence', transform_func=tf_idf_BCR,
                               metric=Cosine_dist, diag=1., df2=df_neg, k=4, l=130)
    d_to_nearest_cp = dist_to_nearest(dis_cp)
    np.save(path_data + 'distance/dis_negation_align-free.npy', dis_cp)
    rep1.to_csv(path_data + 'distance/tf-idf_sample76-99_unique.csv', sep='\t',index=False)
    rep2.to_csv(path_data + 'distance/tf-idf_negation.csv', sep='\t',index=False)
    df_all_unique.loc[:,'sample'].to_csv(path_data + 'distance/sample76-99_unique_sample.csv', sep='\t',index=False)
    print('Detect the second threshold.')
    d_threshold2 = negation_based_cutoff(d_to_nearest_cp, tolerance,visualize=True,
                                         figname=path_data+'fig/detect_threshold2_real.png')

    print('Calculate pairwise distance according to align-free distance.')
    dis = cal_numeric_pdist(rep1, metric=Cosine_dist, diag=1.)
    np.save(path_data + 'distance/dis_sample76-99_align-free.npy', dis)
    return d_threshold1, d_threshold2


def clustering_real(d_threshold1, d_threshold2):
    path_data = '/home/siyuan/thesis/Data/new_data/rerun/'
    real_samples = ["sample%s" % n for n in
                    [76, 77, 78, 79, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]
    outfile = path_data + 'datasets/' + "sample76-99_Nt_info.csv"
    df_all = pd.read_csv(outfile, sep='\t')
    df_all_unique = df_all.drop_duplicates(subset=['JUNCTION','sample'], ignore_index=True)
    df_rep_all = pd.read_csv(path_data + 'distance/tf-idf_sample76-99_unique.csv', sep='\t')

    for sample in real_samples:
        infile = path_data+'datasets/'+sample+'_Nt_info.csv'
        df = pd.read_csv(infile, sep='\t')
        print("=======%s======="%sample)
        print("All sequence:", df.shape)

        print(">>> junction based method.")
        clusters = cluster_by_sequence(df.loc[:,'JUNCTION'].values)
        np.save(path_data + "clustering/clusters_%s_junction.npy" % sample, clusters)
        del clusters

        print(">>> V-J-junction based method.")
        dis, representor = v_j_junction_dis(df, field_group='JUNCTION', field_dis='JUNCTION',
                                            groupby=['V-GENE', 'J-GENE'], metric=Normalized_Levenshtein_dist,
                                            diag=1., default_dist=1.)

        clusters = HAC_cluster_by_distance_fields(representor, dis, d_threshold1)
        np.save(path_data + "clustering/clusters_%s_V-J-junction.npy" % sample, clusters)
        np.save(path_data + "clustering/dis_%s_V-J-junction.npy" % sample, dis)
        np.save(path_data + "clustering/representor_%s_V-J-junction.npy" % sample, representor)
        del dis, representor, clusters

        print(">>> align-free method.")
        sample_index = df_all_unique['sample'] == sample
        df_unique = df_all_unique[sample_index]
        df_rep = df_rep_all[sample_index]
        dis, representor = align_free_dis(df, df_unique, df_rep, field_group='JUNCTION', metric=Cosine_dist, diag=1.)
        # if plot the clustering results:
        # clusters = HAC_cluster_by_distance_fields(representor, dis, d_threshold2,**{"fig1":fig1,"fig2":fig2,"fig3":fig3})
        clusters = HAC_cluster_by_distance_fields(representor, dis, d_threshold2)
        np.save(path_data + "clustering/clusters_%s_a-free.npy" % sample, clusters)
        np.save(path_data + "clustering/dis_%s_a-free.npy" % sample, dis)
        np.save(path_data + "clustering/representor_%s_a-free.npy" % sample, representor)
        del dis, representor, clusters, df
    return

def cat_sequence_simulated():
    samples_2 = ["sampleMS2_%s_db-pass.tab" % (m + 1) for m in range(20)]
    samples_3 = ["sampleMS3_o%s_germ-pass.tab" % (m + 1) for m in range(26)]
    samples_4 = ["sampleMS4_o%s_germ-pass.tab" % (m + 1) for m in range(29)]
    path_data = '/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/datasets/'

    for sample in samples_2 + samples_3 + samples_4:
        print(sample)
        summary_file = path_data + sample
        split = re.split(r'(\d+)', sample)
        subject_id = split[1]
        sample_id = split[3]
        seqfile = path_data + subject_id + '_' + sample_id + '.fasta'

        df = pd.read_csv(summary_file, sep='\t')
        l, _ = df.shape
        df = df.dropna(subset=['SEQUENCE_INPUT'])
        l2, _ = df.shape
        print("Drop %s lines with NaN." % (l - l2))
        sequences = list(df.loc[:, 'SEQUENCE_INPUT'].values)

        ofile = open(seqfile, "w")
        for i in range(len(sequences)):
            ofile.write(">%s" % (i+1) + "\n" + sequences[i] + "\n")
        ofile.close()

def clean_simulated():
    samples_2 = ["sampleMS2_%s_db-pass.tab" % (m + 1) for m in range(20)]
    samples_3 = ["sampleMS3_o%s_germ-pass.tab" % (m + 1) for m in range(26)]
    samples_4 = ["sampleMS4_o%s_germ-pass.tab" % (m + 1) for m in range(29)]
    path_data = '/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/datasets/'
    for sample in samples_2 + samples_3 + samples_4:
        split = re.split(r'(\d+)', sample)
        subject_id = split[1]
        sample_id = split[3]
        infile = path_data + subject_id + '_' + sample_id + '_IgBlast.txt'
        outfile = path_data + "sampleMS%s_%s" % (subject_id, sample_id) + "_Nt_info.csv"
        print("========== %s ===========" % sample)
        df_n = pd.read_csv(infile, sep='\t')
        df_nes = read_data(df_n)
        df_nes.to_csv(outfile, sep='\t', index=False)

def detect_threshold_simulated(tolerance):
    path_data = '/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/'
    samples_2 = ["sampleMS2_%s_db-pass.tab" % (m + 1) for m in range(20)]

    # detect the two thresholds for each sample, and take the average
    threshold1_lst = []
    threshold2_lst = []
    df_neg = pd.read_csv("/home/siyuan/thesis/Data/new_data/rerun/datasets/negative_table_Nt_info.csv",
                         sep='\t').drop_duplicates(
        subset=['JUNCTION'], ignore_index=True)

    for sample in samples_2:
        print("=============%s=============="%sample)
        split = re.split(r'(\d+)', sample)
        subject_id = split[1]
        sample_id = split[3]
        outfile = path_data + 'datasets/' + "sampleMS%s_%s" % (subject_id, sample_id) + "_Nt_info.csv"
        df = pd.read_csv(outfile, sep='\t')

        print('Calculate pairwise distance according to v-j-junction based distance.')
        dis, _ = v_j_junction_dis(df, field_group='JUNCTION', field_dis='JUNCTION',
                                            groupby=['V-GENE', 'J-GENE'], metric=Normalized_Levenshtein_dist,
                                            diag=1., default_dist=1.)
        d_to_nearest_all = dist_to_nearest(dis)
        d_to_nearest_all = d_to_nearest_all[d_to_nearest_all < 1]
        print('Detect the first threshold.')
        d_threshold1 = outlier_based_cutoff(d_to_nearest_all, visualize=True,
                                            x_label="Normalized Levenshtein distance to nearest neighbor",
                                            figname=path_data + 'fig/detect_threshold1_sim.png')
        threshold1_lst.append(d_threshold1)


        print("Calculate distance between sample sequence and negation sequence using align-free method.")
        df_unique = df.drop_duplicates(subset=['JUNCTION'], ignore_index=True)
        dis_cp, rep1, rep2 = cal_numeric_represent_pdis(df_unique, fields='Sequence', transform_func=tf_idf_BCR,
                                                        metric=Cosine_dist, diag=1., df2=df_neg, k=4, l=130)
        d_to_nearest_cp = dist_to_nearest(dis_cp)
        print('Detect the second threshold.')
        d_threshold2 = negation_based_cutoff(d_to_nearest_cp, tolerance, visualize=True,
                                             figname=path_data + 'fig/detect_threshold2_sim.png')
        threshold2_lst.append(d_threshold2)

    return np.mean(threshold1_lst), np.mean(threshold2_lst)


def clustering_simulated(d_threshold1,d_threshold2, samples_2):
    path_data = '/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/'
    df_neg = pd.read_csv("/home/siyuan/thesis/Data/new_data/rerun/datasets/negative_table_Nt_info.csv",
                         sep='\t').drop_duplicates(subset=['JUNCTION'], ignore_index=True)

    for sample in samples_2:
        infile = path_data + 'datasets/' + sample + '_Nt_info.csv'
        df = pd.read_csv(infile, sep='\t')
        print("=======%s=======" % sample)
        print("All sequence:", df.shape)

        print(">>> junction based method.")
        clusters = cluster_by_sequence(df.loc[:, 'JUNCTION'].values)
        np.save(path_data + "clustering/clusters_%s_junction.npy" % sample, clusters)
        del clusters

        print(">>> V-J-junction based method.")
        dis, representor = v_j_junction_dis(df, field_group='JUNCTION', field_dis='JUNCTION',
                                            groupby=['V-GENE', 'J-GENE'], metric=Normalized_Levenshtein_dist,
                                            diag=1., default_dist=1.)

        clusters = HAC_cluster_by_distance_fields(representor, dis, d_threshold1)
        np.save(path_data + "clustering/clusters_%s_V-J-junction.npy" % sample, clusters)
        np.save(path_data + "clustering/dis_%s_V-J-junction.npy" % sample, dis)
        np.save(path_data + "clustering/representor_%s_V-J-junction.npy" % sample, representor)
        del dis, representor, clusters

        print(">>> align-free method.")
        df_unique = df.drop_duplicates(subset=['JUNCTION'], ignore_index=True)
        _, rep1, _ = cal_numeric_represent_pdis(df_unique, fields='Sequence', transform_func=tf_idf_BCR,
                                                        metric=Cosine_dist, diag=1., df2=df_neg, k=4, l=130)
        dis, representor = align_free_dis(df, df_unique, rep1, field_group='JUNCTION', metric=Cosine_dist, diag=1.)
        # if plot the clustering results:
        # clusters = HAC_cluster_by_distance_fields(representor, dis, d_threshold2,**{"fig1":fig1,"fig2":fig2,"fig3":fig3})
        clusters = HAC_cluster_by_distance_fields(representor, dis, d_threshold2)
        np.save(path_data + "clustering/clusters_%s_a-free.npy" % sample, clusters)
        np.save(path_data + "clustering/dis_%s_a-free.npy" % sample, dis)
        np.save(path_data + "clustering/representor_%s_a-free.npy" % sample, representor)
        del dis, representor, clusters, df
    return

if __name__ == '__main__':
    print('Do clustering for each sample.')
    cat_sequence_real()
    clean_real()
    combine_all()
    d_threshold1, d_threshold2 = detect_threshold_real(0.0003)
    print(d_threshold1)
    print(d_threshold2)
    # d_threshold1 = 0.1111111
    # d_threshold2 = 0.1508548650741577
    clustering_real(d_threshold1,d_threshold2)

    cat_sequence_simulated()
    clean_simulated()
    d_threshold1, d_threshold2 = detect_threshold_simulated(0.0003)
    print(d_threshold1)
    print(d_threshold2)
    # d_threshold1 = 0.1003991999717159
    # d_threshold2 = 0.1836092398145795
    samples_2 = ["sampleMS2_%s" % (m + 1) for m in range(20)]
    clustering_simulated(d_threshold1, d_threshold2,samples_2)

    # run IgBlast in bash:
    # """
    # bin/igblastn - germline_db_V
    # database / IGHV_shortname.fasta - germline_db_J
    # database / IGHJ_shortname.fasta - germline_db_D
    # database / IGHD_shortname.fasta - organism
    # human - query /home/siyuan/thesis/Data/new_data/rerun/datasets/sample90.fasta - auxiliary_data
    # optional_file/human_gl.aux - show_translation - outfmt 19 > /home/siyuan/thesis/Data/new_data/rerun/datasets/sample90_IgBlast.txt
    # """

