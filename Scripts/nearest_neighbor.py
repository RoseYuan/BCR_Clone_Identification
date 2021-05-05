from AESA import *


def dist_to_nearest_all_exhaustive(seqs, distance=Normalized_Hamming_dist):
    n_seq = len(seqs)
    if n_seq <= 1:
        raise ValueError("Must have at least two sequences.")
    start_time = time.time()
    dis, counts = dist_pairwise(seqs, distance=distance)
    d_to_nearest_all = dist_to_nearest(dis)
    print("%d sequences, %d calls, use %2f seconds."
          % (n_seq, counts, time.time() - start_time))
    return d_to_nearest_all


def cal_dist_to_nearest_all_exhaustive(df, groupby=None, distance=Normalized_Hamming_dist):
    """
    Calculate the distance to nearest neighbor using exhaustive search,
    and if groupby is specified, only compute the distance when the sequence pair are within the same group, e.g. of the same length.
    :param distance:
    :param groupby:
    :param df:
    :return:
    """
    if groupby is None:
        sequences = df.loc[:, "JUNCTION"].values
        d_to_nearest_all = dist_to_nearest_all_exhaustive(sequences, distance=distance)
    else:
        keys, values = group_seq(df, keys=groupby)
        d_to_nearest_all = np.array([])

        for i, key in enumerate(keys):
            index = list(values)[i]
            sequences = df.loc[index, "JUNCTION"].values
            if len(sequences) > 1:
                print("For group %s = %s: " % (groupby, key))
                d_to_nearest = dist_to_nearest_all_exhaustive(sequences, distance=distance)
                d_to_nearest_all = np.concatenate((d_to_nearest_all, d_to_nearest))
    return d_to_nearest_all


def dist_to_nearest_all_approximate(pilots, seqs, distance=Normalized_Levenshtein_dist):
    d_to_nearest_all = []
    aesa = Aesa(pilots, distance)
    pre_counts = aesa.get_pre_count()
    print('{0} calls during pre-computation'.format(pre_counts))

    for seq in seqs:
        d_to_nearest, counts = aesa.nearest(seq)
        print('{0} calls during nearest neighbour search'.format(counts))
        d_to_nearest_all.append(d_to_nearest)
    return np.concatenate([aesa.get_pilot_d_to_nearest(),np.array(d_to_nearest_all)])


def cal_dist_to_nearest_all_approximate(df, distance=Normalized_Levenshtein_dist):
    keys, values = group_seq(df)
    df_pilots = subsample(df, frac_small=0.99, n=10, groups=values)
    pilots = df_pilots["JUNCTION"].values
    pilot_index = df_pilots.index
    df = df.drop(pilot_index)
    seqs = df["JUNCTION"].values
    print("%d pilots, %d target sequences." % (len(pilots), len(seqs)))
    return dist_to_nearest_all_approximate(pilots, seqs, distance=distance)


def det_cutoff():
    return


if __name__=="__main__":
    path_data = "/Users/lou/Thesis/BCR_Clone_Identification/Data/"
    # outfile = path_data+"Nt_info_expanded.csv"
    # df = pd.read_csv(outfile, sep='\t')
    # df = df.iloc[1:210,:]
    # d = cal_dist_to_nearest_all_approximate(df,distance=Normalized_Hamming_dist)
    # print(d)
    # d = dist_to_nearest_all_exhaustive(df)

    outfile = path_data+"Nt_info.csv"
    df = pd.read_csv(outfile, sep='\t')
    seqs_tf_idf = cal_tf_idf(df.loc[0:4,"V-D-J-REGION"].values,k=2)
    d_to_nearest_all = dist_to_nearest_all_exhaustive(seqs_tf_idf,distance=Cosine_dist)
    print(d_to_nearest_all)