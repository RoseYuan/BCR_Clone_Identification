import pandas as pd
import numpy as np
from libSeq import *
import time


def read_data(nt_file: str, outfile: str, expand=True):
    """
    Get necessary information and save to a file.
    :param nt_file: Nt sequence without gaps
    :param outfile: output file path and name
    :param expand: for sequence with multiple V/J gene annotations: expand rows or not
    :return: None
    """
    df_n = pd.read_csv(nt_file, sep='\t')
    df_n['JUNCTION length'] = df_n['JUNCTION end'] - df_n['JUNCTION start'] + 1
    # sequence with 2 V/J gene annotations: flatten into two rows
    if expand:
        df_n['multiple V-GENE alleles'] = df_n.loc[:, 'V-GENE and allele'].str.split(", or | or ").str.len() >= 2
        df_n['multiple J-GENE alleles'] = df_n.loc[:, 'J-GENE and allele'].str.split(", or | or ").str.len() >= 2
        df_n['V-GENE and allele'] = df_n.loc[:, 'V-GENE and allele'].str.split(", or | or ").values
        df_n['J-GENE and allele'] = df_n.loc[:, 'J-GENE and allele'].str.split(", or | or ").values
        df_n = df_n.explode('V-GENE and allele').reset_index(drop=True)
        df_n = df_n.explode('J-GENE and allele').reset_index(drop=True)

        df_nes = df_n[
            ['Sequence number', 'Sequence ID', 'V-GENE and allele', 'J-GENE and allele', 'V-D-J-REGION', 'JUNCTION',
             'JUNCTION length', 'multiple V-GENE alleles', 'multiple J-GENE alleles']]
    else:
        df_nes = df_n[
            ['Sequence number', 'Sequence ID', 'V-GENE and allele', 'J-GENE and allele', 'V-D-J-REGION', 'JUNCTION',
             'JUNCTION length']]

    df_nes.to_csv(outfile, sep='\t', index=False)


def group_seq(df_nes: pd.DataFrame, keys=('V-GENE and allele', 'J-GENE and allele', 'JUNCTION length')):
    seq_groups = df_nes.groupby(list(keys)).groups
    return seq_groups.keys(), seq_groups.values()


# def retrieve_group(key:tuple,seq_groups:dict,df:pd.DataFrame) -> pd.DataFrame:
#     df_group = pd.DataFrame([])
#     return df_group


def dist_pairwise(seq_list, distance=Normalized_Hamming_dist) -> np.array:
    """
    Calculate pairwise distance within a sequence list.
    If two strings are of different length, d=1
    :param distance: distance metric
    :param seq_list:
    :return:
    """
    l = len(seq_list)
    dis = np.empty(shape=(l, l))
    dis[:] = 1
    for i0, i in enumerate(seq_list):
        for j0, j in enumerate(seq_list):
            if j > i:
                string1 = seq_list[i]
                string2 = seq_list[j]
                d = distance(string1, string2)
                dis[i0, j0] = d
    return dis


def dist_to_nearest(dis: np.array):
    dis_plus = np.copy(dis)
    dis_plus[dis == 0] = np.nan
    d_to_nearest = np.nanmin(dis_plus, axis=0)
    return d_to_nearest


def subsample(df, frac=0.2, random_state=1, groups=None) -> pd.DataFrame:
    """

    :param df:
    :param frac:
    :param random_state:
    :param groups: a list of index, used to stratify df
    :return:
    """
    df_sampled = pd.DataFrame(columns=df.columns)
    if groups is None:
        return df.sample(frac=frac, random_state=random_state)
    else:
        for group in groups:
            df_group_sampled = df.loc[group, :].sample(frac=0.1, random_state=1)
            df_sampled = pd.concat([df_sampled, df_group_sampled])
        return df_sampled


def dist_to_nearest_distribution_Hamming(df):
    """
    Calculate the distance to nearest neighbor using Normalized Hamming distance,
    and only compute the distance when the sequence pair are of the same length.
    :param df:
    :return:
    """
    len_counts = df.loc[:, "JUNCTION length"].value_counts()
    keys, values = group_seq(df, keys=["JUNCTION length"])
    df_summary = pd.DataFrame(columns=["Junction length", "number of seq", "index", "d to nearest"])
    for i, length in enumerate(keys):
        start_time = time.time()
        index = list(values)[i]
        n_seq = len_counts[length]
        sequences = df.loc[index, "JUNCTION"].values
        dis = dist_pairwise(sequences)
        d_to_nearest = dist_to_nearest(dis)
        df_summary.loc[i, :] = [length, n_seq, index, d_to_nearest]
        print("For length %s , %s sequences, use %s seconds." % (length, n_seq,
                                                                 time.time() - start_time))
    return df_summary


def det_cutoff():
    return


def check_duplicate():  # check duplicated sequence across groups
    return
