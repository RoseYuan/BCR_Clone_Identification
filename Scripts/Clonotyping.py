"""Implementation of three clone identification methods: V-gene-based, V-J-junction-based, alignment-free."""

from libSeq import *
from Utils import *
from sklearn.metrics import pairwise_distances as numeric_pairwise_distances
from edist.multiprocess import pairwise_distances_symmetric, pairwise_distances
from HAC_clustering import *
from SeqRepresent import *
"""
Clonotyping methods
"""

# Junction-based method: clustering according to CDR3 sequences

def cluster_by_sequence(seqs):
    """
    Only BCRs with the same (e.g. junction) sequence can be clustered together
    """
    groups = list(np.unique(seqs))
    clusters = [groups.index(x) for x in seqs]
    return clusters


# pairwise distance matrix based method: first calculate a distance matrix, then do clustering
def v_j_junction_dis(df, field_group, field_dis, groupby, field='JUNCTION',metric=Normalized_Levenshtein_dist, diag=1., default_dist=1.):
    # group & represent
    sequences = df.loc[:,field].values
    df_unique = df.drop_duplicates(subset=field_group, ignore_index=True)
    unique_sequences = list(df_unique.loc[:,field].values)
    representor = [unique_sequences.index(x) for x in sequences]
    dis = cal_group_pdist(df_unique, fields=field_dis, groupby=groupby, metric=metric, diag=diag, default_dist=default_dist)
    return dis, representor

def align_free_dis(df, df_unique, df_rep, field_group, metric=Cosine_dist, diag=1.):
    '''
    df: df_nt for sample
    df_unique: df_unique for sample
    df_rep: representation for sample
    '''
    # group & represent
    sequences = df.loc[:,field_group].values
    unique_sequences = list(df_unique.loc[:,field_group].values)
    representor = [unique_sequences.index(x) for x in sequences]
    # calculate the distance matrix
    dis = cal_numeric_pdist(df_rep, metric=metric, diag=diag)
    return dis, representor


def HAC_cluster_by_distance_fields(representor, dis, d_threshold, **kwargs):
    """
    represent a group of sequences by one representor, (e.g. with the same junction sequence), assign the same clustering
    resluts
    df: the whole dataset
    representor: a list that contains the index (in dis) of corresponding representative sequence
    """
    unique_clusters = cluster_HAC(dis, d_threshold, **kwargs)
    clusters = []
    for i in representor:
        clusters.append(unique_clusters[i])
    return clusters

# v-j-junction-based method: unique junction, group by V,J gene assignment, Levenshtein distance between junction, threshold detected by bimodality


"""
Calculating pairwise distance matrix using different methods
"""

# alignment-based method

def cal_edit_pdist(Xs, Ys=None, metric=Normalized_Levenshtein_dist, diag=1.):
    """
    Compute the pairwise edit distances between the objects in Xs or between the objects in Xs and the objects in Ys. 
    Each object in Xs and Ys needs to be a valid input for the given distance function.
    """
    if Ys is None:
        dis = pairwise_distances_symmetric(Xs, metric)
        np.fill_diagonal(dis, diag)
    else:
        dis = pairwise_distances(Xs, Ys, metric)
    return dis

def cal_group_pdist(df1, fields, groupby, df2=None, metric=Normalized_Levenshtein_dist, diag=1., default_dist=1.):
    """
    Calculate pairwise distances within a dataframe or between two dataframes, 
    only compute the distance when the sequence pairs are within the same group.

    :param groupby: a list of columns to group the df1
    :param df2: if not None, compute all distance between seqs in df1 and df2
    :param fields: the columns used to define the distance
    :param metric: distance metric
    :param diag: if the distance matrix is symmetric, fill the diagnol
    :param default_dist: default distance value in the distance matrix if the pairwise distance is not calculated
    :return: distance matrix, and a list of sequence index where no distance is calculated (due to groupby)
    """
    df1_cp = df1.reset_index(drop=True)

    keys1, values1 = group_seq(df1_cp, keys=groupby)
    l1, _ = df1_cp.shape
    no_dis_cal = [] # index list of sequence that the program does not calculate any distance of it
    if df2 is None: # then calculate pairwise distance of df1
        dis = np.full(shape=(l1, l1), fill_value=default_dist)
        for i, key in enumerate(keys1):
            index = list(values1)[i]
            sequences = df1_cp.loc[index, fields].values
            if len(sequences) > 1:
                group_dis = cal_edit_pdist(sequences, metric=metric, diag=diag)
                for m,ind1 in enumerate(index):
                    for n,ind2 in enumerate(index):
                        dis[ind1,ind2] = group_dis[m,n]
            else:
                no_dis_cal.append(index)

    else:
        df2_cp = df2.reset_index(drop=True)
        keys2, values2 = group_seq(df2_cp, keys=groupby)
        l2, _ = df2_cp.shape
        dis = np.full(shape=(l2, l1), fill_value=default_dist)
        for i, key in enumerate(keys2):
            index2 = list(values2)[i]
            if key in keys1: # if the keys are common in both dataframes, calculate the distance matrix
                index1 = list(values1)[list(keys1).index(key)]
                sequences1 = df1_cp.loc[index1, fields].values
                sequences2 = df2_cp.loc[index2, fields].values
                group_dis = cal_edit_pdist(sequences1, sequences2, metric=metric, diag=diag)
                for m,ind2 in enumerate(index2):
                    for n,ind1 in enumerate(index1):
                        dis[ind2,ind1] = group_dis[n,m]
    return dis


# alignment-free method

def cal_numeric_pdist(X,Y=None, metric=Cosine_dist, diag=1.):
    """
    Compute the distance matrix from a vector array X and optional Y, fill the diagonal if the distance matrix is symetric.
    :param X,Y: should be vector arrays, matrix or dataframe with only numerical values.
    """
    dis = numeric_pairwise_distances(X, Y, metric)
    if np.any(dis<0):
        dis[dis<0] = 0
    if Y is None:
        np.fill_diagonal(dis, diag)
        dis = np.tril(dis) + np.triu(dis.T, 1)  # make sure the distance matrix is symmetric
    return dis

def cal_numeric_represent_pdis(df1, fields, transform_func, metric=Cosine_dist, diag=1., df2=None, **kwargs):
    """
    Calculate the pairwise distance by first learn the numeric represenation of the specified fields in dataframes, then compute pairwise numeric distance.
    """
    sequences1 = df1.loc[:, fields].values
    sequences2 = None
    if df2 is not None:
        sequences2 = df2.loc[:, fields].values
    rep1, rep2, _ = transform_func(sequences1, sequences2, **kwargs)
    dis = cal_numeric_pdist(rep1, rep2, metric=metric, diag=diag)
    return dis, rep1, rep2







