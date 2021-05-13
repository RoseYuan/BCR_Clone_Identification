import pandas as pd
import numpy as np
from libSeq import *
import time, re
from itertools import product
from collections import Counter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import multiprocessing as mp
"""
Preprocessing:
"""


def clean_data(df):
    """
    Remove duplicated V-D-J-REGION sequence, and remove items if V-DOMAIN Functionality != "productive"
    :param df:
    :return:
    """
    n_rows0, _ = df.shape
    print("%d sequences in total." % n_rows0)
    df = df.drop_duplicates(subset="V-D-J-REGION", ignore_index=True)
    n_rows1, _ = df.shape
    print("Drop %d duplicated sequences." % (n_rows0 - n_rows1))
    df = df.loc[df["V-DOMAIN Functionality"] == "productive", :]
    n_rows2, _ = df.shape
    print("Drop %d sequences by filtering V-DOMAIN Functionality." % (n_rows1 - n_rows2))
    print("%d sequences remain." % n_rows2)
    return df


def read_VJ_genes(allele_notation) -> str:
    """
    Get the V/J fragment annotation regardless of the allele types. If multiple fragment types
    appear, only preserve the first one.

    :param series:
    :return:
    """
    alleles = re.split(", or | or |\*", allele_notation)
    genes = [gene for gene in alleles if "Homsap" in gene]
    genes = list(set(genes))
    return genes[0]


def read_data(nt_file: str, outfile: str):
    """
    Get necessary information and save to a file.
    :param nt_file: Nt sequence without gaps
    :param outfile: output file path and name
    :return: None
    """
    df_n = pd.read_csv(nt_file, sep='\t')
    # clean the data
    df_n = clean_data(df_n)
    # get junction length
    df_n['JUNCTION length'] = df_n['JUNCTION end'] - df_n['JUNCTION start'] + 1
    # get V/J gene annotations
    df_n['V-GENE'] = df_n['V-GENE and allele'].apply(read_VJ_genes)
    df_n['J-GENE'] = df_n['J-GENE and allele'].apply(read_VJ_genes)
    df_nes = df_n[
        ['Sequence number', 'Sequence ID', 'V-GENE', 'J-GENE', 'V-D-J-REGION', 'JUNCTION',
         'JUNCTION length']]
    rows1, _ = df_nes.shape
    df_nes = df_nes.dropna()
    rows2, _ = df_nes.shape
    print("Drop %d rows contain nan." % (rows1 - rows2))
    df_nes = df_nes.sort_values(by=['JUNCTION length'])
    df_nes.to_csv(outfile, sep='\t', index=False)


def group_seq(df_nes: pd.DataFrame, keys=('V-GENE', 'J-GENE', 'JUNCTION length')):
    seq_groups = df_nes.groupby(list(keys)).groups
    return seq_groups.keys(), seq_groups.values()


"""
Calculating distances:
"""


def dist_pairwise(seq_list1, seq_list2=None, distance=Normalized_Hamming_dist) -> np.array:
    """
    Calculate pairwise distance within a sequence list.
    If two strings are of different length, d=1.5
    :param seq_list1:
    :param distance: distance metric
    :param seq_list2:
    :return:
    """
    if seq_list2 is None:
        seq_list2 = seq_list1
    counts = 0
    l1 = len(seq_list1)
    l2 = len(seq_list2)
    dis = np.empty(shape=(l1, l2))
    dis[:] = 1

    for i in range(l1):
        for j in range(l2):
            if i!=j:
                d = distance(seq_list1[i], seq_list2[j])
                counts += 1
                dis[i, j] = d
    # dis = np.triu(dis)
    # dis = dis + dis.T - np.diag(np.diag(dis))
    return dis, counts


def dist_to_nearest(dis: np.array):
    # dis = np.triu(dis)
    # dis = dis + dis.T - np.diag(np.diag(dis))
    d_to_nearest = np.nanmin(dis, axis=0)
    return d_to_nearest


"""
Sampling:
"""


def subsample(df, random_state=1, frac_small=0.9, frac=None, n=None, groups=None) -> pd.DataFrame:
    """

    :param frac_small:
    :param n: cannot be used with frac
    :param df:
    :param frac: cannot be used with n
    :param random_state:
    :param groups: a list of index, used to stratify df
    :return:
    """
    if n is not None and frac is not None:
        raise ValueError("n cannot be used with frac! Choose one to use.")

    df_sampled = pd.DataFrame(columns=df.columns)

    if groups is None:
        return df.sample(frac=frac, n=n, random_state=random_state)
    else:
        for group in groups:
            if (n is not None) and (len(group) < n):
                df_group_sampled = df.loc[group, :].sample(frac=frac_small, random_state=1)
            else:
                df_group_sampled = df.loc[group, :].sample(frac=frac, n=n, random_state=1)
            df_sampled = pd.concat([df_sampled, df_group_sampled])
        return df_sampled


def check_duplicate():  # check duplicated sequence across groups
    return


"""
Compute tf-idf representation:
"""


def k_mer_set(k=4, atoms=["a", "t", "c", "g"]) -> list:
    K = [''.join(kmer) for kmer in product(atoms, repeat=k)]
    K.sort()
    return K


def bag_of_words(sequence: str, K: list):
    sequence = sequence.lower()
    bag = dict.fromkeys(K, 0)
    k = len(K[0])
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in K:
            bag[kmer] += 1
    return bag


def count_words_in_collection(bag_list: list) -> dict:
    counter = Counter()
    for bag in bag_list:
        counter.update(bag)
    return dict(counter), len(bag_list)


def idf(counter: dict, n_seq: int) -> dict:
    idf_k = {}
    for kmer in counter.keys():
        if counter[kmer] != 0:
            idf_k[kmer] = np.log(n_seq / counter[kmer])
    return idf_k


def tf_idf(bag: dict, idf_k: dict):
    """
    Calculate the tf-idf representation of a sequence.
    :param bag:
    :param idf_k:
    :return:
    """
    tf_idf_vector = {}
    no_occured_kmers = bag.keys() - idf_k.keys()
    for kmer in no_occured_kmers:
        bag.pop(kmer, None)
    if bag.keys() != idf_k.keys():
        raise ValueError("Different kmer sets. Please check.")
    for kmer in idf_k.keys():
        tf_idf_vector[kmer] = bag[kmer] * idf_k[kmer]
    l2_norm = np.linalg.norm(np.array(list(tf_idf_vector.values())))
    nor_tf_idf_dict = {k: v / l2_norm for k, v in tf_idf_vector.items()}
    return nor_tf_idf_dict


def cal_tf_idf(sequences, k=4, atoms=["a", "t", "c", "g"]):
    """

    :param sequences:
    :param k:
    :param atoms:
    :return: a list of dict
    """
    K = k_mer_set(k, atoms)
    bag_list = [bag_of_words(seq, K) for seq in sequences]
    counter, n_seq = count_words_in_collection(bag_list)
    idf_k = idf(counter, n_seq)
    tf_idf_list = [tf_idf(bag, idf_k) for bag in bag_list]
    return tf_idf_list


"""
Detect the threshold:
"""


def outlier_based_cutoff(d_to_nearest_all):
    bins = 30
    cov = 1.5
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Distance-to-nearest neighbor distribution")

    # the histogram
    ax[0].hist(d_to_nearest_all, bins=bins, label="frequency")
    ax[0].set_yscale('log')

    ax[0].set_xlabel("Normalized Hamming distance")
    ax[0].set_ylabel("Frequency")

    # the smoothed curve
    freq = np.histogram(d_to_nearest_all, bins=bins)
    freq_pos = [f if f > 0 else f + 1 for f in freq[0]]  # mask zero
    smoothed_freq0 = gaussian_filter1d(np.log(freq_pos), cov)
    ax[0].plot(freq[1][:-1], np.exp(smoothed_freq0), label='smoothed')
    ax[0].legend()

    # ___ detection of local minimums and maximums ___
    min_ind = (np.diff(np.sign(np.diff(smoothed_freq0))) > 0).nonzero()[0] + 1

    # local min
    loc_min = freq[1][min_ind]
    ax[1].plot(freq[1][:-1], np.exp(smoothed_freq0), label='smoothed', c='orange')
    ax[1].set_yscale('log')
    ax[1].set_xlabel("Normalized Hamming distance to nearest")
    ax[1].set_ylabel("Frequency")
    ax[1].set_ylim(ax[0].get_ylim())
    plt.axvline(loc_min, ymin=-0.002, linestyle='--', c='grey')
    plt.text(loc_min - 0.05, 1, "%1f" % loc_min[0], color='red')
    plt.grid()
    ax[1].legend()
    plt.show()

    return loc_min


def negation_based_cutoff(d_to_nearest_cp, tolerance):
    """
    :param d_to_nearest_cp: the distribution of distances between negation sequences and their closest
    counterpart in the repertoire
    :param tolerance: the fraction of the distances to negation sequences that are allowed within the
    cluster (false-positive rate)
    :return: the cutoff value
    """
    return np.quantile(d_to_nearest_cp, tolerance)


if __name__ == "__main__":
    K = k_mer_set(k=2)
    print("K:", K)
    s1 = "aAtGTgac"
    s2 = "aAAtGcTga"
    s3 = "aAttcgatca"
    s4 = "aAtctaggcg"
    z = bag_of_words(s1, K)
    a = bag_of_words(s2, K)
    b = bag_of_words(s3, K)
    c = bag_of_words(s4, K)
    counter, n = count_words_in_collection([z, a, b, c])
    print("bag:", [z, a, b, c])
    print("Counter:", counter)
    idf_k = idf(counter, n)

    nor_tf_idf = tf_idf(a, idf_k)
    print(idf_k)
    print(nor_tf_idf)
    print(sum(nor_tf_idf.values()))

    seqs = [s1, s2, s3, s4]
    print(cal_tf_idf(seqs, k=2))
    tf_idf_seq = cal_tf_idf(seqs, k=2)
    print("tf_idf_seq:", tf_idf_seq)
    dis = dist_pairwise(tf_idf_seq, distance=Cosine_dist)
    print("dis:", dis)
