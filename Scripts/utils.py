import pandas as pd
import numpy as np
from libSeq import *
import time, re


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
    alleles = re.split(", or | or |\*",allele_notation)
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
    df_nes.to_csv(outfile, sep='\t', index=False)


def group_seq(df_nes: pd.DataFrame, keys=('V-GENE', 'J-GENE', 'JUNCTION length')):
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
    counts = 0
    l = len(seq_list)
    dis = np.empty(shape=(l, l))
    dis[:] = 1.5
    for i, string1 in enumerate(seq_list):
        for j, string2 in enumerate(seq_list):
            if j > i:
                d = distance(string1, string2)
                counts += 1
                dis[i, j] = d
    return dis, counts


def dist_to_nearest(dis: np.array):
    dis = np.triu(dis)
    dis = dis + dis.T - np.diag(np.diag(dis))
    d_to_nearest = np.nanmin(dis, axis=0)
    return d_to_nearest


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
