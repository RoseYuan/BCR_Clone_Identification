import numpy as np
import pandas as pd

"""
Hyper-parameter tunning
"""


def hist_intersection(h1, h2, bins):
    bins = np.diff(bins)
    sm = 0
    for i in range(len(bins)):
        sm += min(bins[i]*h1[i], bins[i]*h2[i])
    return sm

"""
Sampling:
"""


def subsample(df, random_state=1, frac_small=0.9, frac=None, n=None, groups=None) -> pd.DataFrame:
    """
    Subsampling the row of df, either by sampling n rows or by sampling a fraction of rows.
    When groups is not None, sampling is stratified by groups. If sample n rows from each group, sample
    frac-small of rows from group that do not have enough n rows.
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



def group_seq(df_nes: pd.DataFrame, keys:list):
    """
    Group the dataframe by a set of keys.
    :return: the list of the keys and the list of corresponding items.
    """
    if keys is None:
        return [0], [df_nes.index]
    else:
        seq_groups = df_nes.groupby(list(keys)).groups
        return seq_groups.keys(), seq_groups.values()


def dist_to_nearest(dis: np.array):
    """
    Get the distance to the nearest vector given a distance matrix.
    """
    return np.nanmin(dis, axis=1)

