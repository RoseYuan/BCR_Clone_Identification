"""
Distance function.
"""

import Levenshtein
import numpy as np


def Normalized_Hamming_dist(string1,string2):
    """Returns the Hamming distance between array A and B
    if A and B are of different length, the distance is 1"""
    if len(string1) != len(string2):
        return 1
    else:
        H = 0
        for i in range(len(string1)):
            if string1[i] != string2[i]:
                H += 1     
        return H/len(string1)


def Normalized_Levenshtein_dist(string1, string2):
    """
    [ref] Yujian, Li, and Liu Bo. "A normalized Levenshtein distance metric." 
    IEEE transactions on pattern analysis and machine intelligence 29.6 (2007): 1091-1095.
    """
    if len(string1)==0 and len(string2)==0:
        return 1
    else:
        Lev = Levenshtein.distance(string1, string2)
        norm_lev = 2*Lev/(len(string1)+len(string2)+Lev)
        return norm_lev


def Cosine_dist(vec1, vec2):
    "Calculate the cosine distance between two vectors of te same dimension."
    if len(vec1) != len(vec2):
        raise ValueError("different length! Cannot compute the inner product.")
    return 1-np.dot(vec1, vec2)