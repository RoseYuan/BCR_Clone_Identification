# a more general way to include well-defined diversiy index and Hill's diversity

import numpy as np
import random
import matplotlib.pyplot as plt
import math

"""
    Implementation various diversity metric for biological samples,
    and similarity metric between samples.

    A sample is a dictionary containing the name of the species as a key, 
    and the number of detected individual for that species as a value. ex:
        sample["mouse"] = 5
        sample["rat"] = 10

    During the calculations, the occurences are always normalized to the total number of detected individuals

    [ref] Hill MO. Diversity and evenness: a unifying notation and its consequences. 
          Ecology. 1973 Mar;54(2):427-32.
"""


# ----------------------- Biodiversity tools ----------------------- #
def normalize_sample(sample):
    Ntot = np.sum(np.array(list(sample.values())).astype(np.float))
    norm_sample = {k: float(v) / Ntot for k, v in sample.items()}
    return norm_sample


def species_accumulation_curve(sample):
    speciesID = []  # build a list of all individual wih their species
    for species, count in sample.items():
        for i in range(count):
            speciesID.append(species)

    species_set = set()  # build the accumulation curve
    Ntot = len(speciesID)
    accumulation = np.zeros(Ntot)
    random_individuals = random.sample(range(Ntot), Ntot)
    i = 0
    for indiv in random_individuals:
        species_set.add(speciesID[indiv])
        accumulation[i] = len(species_set)
        i += 1

    plt.plot(accumulation)
    plt.xlabel("Individual considered")
    plt.ylabel("Observed species")
    plt.show()


def cal_acc_curve(sample):
    """Calculate the species accumulation curve"""
    speciesID = []  # build a list of all individual wih their species
    for species, count in sample.items():
        for i in range(count):
            speciesID.append(species)

    species_set = set()  # build the accumulation curve
    Ntot = len(speciesID)
    accumulation = np.zeros(Ntot)
    random_individuals = random.sample(range(Ntot), Ntot)
    i = 0
    for indiv in random_individuals:
        species_set.add(speciesID[indiv])
        accumulation[i] = len(species_set)
        i += 1

    return accumulation


# ----------------------- Diversity indexes ----------------------- #
def Hill_diversity(sample, q):
    inf_threshold = 100
    if q == 0:
        Diversity = np.count_nonzero(np.array(list(sample.values())))
    else:
        sample = normalize_sample(sample)
        pi = np.array(list(sample.values()))  # relative  abundance  of  species
        pi = pi[pi > 0]
        if q == 1:  # exponential of Shanon entropy
            Diversity = np.exp(np.sum(-pi * np.log(pi)))
        elif q >= inf_threshold:
            Diversity = 1 / np.max(pi)
        else:
            Diversity = np.power(np.sum(np.power(pi, q)), 1 / (1 - q))
    return Diversity

def richness(sample):
    sample = {k: float(v) for k, v in sample.items()}
    ki = np.array(list(sample.values()))
    S_obs = np.count_nonzero(ki)
    return S_obs

def richness_chao(sample):
    """
     [ref] A.  Chao,  “Nonparametric  estimation  of  the  number  of  classes  in  a  population"
           Scandinavian Journal of statistics, pp. 265–270, 1984.
    """
    sample = {k: float(v) for k, v in sample.items()}
    ki = np.array(list(sample.values()))
    S_obs = np.count_nonzero(ki)

    if np.min(ki) < 1:
        denormalize_factor = int(1 / np.min(ki))
        sample = {k: v * denormalize_factor for k, v in sample.items()}
        ki = np.array(list(sample.values()))
    f1 = np.size(np.where(ki == 1))
    f2 = np.size(np.where(ki == 2))

    if f1 == 0 or f2 == 0:
        S_chao = S_obs
    else:
        S_chao = S_obs + f1 * (f1 - 1) / (2 * (f2 + 1))
    return S_chao


def Shannon_entropy(sample):
    entropy = np.log(Hill_diversity(sample, 1))
    return entropy


def Shannon_entropy_Chao(sample):
    """
     [ref] A. Chao and T.-J. Shen, “Nonparametric estimation of shannon’s index of diversity
           when there are unseen species in sample,”Environmental and ecological statistics,
           vol. 10, no. 4, pp. 429–443, 2003.
    """
    sample = {k: float(v) for k, v in sample.items()}
    ki = np.array(list(sample.values()))
    if np.min(ki) < 1:
        denormalize_factor = int(1 / np.min(ki))
        sample = {k: v * denormalize_factor for k, v in
                  sample.items()}  # Sample should not be normalized to compute Shanon entropy
        ki = np.array(list(sample.values()))
    Ntot = np.sum(np.array(list(sample.values())).astype(np.float))
    f1 = np.size(np.where(ki == 1))
    C = 1 - f1 / Ntot

    pi = ki / Ntot * C
    pi = pi[pi > 0]
    entropy_chao = np.sum(-pi * np.log(pi) / (1 - np.power(1 - pi, Ntot)))
    return entropy_chao


def Simpson_index(sample):
    # probability that two  entities  taken  at  random  from
    # the  dataset  are  of  the  same  type
    Simpson = 1 / Hill_diversity(sample, 2)
    return Simpson


def eveness(sample, a=1, b=0):
    Eveness = Hill_diversity(sample, a) / Hill_diversity(sample, b)
    return Eveness


def eveness_chao(sample):
    Eveness_chao = np.exp(Shannon_entropy_Chao(sample)) / richness_chao(sample)
    return Eveness_chao


def dominance(sample):
    sample = normalize_sample(sample)
    Dominance = np.max(list(sample.values()))
    return Dominance


# ----------------------- Similarity indexes ----------------------- #
def Dice_similarity(sample1, sample2, normalize=True):
    if normalize == True:
        sample1 = normalize_sample(sample1)
        sample2 = normalize_sample(sample2)

    species_list = set(list(sample1.keys()) + list(sample2.keys()))
    Dice = 0
    for species in species_list:
        if species in sample1.keys() and species in sample2.keys():
            Dice += min(sample1[species], sample2[species])
    return Dice


def Jaccard_similarity(sample1, sample2):
    Dice = Dice_similarity(sample1, sample2)
    Jaccard = Dice / (2 - Dice)
    return Jaccard

# --------------------- axis scalar -------------------------------- #
# get a series of alpha by transformation function
def transform_func(initial_axis):
    y1 = np.tan(math.pi*initial_axis/2)
    y2 = np.exp(y1)
    return y2

def rev_transform(y2):
    y1 = np.log(y2)
    initial_axis = np.arctan(y1)*2/math.pi
    return initial_axis