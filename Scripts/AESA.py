import math
from utils import *


class Aesa:
    """
    the nearest - neighbour approximating and eliminating search algorithm(AESA), Mico, Oncina, Vidal '94
    modified code originally from https://tavianator.com/2016/aesa.html
    """
    def __init__(self, candidates, distance):
        """
        Initialize an AESA index.

        candidates: The list of candidate points.
        distance: The distance metric.
        """

        self.candidates = candidates
        self.distance = distance

        # Pre-compute all pairs of distances
        self.precomputed, self.pre_counts = dist_pairwise(candidates,distance=distance)
        self.best = None

    def get_pre_count(self):
        return self.pre_counts

    def get_pilot_d_to_nearest(self):
        return dist_to_nearest(self.precomputed)

    def nearest(self, target):
        """Return the nearest candidate to 'target'."""

        size = len(self.candidates)

        # All candidates start out alive
        alive = list(range(size))
        # All lower bounds start at zero
        lower_bounds = [0] * size

        best_dist = math.inf
        counts = 0
        mask = None
        # Loop until no more candidates are alive
        while alive:
            # *Approximating*: select the candidate with the best lower bound
            active = min(alive, key=lambda i: lower_bounds[i])
            # Compute the distance from target to the active candidate
            # This is the only distance computation in the whole algorithm
            active_dist = self.distance(target, self.candidates[active])
            counts+=1
            # Update the best candidate if the active one is closer
            if active_dist < best_dist:
                # Only get non-identical nearest neighbor
                if active_dist == 0:
                    mask = active
                else:
                    self.best = active
                    best_dist = active_dist

            # *Eliminating*: remove candidates whose lower bound exceeds the best
            old_alive = alive
            alive = []

            for i in old_alive:
                # Compute the lower bound relative to the active candidate
                lower_bound = abs(active_dist - self.precomputed[active][i])
                # Use the highest lower bound overall for this candidate
                lower_bounds[i] = max(lower_bounds[i], lower_bound)
                # Check if this candidate remains alive
                if (lower_bounds[i] < best_dist) and (i!= active): # remove rejected points and known points
                    if i == mask:
                        continue
                    alive.append(i)

        return best_dist, counts

if __name__ == "__main__":

    # from random import random
    # dimensions = 3
    # def random_point():
    #     return [random() for i in range(dimensions)]
    #
    # def euclidean_distance(x, y):
    #     global count
    #     count += 1
    #
    #     s = 0
    #     for i in range(len(x)):
    #         d = x[i] - y[i]
    #         s += d * d
    #     return math.sqrt(s)
    #
    # count = 0
    # points = [random_point() for n in range(1000)]
    # aesa = Aesa(points, euclidean_distance)
    # print('{0} calls during pre-computation'.format(count))
    # count = 0
    #
    # aesa.nearest(random_point())
    # print('{0} calls during nearest neighbour search'.format(count))
    # count = 0
    #
    # for i in range(1000):
    #     aesa.nearest(random_point())
    # print('{0} calls on average during nearest neighbour search'.format(count / 1000))
    # count = 0

    def eu_distance_1d(x,y):
        return abs(x-y)

    points = [1,2,3,4,5,8,7,8,6,10,100]
    aesa = Aesa(points, eu_distance_1d)

    print('{0} calls during pre-computation'.format(aesa.get_pre_count()))

    d_to_nn, count = aesa.nearest(110)
    print('Distance to nearest neighbor: %d'%(d_to_nn))
    print('{0} calls during nearest neighbour search'.format(count))

