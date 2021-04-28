import math
from utils import *


class Aesa:
    def __init__(self, candidates, distance):
        """
        Initialize an AESA index.

        candidates: The list of candidate points.
        distance: The distance metric.
        """

        self.candidates = candidates
        self.distance = distance

        # Pre-compute all pairs of distances
        self.precomputed = dist_pairwise(candidates)

    def nearest(self, target):
        """Return the nearest candidate to 'target'."""

        size = len(self.candidates)

        # All candidates start out alive
        alive = list(range(size))
        # All lower bounds start at zero
        lower_bounds = [0] * size

        best_dist = math.inf

        # Loop until no more candidates are alive
        while alive:
            # *Approximating*: select the candidate with the best lower bound
            active = min(alive, key=lambda i: lower_bounds[i])
            # Compute the distance from target to the active candidate
            # This is the only distance computation in the whole algorithm
            active_dist = self.distance(target, self.candidates[active])

            # Update the best candidate if the active one is closer
            if active_dist < best_dist:
                best = active
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
                if lower_bounds[i] < best_dist:
                    alive.append(i)

        return self.candidates[best]

if __name__ == "__main__":
    from random import random

    dimensions = 3


    def random_point():
        return [random() for i in range(dimensions)]


    count = 0


    def euclidean_distance(x, y):
        global count
        count += 1

        s = 0
        for i in range(len(x)):
            d = x[i] - y[i]
            s += d * d
        return math.sqrt(s)


    points = [random_point() for n in range(1000)]
    aesa = Aesa(points, euclidean_distance)

    print('{0} calls during pre-computation'.format(count))
    count = 0

    aesa.nearest(random_point())

    print('{0} calls during nearest neighbour search'.format(count))
    count = 0

    for i in range(1000):
        aesa.nearest(random_point())

    print('{0} calls on average during nearest neighbour search'.format(count / 1000))
    count = 0
