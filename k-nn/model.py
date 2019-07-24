import csv
from pprint import pprint
import random
import math
import operator

class KNN():

    def __init__(self, x_train, y_train, k=3, distance_fn='euclidian'):
        self.k = k
        self.distance_fn = distance_fn
        self.x_train = x_train
        self.y_train = y_train

    def find_neighbors(self, point):
        neighbors = []
        distances = self.distance(point)[:self.k]
        for x, y in distances:
            neighbors += [(x, y)]
        return neighbors

    def accuracy(y_pred, y_test):
        correct = 0
        for a, b in zip(y_pred, y_test):
            correct += 1 if a == b else 0
        return (correct/float(len(y_test))) * 100

    def distance(self, point):
        if self.distance_fn == 'euclidian':
            distances = []
            for training_point in zip(self.x_train, self.y_train):
                d = self.euclidian_distance(*training_point[:-1], point)
                distances += [(d, training_point[-1])]
            distances.sort()
            return distances
        else:
            return None

    def euclidian_distance(self, point_a, point_b):
        dif_sum = 0
        for a, b in zip(point_a, point_b):
            dif = a - b
            dif_sum += math.pow(dif, 2)
        result = math.sqrt(dif_sum)
        return result

    def vote(self, neighbors):
        votes = {}
        for neighbor in neighbors:
            response = neighbor[1]
            if response in votes:
                votes[response] += 1
            else:
                votes[response] = 1
        votes = sorted(
            votes.items(),
            key=operator.itemgetter(1),
            reverse = True
        )
        return votes[0][0]
    
    def test(self, x_test):
        y_pred = []
        for point in x_test:
            neighbors = self.find_neighbors(point)
            y_pred += [self.vote(neighbors)]
        return y_pred


    # Fonction load
    @staticmethod
    def load(filename, split, pop_header=True):
        x_train, y_train, x_test, y_test, header = [], [], [], [], []
        with open(filename) as csvfile:
            lines = csv.reader(csvfile)
            lines = list(lines)
            if pop_header:
                header = lines.pop(0)
            for line in lines:
                for i, v in enumerate(line[:-1]):
                    v = float(v)
                    line[i] = v
                if random.random() < split:
                    x_train += [line[:-1]]
                    y_train += [line[-1]]
                else:
                    x_test += [line[:-1]]
                    y_test += [line[-1]]
        return header, x_train, y_train, x_test, y_test