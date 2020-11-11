import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features

Feature.dir = 'features'


class Carat(Feature):
    def create_features(self):
        self.train['carat'] = train['carat']
        self.test['carat'] = test['carat']


class Cut(Feature):
    # Describe cut quality of the diamond. Quality in increasing order Fair, Good, Very Good, Premium, Ideal
    def create_features(self):
        self.train['cut'] = train['cut'].replace(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], [0, 1, 2, 3, 4])
        self.test['cut'] = test['cut'].replace(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], [0, 1, 2, 3, 4])


class Color(Feature):
    #  diamond colour, from J (worst) to D (best)
    def create_features(self):
        self.train['color'] = train['color'].replace(['J', 'I', 'H', 'G', 'F', 'E', 'D'], [0, 1, 2, 3, 4, 5, 6])
        self.test['color'] = test['color'].replace(['J', 'I', 'H', 'G', 'F', 'E', 'D'], [0, 1, 2, 3, 4, 5, 6])


class Clarity(Feature):
    #  diamond colour, from J (worst) to D (best)
    def create_features(self):
        self.train['clarity'] = train['clarity'].replace(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], [0, 1, 2, 3, 4, 5, 6, 7])
        self.test['clarity'] = test['clarity'].replace(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], [0, 1, 2, 3, 4, 5, 6, 7])


class Depth(Feature):
    def create_features(self):
        self.train['depth'] = train['depth']
        self.test['depth'] = test['depth']


class Table(Feature):
    def create_features(self):
        self.train['table'] = train['table']
        self.test['table'] = test['table']


class X(Feature):
    def create_features(self):
        self.train['x'] = train['x']
        self.test['x'] = test['x']


class Y(Feature):
    def create_features(self):
        self.train['y'] = train['y']
        self.test['y'] = test['y']


class Z(Feature):
    def create_features(self):
        self.train['z'] = train['z']
        self.test['z'] = test['z']


if __name__ == '__main__':
    args = get_arguments()
    train = pd.read_feather('./data/interim/train.feather')
    test = pd.read_feather('./data/interim/test.feather')
    print(train.head())

    generate_features(globals(), args.force)
