import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
import bhtsne

from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
Feature.dir = 'features'


def fixing_skewness(df):
    # Getting all the data that are not of "object" type.
    numeric_feats = df.dtypes[df.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skewed_feats[abs(skewed_feats) > 0.5]
    skewed_features = high_skew.index
    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))
    return df


class NumericalFeatures(Feature):
    def create_features(self):
        features = all_df[numeric_features].copy()
        # fillna
        features['x'] = features['x'].replace(0, features['x'].median())
        features['y'] = features['y'].replace(0, features['y'].median())
        features['z'] = features['z'].replace(0, features['z'].median())

        # fixed skew
        fixed_df = fixing_skewness(features)

        scaler = StandardScaler()
        # scaler = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
        scaled_df = pd.DataFrame(scaler.fit_transform(fixed_df), columns=numeric_features)
        '''
        n_pca = 1
        pca_cols = ["pca"+str(i) for i in range(n_pca)]
        pca = PCA(n_components=n_pca)
        pca_df = pd.DataFrame(pca.fit_transform(all_df[numeric_features]), columns=pca_cols)
        pca_df[pca_cols] = scaler.fit_transform(pca_df[pca_cols])
        n_tsne = 2
        tsne_cols = ["tsne"+str(i) for i in range(n_tsne)]
        embeded = pd.DataFrame(bhtsne.tsne(all_df[numeric_features].astype(np.float64), dimensions=n_tsne, rand_seed=10), columns=tsne_cols)
        features = pd.concat([scaled_df, pca_df, embeded], axis=1)
        '''

        self.train = scaled_df[:train.shape[0]].reset_index(drop=True)
        self.test = scaled_df[train.shape[0]:].reset_index(drop=True)


class CategoricalFeatures(Feature):
    def create_features(self):
        final_features = pd.get_dummies(all_df[categorical_features].astype(str))
        self.train = final_features[:train.shape[0]]
        self.test = final_features[train.shape[0]:]


class CategoricalLabel(Feature):
    def create_features(self):
        features = all_df[categorical_features].copy()
        # label encode
        features['cut'] = all_df['cut'].replace(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], [0, 1, 2, 3, 4])
        features['color'] = all_df['color'].replace(['J', 'I', 'H', 'G', 'F', 'E', 'D'], [0, 1, 2, 3, 4, 5, 6])
        features['clarity'] = all_df['clarity'].replace(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], [0, 1, 2, 3, 4, 5, 6, 7])
        self.train = features[:train.shape[0]]
        self.test = features[train.shape[0]:]


class FewPolynomial(Feature):
    def create_features(self):
        features = all_df.copy()
        # fillna
        features['x'] = features['x'].replace(0, features['x'].median())
        features['y'] = features['y'].replace(0, features['y'].median())
        features['z'] = features['z'].replace(0, features['z'].median())
        # label encode
        features['cut'] = all_df['cut'].replace(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], [0, 1, 2, 3, 4])
        features['color'] = all_df['color'].replace(['J', 'I', 'H', 'G', 'F', 'E', 'D'], [0, 1, 2, 3, 4, 5, 6])
        features['clarity'] = all_df['clarity'].replace(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], [0, 1, 2, 3, 4, 5, 6, 7])

        poly_cols = ["depth table", "carat clarity", "carat color", "color clarity", "clarity depth"]
        features["depth table"] = features["depth"]*features["table"]
        features["carat clarity"] = features["carat"]*features["clarity"]
        features["carat color"] = features["carat"]*features["color"]
        features["color clarity"] = features["color"]*features["clarity"]
        features["clarity depth"] = features["clarity"]*features["depth"]

        # fixed skew
        fixed_df = fixing_skewness(features)
        poly_df = fixed_df[poly_cols]
        self.train = poly_df[:train.shape[0]].reset_index(drop=True)
        self.test = poly_df[train.shape[0]:].reset_index(drop=True)


class FewPolynomial3d(Feature):
    def create_features(self):
        features = all_df.copy()
        # fillna
        features['x'] = features['x'].replace(0, features['x'].median())
        features['y'] = features['y'].replace(0, features['y'].median())
        features['z'] = features['z'].replace(0, features['z'].median())
        # label encode
        features['cut'] = all_df['cut'].replace(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], [0, 1, 2, 3, 4])
        features['color'] = all_df['color'].replace(['J', 'I', 'H', 'G', 'F', 'E', 'D'], [0, 1, 2, 3, 4, 5, 6])
        features['clarity'] = all_df['clarity'].replace(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], [0, 1, 2, 3, 4, 5, 6, 7])

        poly_cols = ["carat^2 clarity", "carat color clarity"]

        features["carat^2 clarity"] = features["carat"]*features["carat"]*features["clarity"]
        features["carat color clarity"] = features["carat"]*features["color"]*features["clarity"]

        # fixed skew
        fixed_df = fixing_skewness(features)
        poly_df = fixed_df[poly_cols]

        self.train = poly_df[: train.shape[0]].reset_index(drop=True)
        self.test = poly_df[train.shape[0]:].reset_index(drop=True)


class Polynomial2d(Feature):
    def create_features(self):
        features = all_df.copy()
        # fillna
        features['x'] = features['x'].replace(0, features['x'].median())
        features['y'] = features['y'].replace(0, features['y'].median())
        features['z'] = features['z'].replace(0, features['z'].median())
        # label encode
        features['cut'] = all_df['cut'].replace(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], [0, 1, 2, 3, 4])
        features['color'] = all_df['color'].replace(['J', 'I', 'H', 'G', 'F', 'E', 'D'], [0, 1, 2, 3, 4, 5, 6])
        features['clarity'] = all_df['clarity'].replace(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], [0, 1, 2, 3, 4, 5, 6, 7])

        poly = PolynomialFeatures(2)
        original_fea_num = features.shape[1]
        poly_np = poly.fit_transform(features)[:, original_fea_num+1:]
        poly_features = poly.get_feature_names(features.columns)[original_fea_num+1:]
        poly_df = pd.DataFrame(poly_np, columns=poly_features)

        # fixed skew
        poly_df = fixing_skewness(poly_df)
        self.train = poly_df[: train.shape[0]].reset_index(drop=True)
        self.test = poly_df[train.shape[0]:].reset_index(drop=True)


class Polynomial3d(Feature):
    def create_features(self):
        features = all_df.copy()
        # fillna
        features['x'] = features['x'].replace(0, features['x'].median())
        features['y'] = features['y'].replace(0, features['y'].median())
        features['z'] = features['z'].replace(0, features['z'].median())
        # label encode
        features['cut'] = all_df['cut'].replace(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], [0, 1, 2, 3, 4])
        features['color'] = all_df['color'].replace(['J', 'I', 'H', 'G', 'F', 'E', 'D'], [0, 1, 2, 3, 4, 5, 6])
        features['clarity'] = all_df['clarity'].replace(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], [0, 1, 2, 3, 4, 5, 6, 7])

        poly = PolynomialFeatures(3)
        original_fea_num = features.shape[1]
        poly_np = poly.fit_transform(features)[:, original_fea_num+1:]
        poly_features = poly.get_feature_names(features.columns)[original_fea_num+1:]
        poly_df = pd.DataFrame(poly_np, columns=poly_features)
        # fixed skew
        poly_df = fixing_skewness(poly_df)
        self.train = poly_df[: train.shape[0]].reset_index(drop=True)
        self.test = poly_df[train.shape[0]:].reset_index(drop=True)


class Pca(Feature):
    # 悪化
    def create_features(self):
        n_pca = 1
        features = all_df.copy()

        # fillna
        features['x'] = features['x'].replace(0, features['x'].median())
        features['y'] = features['y'].replace(0, features['y'].median())
        features['z'] = features['z'].replace(0, features['z'].median())
        # label encode
        features['cut'] = all_df['cut'].replace(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], [0, 1, 2, 3, 4])
        features['color'] = all_df['color'].replace(['J', 'I', 'H', 'G', 'F', 'E', 'D'], [0, 1, 2, 3, 4, 5, 6])
        features['clarity'] = all_df['clarity'].replace(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], [0, 1, 2, 3, 4, 5, 6, 7])

        scaler = StandardScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

        pca_cols = ["pca"+str(i) for i in range(n_pca)]
        pca = PCA(n_components=n_pca)
        pca_df = pd.DataFrame(pca.fit_transform(scaled_df), columns=pca_cols)
        self.train = pca_df[: train.shape[0]].reset_index(drop=True)
        self.test = pca_df[train.shape[0]:].reset_index(drop=True)
        '''
        n_tsne = 2
        tsne_cols = ["tsne"+str(i) for i in range(n_tsne)]
        embeded = pd.DataFrame(bhtsne.tsne(all_df[numeric_features].astype(np.float64), dimensions=n_tsne, rand_seed=10), columns=tsne_cols)
        features = pd.concat([scaled_df, pca_df, embeded], axis=1)
        '''


if __name__ == '__main__':
    args = get_arguments()
    train = pd.read_feather('./data/interim/train.feather')
    test = pd.read_feather('./data/interim/test.feather')

    numeric_features = ["carat", "depth", "table", "x", "y", "z"]
    categorical_features = ["cut", "clarity", "color"]
    all_df = pd.concat([train.drop(['id', 'price'], axis=1), test.drop('id', axis=1)])
    train = train.drop(['id', 'price'], axis=1)
    test = test.drop('id', axis=1)

    generate_features(globals(), args.force)
