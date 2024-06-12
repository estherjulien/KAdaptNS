import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import joblib
import copy
import time


class TrainRF:
    def __init__(self, X, y, problem, N, K, min_train=10, h_train=2, ct=5, balanced=True, sc_pre="dive", sc_min_max=0):
        self.rng = np.random.RandomState(1)
        self.X = X
        self.y = y
        self.class_thresh = ct/100
        self.balanced = balanced
        self.sc_pre = sc_pre
        self.sc_min_max = sc_min_max

        # paths
        if "cb" in problem:
            os.makedirs("src/cb/data", exist_ok=True)
            ml_info = f"{problem}_N{N}_K{K}_min{min_train}_nodes{h_train}_ct{ct}_bal_scp-{sc_pre}_scmm{sc_min_max}"
            self.ml_model_path = f"src/cb/data/ml_model_{ml_info}.joblib"
            self.ml_info_path = f"src/cb/data/ml_info_{ml_info}.pkl"
            self.feat_imp_path = f"src/cb/data/ml_feat_imp_{ml_info}.pkl"
        elif "sp" in problem:
            os.makedirs("src/sp/data", exist_ok=True)
            ml_info = f"{problem}_N{N}_K{K}_min{min_train}_nodes{h_train}_ct{ct}_bal_scp-{sc_pre}_scmm{sc_min_max}"
            self.ml_model_path = f"src/sp/data/ml_model_{ml_info}.joblib"
            self.ml_info_path = f"src/sp/data/ml_info_{ml_info}.pkl"
            self.feat_imp_path = f"src/sp/data/ml_feat_imp_{ml_info}.pkl"

    def data_pipeline(self):
        print(" Preparing data...")
        X = self.X
        y = pd.Series(self.y)
        # clean train data
        X = X.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        Y = y.replace([np.inf, -np.inf], np.nan).dropna()

        # find similar indices
        same_indices = X.index.intersection(Y.index)
        X = X.loc[same_indices]
        Y = Y[same_indices]

        # check the classes
        og = []
        for p in np.arange(0.1, 1.1, 0.1):
            perc = ((Y >= p - 0.1) & (Y <= p)).sum() / len(Y)
            og.append(np.round(perc, 2))
        print("     distribution original data:", og)

        Y[Y < self.class_thresh] = 0
        Y[Y >= self.class_thresh] = 1

        before = []
        for p in np.arange(0, 2):
            perc = (Y == int(p)).sum() / len(Y)
            before.append(np.round(perc, 2))
        print("     distribution before balancing:", before)

        if self.balanced:
            print("     Balancing data...")
            X["class"] = Y
            g = X.groupby('class')
            g = pd.DataFrame(
                g.apply(lambda x: x.sample(int(g.size().mean()), random_state=self.rng, replace=True).reset_index(drop=False)))
            X = g
            X.index = X["index"]
            X.drop(["index", "class"], axis=1, inplace=True)
            Y = Y[X.index]

        after = []
        for p in np.arange(0, 2):
            perc = (Y == int(p)).sum() / len(Y)
            after.append(np.round(perc, 2))
        print("     distribution after balancing:", after)
        scales = []
        if self.sc_min_max:
            print("     Min/max scaling of data")
            scales = [X.min(axis=0), X.max(axis=0)]
            X_sc = (X - scales[0])/(scales[1] - scales[0])
            if any((scales[1] - scales[0]) == 0):
                culprits = scales[1][((scales[1] - scales[0]) == 0)].index
                for i in culprits:
                    X_sc[i] = 0
            X = X_sc

        print(" Data preparation completed.")

        return X, Y.to_numpy(), scales

    def train_rf(self):
        # get data
        X, y, scales = self.data_pipeline()
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=self.rng)
        # train NN
        print(" Training random forest...")
        rf = RandomForestClassifier(random_state=self.rng)
        rf._scales = scales
        start_time = time.time()
        rf.fit(X_train, y_train)
        print(" Finished training.")

        # evaluation
        score = rf.score(X_test, y_test)
        data_0_index = np.where(y_test == 0)[0]
        type_1 = 1 - rf.score(X_test.iloc[data_0_index], y_test[data_0_index])
        data_1_index = np.where(y_test == 1)[0]
        type_2 = 1 - rf.score(X_test.iloc[data_1_index], y_test[data_1_index])

        print(f"    RF classification test accuracy = {score}")
        print(f"    RF classification test type I error = {type_1}")
        print(f"    RF classification test type II error = {type_2}")

        # feature importance
        feature_importance = pd.Series(rf.feature_importances_, index=self.X.columns)
        print("     Feature importance:\n")
        print(feature_importance)

        # save
        joblib.dump(rf, self.ml_model_path)
        feature_importance.to_pickle(self.feat_imp_path)

        model_info = pd.Series(
            [len(X_train), score, type_1, type_2, time.time() - start_time],
            index=["datapoints", "accuracy", "type_1", "type_2", "runtime"])
        model_info.to_pickle(self.ml_info_path)