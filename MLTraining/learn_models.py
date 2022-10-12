from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
import time

"""
This file consists of two codes:    1. code for training a random forest with sklearn.
                                    2. code to clean and balance the data
"""


# CLASSIFICATION
def train_suc_pred_rf_class(X, Y, features, problem_type="test", estimators=100, instances=1000,
                            save_map="Results", class_thresh=False, balanced=False):
    model_name = f"../{save_map}/rf_class_{problem_type}.joblib"

    X, Y, df_X, df_Y, before, after = clean_data(X, Y, classification=True, class_thresh=class_thresh, balanced=balanced)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, stratify=Y, test_size=0.01)
    print(f"TRAIN DATA = {len(X_train)}")
    # MODEL
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=estimators)
    rf.fit(X_train, Y_train)

    # evaluation
    score = rf.score(X_val, Y_val)
    data_0_index = np.where(Y_val == 0)[0]
    type_1 = 1 - rf.score(X_val[data_0_index], Y_val[data_0_index])
    data_1_index = np.where(Y_val == 1)[0]
    type_2 = 1 - rf.score(X_val[data_1_index], Y_val[data_1_index])

    print(f"RF classification validation accuracy = {score}")
    print(f"RF classification validation type I error = {type_1}")
    print(f"RF classification validation type II error = {type_2}")

    # feature importance
    feature_importance = pd.Series(rf.feature_importances_, index=features)
    print("Feature importance:\n")
    print(feature_importance)

    # save
    joblib.dump(rf, model_name)
    feature_importance.to_pickle(f"../{save_map}/Info/rf_class_feat_imp_{problem_type}.pickle")

    model_info = pd.Series([instances, len(X_train), class_thresh, score, type_1, type_2, estimators, *before, *after, time.time() - start_time],
                           index=["instances", "datapoints", "class_threshold", "accuracy", "type_1", "type_2", "estimators",
                                  "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9",
                                  "0.9-1", "0", "1", "runtime"])
    model_info.to_pickle(f"../{save_map}/Info/rf_class_info_{problem_type}.pickle")

    return score

# CLEAN AND BALANCE DATA
def clean_data(X, Y, classification=False, class_thresh=False, balanced=False):
    # clean train data
    X = X.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    Y = Y.replace([np.inf, -np.inf], np.nan).dropna()

    # find similar indices
    same_indices = X.index.intersection(Y.index)
    X = X.loc[same_indices]
    Y = Y[same_indices]

    # check the classes
    before = []
    for p in np.arange(0.1, 1.1, 0.1):
        perc = ((Y >= p-0.1) & (Y <= p)).sum()/len(Y)
        before.append(perc)
    print(before)

    if classification:
        if class_thresh:
            Y[Y < class_thresh] = 0
            Y[Y >= class_thresh] = 1
        else:
            # change Y to integers
            Y = Y.astype(int)

        if balanced:
            X["class"] = Y
            g = X.groupby('class')
            g = pd.DataFrame(g.apply(lambda x: x.sample(int(g.size().mean()), replace=True).reset_index(drop=False)))
            X = g
            X.index = X["index"]
            X.drop(["index", "class"], axis=1, inplace=True)
            Y = Y[X.index]

        after = []
        for p in np.arange(0, 2):
            perc = (Y == int(p)).sum() / len(Y)
            after.append(perc)
        print(after)
    return X.to_numpy(), Y.to_numpy(), X, Y, before, after
