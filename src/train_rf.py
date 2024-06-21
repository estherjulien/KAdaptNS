from argparse import ArgumentParser
from rf.model import TrainRF

import pandas as pd
import numpy as np
import pickle as pkl

import os
import glob

"""
MAIN file for training a random forest
INPUT:  problem         - benchmark problem 
        N               - instance size
        K               - number of second-stage decisions in training data
        K_train         - value of K used to generate training data
        min_train       - num. minutes spent per instance in generating training data
        h_train         - number of hours spend on generating training data
        ct              - classification threshold = from which probability score on the quality is considered good.
        
        -- scaling 
        sc_pre          - scaling method (dive, alt, no), default dive
        sc_min_max      - min/max scaling of input data, default 0
OUTPUT: - trained random forest model
        - feature importance scores
        - metadata (e.g., data size)
        - combined data set
        saved in src/<benchmark>/data/
"""


def combine_dataset(args, data_inst_path, data_path, features):
    num_inst = int(args.h_train * (60/args.min_train))
    # combine data
    X = {"dive": [], "alt": [], "no": []}
    y = []
    files = glob.glob(data_inst_path)
    print(data_inst_path)
    for i in range(num_inst):
        f = files.pop()
        with open(f, "rb") as handle:
            data = pkl.load(handle)
        for t, x in data["X"].items():
            X[t].append(x)
        y.append(data["Y"])

    X = {t: np.concatenate(x) for t, x in X.items()}
    X = {t: pd.DataFrame(x, columns=features) for t, x in X.items()}
    y = np.concatenate(y)
    with open(data_path, "wb") as handle:  # number of datapoints
        pkl.dump([X, y], handle)


def main(args):
    if "cb" in args.problem:
        data_inst_path = f"src/cb/data/train_data/inst_results/data_results_{args.problem}_N{args.N}_K{args.K}_m{args.min_train}_*"
        data_path = f"src/cb/data/ml_data_{args.problem}_N{args.N}_K{args.K}_min{args.min_train}_nodes{args.h_train}.pkl"
        features = ["obj_node", "obj_pre", "zeta_node", "zeta_pre", "depth",
                    "coords", "obj_det", "x_det", "y_det", "obj_stat", "y_stat", "slack", "const_to_z_dist",
                    "const_to_const_dist"]
    elif "sp" in args.problem:
        data_inst_path = f"src/sp/data/train_data/inst_results/data_results_{args.problem}_N{args.N}_K{args.K}_m{args.min_train}_*"
        data_path = f"src/sp/data/ml_data_{args.problem}_N{args.N}_K{args.K}_min{args.min_train}_nodes{args.h_train}.pkl"
        features = ["obj_node", "obj_pre", "zeta_node", "zeta_pre", "depth",
                    "coords", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist"]
    if not os.path.exists(data_path):
        combine_dataset(args, data_inst_path, data_path, features)

    with open(data_path, "rb") as handle:
        data_sets, y = pkl.load(handle)

    X = data_sets[args.sc_pre]

    model = TrainRF(X, y, args.problem, args.N, args.K, args.min_train, args.h_train, ct=args.ct,
                    sc_pre=args.sc_pre, sc_min_max=args.sc_min_max)
    model.train_rf()


if __name__ == '__main__':
    parser = ArgumentParser()
    # about data
    parser.add_argument('--problem', type=str, default="cb")
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--K', type=int, default=6)
    parser.add_argument('--min_train', type=int, default=10)
    parser.add_argument('--h_train', type=int, default=2)
    parser.add_argument('--ct', type=int, default=5)

    # scaling
    parser.add_argument('--sc_pre', type=str, default="dive", choices=["dive", "alt", "no"])
    parser.add_argument('--sc_min_max', type=int, default=0)

    args = parser.parse_args()

    main(args)
