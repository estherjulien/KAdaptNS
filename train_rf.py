from argparse import ArgumentParser
from rf.model import TrainRF


import pandas as pd
import numpy as np
import pickle as pkl

import os
import glob


def combine_dataset(args, data_inst_path, data_path, features):
    num_inst = int(args.h_train * (60/args.min_train))
    # combine data
    X = {"dive": [], "alt": [], "no": []}
    y = []
    files = glob.glob(data_inst_path)
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
        data_path = "cb/data/local_results/old_data/train_data_cb_N10_K6_min10_nodes2.pickle"

        # data_inst_path = f"cb/data/train_data/inst_results/data_results_{args.problem}_N{args.N}_K{args.K}_m{args.min_train}_*"
        # data_path = f"cb/data/ml_data_{args.problem}_N{args.N}_K{args.K}_min{args.min_train}_nodes{args.h_train}.pkl"
        features = ["obj_node", "obj_pre", "zeta_node", "zeta_pre", "depth",
                    "coords", "obj_det", "x_det", "y_det", "obj_stat", "y_stat", "slack", "const_to_z_dist",
                    "const_to_const_dist"]
    elif "sp" in args.problem:
        data_inst_path = f"sp/data/train_data/inst_results/data_results_{args.problem}_N{args.N}_K{args.K}_m{args.min_train}_*"
        data_path = f"sp/data/ml_data_{args.problem}_N{args.N}_K{args.K}_min{args.min_train}_nodes{args.h_train}.pkl"
        features = ["obj_node", "obj_pre", "zeta_node", "zeta_pre", "depth",
                    "coords", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist"]
    if not os.path.exists(data_path):
        combine_dataset(args, data_inst_path, data_path, features)

    # with open(data_path, "rb") as handle:
    #     data_sets, y = pkl.load(handle)
    with open(data_path, "rb") as handle:
        data = pkl.load(handle)

    X = pd.DataFrame(data["X"], columns=features)
    y = data["Y"]
    if args.problem == "cb_1":
        # delete objective and violation difference
        X = X.drop(["obj_pre", "zeta_pre"], axis=1)
    elif args.problem == "cb_2":
        # delete objective and violation
        X = X.drop(["obj_node", "zeta_node"], axis=1)
    elif args.problem == "cb_3":
        # delete all static
        X = X.drop(["obj_stat", "y_stat"], axis=1)
    elif args.problem == "cb_4":
        # delete all deterministic
        X = X.drop(["obj_det", "x_det", "y_det"], axis=1)
    elif args.problem == "cb_5":
        # delete all k-dependent features
        X = X.drop(["slack", "const_to_z_dist", "const_to_const_dist"], axis=1)
    else:
        pass

    # X = data_sets[args.sc_pre]
    model = TrainRF(X, y, args.problem, args.N, args.K, args.min_train, args.h_train, ct=args.ct,
                    sc_pre=args.sc_pre, sc_min_max=args.sc_min_max)
    model.train_rf()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default="cb")
    # about data
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--K', type=int, default=6)
    parser.add_argument('--min_train', type=int, default=10)
    parser.add_argument('--h_train', type=int, default=2)
    parser.add_argument('--ct', type=int, default=5)

    parser.add_argument('--sc_pre', type=str, default="dive", choices=["dive", "alt", "no"])
    parser.add_argument('--sc_min_max', type=int, default=0)

    args = parser.parse_args()

    main(args)
