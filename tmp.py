import pickle as pkl
import pandas as pd
import numpy as np
import glob


def combine_dataset(num_inst):
    # combine data
    X = []
    y = []
    cols = ["obj_node", "obj_pre", "zeta_node", "zeta_pre", "depth",
            "coords", "obj_det", "x_det", "y_det",
            "obj_stat", "y_stat", "slack", "const_to_z_dist", "const_to_const_dist"]
    files = glob.glob("CapitalBudgeting/Data/Results/TrainData/inst_results/data_results_cb_N10_K6_m10_*")
    for i in range(num_inst):
        f = files.pop()
        with open(f, "rb") as handle:
            data = pkl.load(handle)
        X.append(pd.DataFrame(data["X"], columns=cols))
        y.append(data["Y"])

    X = pd.concat(X, ignore_index=True)
    y = np.concatenate(y)
    with open("ml_data_l-1_ntr-10_Ktr-6_ntr-12_old.pkl", "wb") as handle:  # number of datapoints
        pkl.dump([X, y], handle)


if __name__ == "__main__":
    combine_dataset(12)