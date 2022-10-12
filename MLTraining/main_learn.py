from learn_models import *
import pickle
import sys


"""
MAIN file for training a random forest
INPUT:  N = instance size
        K = number of second-stage decisions (or subsets)
        train_sp = data will be opened of shortest path (sphere) data if 1, otherwise data of capital budgeting.
OUTPUT: trained random forest with metadata on e.g. accuracy
        model saved in {problem type}/Data/Models
        metadata saved in {problem type}/Data/Models/Info
"""


def train_model(N, K, train_sp=False, ct=0.05, minutes=5, nodes=2, balanced=True):
    if train_sp:
        att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist",
                      "const_to_const_dist"]
        save_name = "sp_sphere"
        save_map = "ShortestPath/Data/Models"
    else:
        att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
                      "const_to_const_dist"]
        save_name = "cb"
        save_map = "CapitalBudgeting/Data/Models"

    # define features
    features = ["theta_node", "theta_pre", "zeta_node", "zeta_pre", "depth", *att_series]

    # GET TRAINING DATA
    with open(f"../CBData/train_data_cb_N10_K{K}_min{minutes}_nodes{nodes}.pickle", "rb") as handle:
        res = pickle.load(handle)
    X = res["X"]
    Y = res["Y"]

    # CLASSIFICATION
    problem_type = f"{save_name}_N{N}_K{K}_min{minutes}_nodes{nodes}_ct{int(ct*100)}"
    print(f"nodes = {nodes}, minutes = {minutes}, K = {K}, N = {N}, ct = {ct}, balanced = {balanced}")
    if balanced:
        problem_type += "_bal"
    df_X = pd.DataFrame(X)
    df_Y = pd.Series(Y)
    train_suc_pred_rf_class(df_X, df_Y, features, problem_type=problem_type, save_map=save_map, class_thresh=ct,
                            balanced=balanced)


if __name__ == "__main__":
    N = int(sys.argv[1])
    K = int(sys.argv[2])
    train_sp = True if int(sys.argv[3]) else False

    train_model(N, K, train_sp=train_sp)
