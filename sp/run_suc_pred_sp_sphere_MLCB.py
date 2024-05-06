from Method.SucPred import algorithm
import pickle
import sys

"""
MAIN file for K-B&B-NodeSelection for shortest path with a capital budgeting-trained ML model
INPUT:  -- instance parameters
        i = instance number
        N = instance size
        K = number of second-stage decisions
        -- ML model parameters
        K_ML = value of K used to generate training data
OUTPUT: solution of K-B&B-NodeSelection
        saved in ShortestPath/Data/Results/Decisions
"""


if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    K_ML = int(sys.argv[4])

    ct = 5
    hours = 2
    max_level = None

    if K_ML == 2:
        m = 2
    elif K_ML in [3, 4, 5]:
        m = 5
    else:
        m = 10

    # load environment
    with open(f"ShortestPath/Data/Instances/inst_results/sp_env_sphere_N{N}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    success_model_name = f"CapitalBudgeting/Data/Models/rf_class_cb_no_fs_N10_K{K_ML}_min{m}_nodes{hours}_ct{ct}_bal.joblib"

    # run algorithm
    problem_type = f"sp_sphere_suc_pred_ML[CB_N10_K{K_ML}_m{m}_nodes{hours}_ct{ct}]_T[N{N}_K{K}]_L{max_level}"
    algorithm(K, env, max_level=max_level, problem_type=problem_type, success_model_name=success_model_name)

