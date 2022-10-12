from Method.SucPred import algorithm
import pickle
import sys

"""
MAIN file for K-B&B-NodeSelection for capital budgeting with a shortest-path-trained ML model
INPUT:  -- instance parameters
        i = instance number
        N = instance size
        K = number of second-stage decisions
        -- ML model parameters
        K_ML = value of K used to generate training data
OUTPUT: solution of K-B&B-NodeSelection
        saved in CapitalBudgeting/Data/Results/Decisions
"""

if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    K_ML = int(sys.argv[4])

    if K_ML == 2:
        m = 15
        T = 10
    elif K_ML == 3:
        m = 15
        T = 5
    elif K_ML == 4:
        m = 20
        T = 5
    elif K_ML == 5:
        m = 20
        T = 10
    elif K_ML == 6:
        m = 15
        T = 10
    ct = 5
    max_level = 40

    # load environment
    with open(f"CapitalBudgeting/Data/Instances/inst_results/cb_env_N{N}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    success_model_name = f"ShortestPath/Data/Models/rf_class_sp_sphere_N20_K{K_ML}_min{m}_nodes{T}_ct{ct}_bal.joblib"

    att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist"]
    # run algorithm with threshold
    problem_type = f"cb_suc_pred_ML[SP_N20_K{K_ML}_m{m}_nodes{T}_ct{ct}]_T[N{N}_K{K}]_L{max_level}"
    algorithm(K, env, max_level=max_level, problem_type=problem_type, success_model_name=success_model_name,
              att_series=att_series)

