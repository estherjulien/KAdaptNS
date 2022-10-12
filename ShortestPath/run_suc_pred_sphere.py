from Method.SucPred import algorithm
import pickle
import sys

"""
MAIN file for K-B&B-NodeSelection for shortest path (node selection with ML in K-adaptability branch-and-bound)
INPUT:  -- instance parameters
        i = instance number
        N = instance size
        K = number of second-stage decisions
        -- ML model parameters
        K_ML = value of K used to generate training data
        hours = number of hours spend on generating training data
        -- K-B&B-NodeSelection parameters
        max_level = maximum level up to where ML model is applied for node selection. from then on random
OUTPUT: solution of K-B&B-NodeSelection
        saved in ShortestPath/Data/Results/Decisions
"""


if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    K_ML = int(sys.argv[4])
    hours = int(sys.argv[5])

    if hours == 0:
        hours = None

    if len(sys.argv) == 7:
        max_level = int(sys.argv[6])
        if max_level == 0:
            max_level = None
    else:
        max_level = None

    ct = 5

    if K_ML == 2:
        minutes = 15
        if hours is None:
            nodes = 10
    elif K_ML == 3:
        minutes = 15
        if hours is None:
            nodes = 5
    elif K_ML == 4:
        minutes = 20
        if hours is None:
            nodes = 5
    elif K_ML == 5:
        minutes = 20
        if hours is None:
            nodes = 10
    else:
        minutes = 15
        if hours is None:
            nodes = 10

    # load environment
    with open(f"ShortestPath/Data/Instances/inst_results/sp_env_sphere_N{N}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    success_model_name = f"ShortestPath/Data/Models/rf_class_sp_sphere_N20_K{K_ML}_min{minutes}_nodes{hours}_ct{ct}_bal.joblib"

    # run algorithm
    problem_type = f"sp_sphere_suc_pred_ML[N20_K{K_ML}_m{minutes}_nodes{hours}_ct{ct}]_T[N{N}_K{K}]_L{max_level}"
    algorithm(K, env, max_level=max_level, problem_type=problem_type, success_model_name=success_model_name)

