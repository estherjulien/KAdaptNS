from Method.SucPred import algorithm
import pickle
import sys

"""
MAIN file for K-B&B-NodeSelection for capital budgeting (node selection with ML in K-adaptability branch-and-bound)
INPUT:  -- instance parameters
        i = instance number
        N = instance size
        K = number of second-stage decisions
        -- ML model parameters
        K_ML = value of K used to generate training data
        minutes = num. minutes spend per instance in generating training data
        hours = number of hours spend on generating training data
        ct = classification threshold - from which probability score on the quality is considered good.
        -- K-B&B-NodeSelection parameters
        max_level = maximum level up to where ML model is applied for node selection. from then on random
OUTPUT: solution of K-B&B-NodeSelection
        saved in CapitalBudgeting/Data/Results/Decisions
"""

if __name__ == "__main__":
    # INSTANCE PARAMETERS
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    # ML PARAMETERS
    K_ML = int(sys.argv[4])
    minutes = int(sys.argv[5])
    hours = int(sys.argv[6])
    ct = int(sys.argv[7])
    # MODEL PARAMETERS
    max_level = int(sys.argv[8])

    if max_level == 0:
        max_level = None

    # load environment
    with open(f"CapitalBudgeting/Data/Instances/inst_results/cb_env_N{N}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    success_model_name = f"CapitalBudgeting/Data/Models/rf_class_cb_N10_K{K_ML}_min{minutes}_nodes{hours}_ct{ct}_bal.joblib"

    # run algorithm
    problem_type = f"cb_suc_pred_ML[N10_K{K_ML}_m{minutes}_nodes{hours}_ct{ct}]_T[N{N}_K{K}]_L{max_level}"
    algorithm(K, env, max_level=max_level, problem_type=problem_type, success_model_name=success_model_name)

