from Method.SucPred import algorithm
import pickle
import sys


if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    gamma_perc = int(sys.argv[3])
    K = int(sys.argv[4])
    max_level = int(sys.argv[5])
    thresh = int(sys.argv[6])

    # load environment
    with open(f"Knapsack/Data/Instances/inst_results/ks_test_env_N{N}_g{gamma_perc}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    success_model_name = f"Knapsack/Data/Models/rf_class_ks_p5_N10_K4_ct70_all.joblib"

    # run algorithm with threshold
    if thresh:
        problem_type = f"ks_suc_pred_rf_p5_N{N}_g{gamma_perc}_K{K}_L{max_level}"
        algorithm(K, env, max_level=max_level, problem_type=problem_type, success_model_name=success_model_name, thresh=0.1)
    else:
        problem_type = f"ks_suc_pred_rf_p5_nt_N{N}_g{gamma_perc}_K{K}_L{max_level}"
        algorithm(K, env, max_level=max_level, problem_type=problem_type, success_model_name=success_model_name)

