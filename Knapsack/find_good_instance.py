from ProblemFunctions.Env import KnapsackEnv
from Method.Random import algorithm

import pickle
import sys


if __name__ == "__main__":
    N = 100
    K = 4
    array_job = int(sys.argv[1])
    inst_num = int(sys.argv[2])
    budget_perc, gamma_perc = [(b, g) for b in [5, 10, 25, 50] for g in [5, 15, 25, 50]][array_job - 1]

    # test_results = pd.DataFrame(index=[(budget_perc, gamma_perc, inst_num)], columns=["obj_diff", "num_robust_sols", "runtime"], dtype=float)
    # test_results.index = pd.MultiIndex.from_tuples(test_results.index)
    # open instance
    with open(f"Knapsack/Data/Instances/inst_results/ks_test_env_N{N}_g{gamma_perc}_b{budget_perc}_{inst_num}.pickle", "rb") as handle:
        env = pickle.load(handle)

    # ALGORITHM
    problem_type = f"ks_test_random_N{N}_b{budget_perc}_g{gamma_perc}_K{K}"
    results = algorithm(K, env, problem_type=problem_type, time_limit=20*60)
    init_obj = list(results["inc_thetas_t"].values())[0]

    # test_results.loc[(budget_perc, gamma_perc, inst_num)] = [init_obj/results["theta"], len(results["inc_thetas_t"]), results["runtime"]]
    # test_results.to_pickle(f"Knapsack/df_test_results_N{N}_K{K}_b{budget_perc}_g{gamma_perc}_{inst_num}.pickle")
