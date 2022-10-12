from Method.Random import algorithm
import pickle
import sys

if __name__ == "__main__":
    array_num = int(sys.argv[1])
    i = int(sys.argv[2])
    N = int(sys.argv[3])
    K = int(sys.argv[4])

    budget_perc, gamma_perc = [(b, g) for b in [5, 15, 35, 50] for g in [5, 15, 35, 50]][array_num-1]

    # load environment
    with open(f"Knapsack/Data/Instances/inst_results/NEW_ks_env_N{N}_b{budget_perc}_g{gamma_perc}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    # run random algorithm
    problem_type = f"NEW_ks_random_N{N}_b{budget_perc}_g{gamma_perc}_K{K}"
    algorithm(K, env, problem_type=problem_type)

