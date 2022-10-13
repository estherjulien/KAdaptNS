from Method.Random import algorithm
import pickle
import sys

"""
MAIN file for K-B&B (random depth first K-adaptability branch-and-bound)
INPUT:  i = instance number
        N = instance size
        K = number of second-stage decisions
OUTPUT: solution of K-B&B
        saved in CapitalBudgeting/Data/Results/Decisions
"""

if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    # load environment
    with open(f"CapitalBudgeting/Data/Instances/inst_results/cb_env_N{N}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    # run random algorithm
    problem_type = f"cb_random_N{N}_K{K}"
    algorithm(K, env, problem_type=problem_type)

