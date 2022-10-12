from Method.Random import algorithm
import pickle
import sys

"""
MAIN file for K-B&B (random depth first K-adaptability branch-and-bound)
INPUT:  i = instance number
        N = instance size
        K = number of second-stage decisions
        sphere = 1 if instance is of sphere type, 0 otherwise
OUTPUT: solution of K-B&B
        saved in ShortestPath/Data/Results/Decisions
"""

if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    sphere = int(sys.argv[4])

    if sphere:
        env_type = "sphere"
    else:
        env_type = "normal"

    # load environment
    with open(f"ShortestPath/Data/Instances/inst_results/sp_env_{env_type}_N{N}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    # run random algorithm
    problem_type = f"sp_random_{env_type}_N{N}_K{K}"
    algorithm(K, env, problem_type=problem_type, time_limit=30 * 60)
