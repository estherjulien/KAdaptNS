from ProblemFunctions.EnvNormal import Graph as Graph_normal
from ProblemFunctions.EnvSphere import Graph as Graph_sphere

from joblib import Parallel, delayed
import numpy as np
import pickle
import sys

"""
MAIN file for creating shortest path instances
INPUT:  num_instances = total number of instances
        N = instance size
        sphere = 1 if instance is of sphere type, 0 otherwise
OUTPUT: a list of environments of Graph (in ShortestPath/ProblemFunctions/Env.py),
        saved in ShortestPath/Data/Instances
"""


if __name__ == "__main__":
    num_instances = int(sys.argv[1])
    N = int(sys.argv[2])
    sphere = int(sys.argv[3])

    if sphere:
        Graph = Graph_sphere
        env_type = "sphere"
    else:
        Graph = Graph_normal
        env_type = "sphere"

    # make environments
    env_list = Parallel(n_jobs=-1)(delayed(Graph)(N=N,
                                                  inst_num=i)
                                   for i in np.arange(num_instances))

    # save per instance
    for i, env in enumerate(env_list):
        with open(f"ShortestPath/Data/Instances/inst_results/sp_env_{env_type}_N{N}_{i}.pickle", "wb") as handle:
            pickle.dump(env, handle)

    # save all
    with open(f"ShortestPath/Data/Instances/sp_env_{env_type}_list_N{N}_{num_instances}.pickle", "wb") as handle:
        pickle.dump(env_list, handle)