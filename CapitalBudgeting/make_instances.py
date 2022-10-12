from ProblemFunctions.Env import ProjectsInstance
from joblib import Parallel, delayed
import numpy as np
import pickle
import sys

"""
MAIN file for creating capital budgeting instances
INPUT:  num_instances = total number of instances
        N = instance size
OUTPUT: a list of environments of ProjectInstance (in CapitalBudgeting/ProblemFunctions/Env.py),
        saved in CapitalBudgeting/Data/Instances
"""


if __name__ == "__main__":
    num_instances = int(sys.argv[1])
    N = int(sys.argv[2])

    env_list = Parallel(n_jobs=-1)(delayed(ProjectsInstance)(N=N, inst_num=i) for i in np.arange(num_instances))

    # save per instance
    for i, env in enumerate(env_list):
        with open(f"CapitalBudgeting/Data/Instances/inst_results/cb_env_N{N}_{i}.pickle", "wb") as handle:
            pickle.dump(env, handle)

    # save all
    with open(f"CapitalBudgeting/Data/Instances/cb_env_list_N{N}_{num_instances}.pickle", "wb") as handle:
        pickle.dump(env_list, handle)

