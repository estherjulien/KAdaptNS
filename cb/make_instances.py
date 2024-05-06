import os

from problem_functions.environment import ProjectsInstance
from joblib import Parallel, delayed

from argparse import ArgumentParser
import pickle as pkl

"""
MAIN file for creating capital budgeting instances
INPUT:  num_instances = total number of instances
        N = instance size
OUTPUT: a list of environments of ProjectInstance (in CapitalBudgeting/ProblemFunctions/Env.py),
        saved in CapitalBudgeting/Data/Instances
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--N', type=int, default=10)
    args = parser.parse_args()

    env_list = [ProjectsInstance("cb", N=args.N, inst_num=i) for i in range(1, args.num_instances+1)]
    Parallel(n_jobs=-1)(delayed(env.make_test_inst)() for env in env_list)

