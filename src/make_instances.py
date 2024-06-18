from joblib import Parallel, delayed
from argparse import ArgumentParser

"""
MAIN file for creating benchmark problem test instances
INPUT:  problem         - benchmark problem 
        num_instances   - total number of instances
        N               - instance size
OUTPUT: instances for the benchmark problem
        saved in src/<benchmark>/data/instances/
"""


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str)
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--N', type=int, default=10)
    args = parser.parse_args()

    if args.problem == "cb":
        from cb.problem_functions.environment import ProjectsInstance as Env
    elif "sp" in args.problem:
        from sp.problem_functions.environment import Graph as Env
    elif "kp" in args.problem:
        from kp.problem_functions.environment import KnapsackEnv as Env

    env_list = [Env(args.problem, N=args.N, inst_num=i) for i in range(1, args.num_instances+1)]
    Parallel(n_jobs=-1)(delayed(env.make_test_inst)() for env in env_list)
