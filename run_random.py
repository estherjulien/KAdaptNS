from multiprocessing import Process
from joblib import Parallel, delayed
from argparse import ArgumentParser
import time

"""
MAIN file for K-B&B (random depth first K-adaptability branch-and-bound)
INPUT:  i = instance number
        N = instance size
        K = number of second-stage decisions
OUTPUT: solution of K-B&B
        saved in cb/data/results/decisions
"""


def main(args, i=None):
    if args.problem == "cb":
        from cb.method.random import algorithm
        from cb.problem_functions.environment import ProjectsInstance as Env
        # load environment
    elif "sp" in args.problem:
        from sp.method.random import algorithm
        from sp.problem_functions.environment import Graph as Env

    problem_type = f"cb_random_N{args.N}_K{args.K}"
    if i is not None:
        args.inst_num = i
    # load test environment
    env = Env(args.problem, args.N, args.inst_num)
    env.read_test_inst()

    # run random algorithm
    algorithm(args.K, env, problem_type=problem_type, time_limit=args.time_limit*60)


def main_parallel(args):
    s_start = (args.job_num - 1)*args.n_procs + 1
    s_end = args.job_num*args.n_procs + 1
    # Parallel(n_jobs=args.n_procs)(delayed(main)(args, i) for i in range(s_start, s_end))
    # start processes
    procs = []
    seed_ids = range(s_start, s_end)
    print(seed_ids)
    for i in seed_ids:
        proc = Process(target=main, args=[args, i])
        proc.start()
        procs.append(proc)
    # collect finished processes
    for proc in procs:
        proc.join()
        proc.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default='cb')
    parser.add_argument('--inst_num', type=int, default=1)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--time_limit', type=int, default=30)

    # PARALLEL RUN params
    parser.add_argument('--parallel', type=int, default=0)
    parser.add_argument('--n_procs', type=int, default=8)
    parser.add_argument('--job_num', type=int, default=1)
    args = parser.parse_args()

    if args.parallel:
        main_parallel(args)
    else:
        main(args)