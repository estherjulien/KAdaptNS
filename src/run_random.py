from multiprocessing import Process
from argparse import ArgumentParser

"""
MAIN file for K-B&B (random depth first K-adaptability branch-and-bound)
INPUT:  -- instance parameters
        problem         - benchmark problem 
        inst_num        - instance id
        N               - instance size
        K               - number of second-stage decisions
        time_limit      - time limit of algorithm (in minutes)
        
        -- knapsack parameters
        kp_g    - knapsack parameter gamma (uncertainty)
        kp_b    - knapsack parameter c (for budget capacity)
        
OUTPUT: solution of K-B&B
        saved in src/<benchmark>/data/results/random/
"""


def main(args, i=None):
    # load environment
    if "cb" in args.problem:
        # from src.cb.method.random import algorithm
        from cb.problem_functions.environment import ProjectsInstance as Env
        from cb.method.random import algorithm
        env = Env(args.problem, args.N, args.inst_num)
        problem_type = f"{args.problem}_random_N{args.N}_K{args.K}"
    elif "sp" in args.problem:
        # from src.sp.method.random import algorithm
        from sp.problem_functions.environment import Graph as Env
        from sp.method.random import algorithm
        env = Env(args.problem, args.N, args.inst_num)
        problem_type = f"{args.problem}_random_N{args.N}_K{args.K}"
    elif "kp" in args.problem:
        # from src.kp.method.random import algorithm
        from kp.problem_functions.environment import KnapsackEnv as Env
        from kp.method.random import algorithm
        env = Env(args.problem, args.N, args.kp_g, args.kp_b, args.inst_num)
        problem_type = f"{args.problem}_random_N{args.N}_g{args.kp_g}_b{args.kp_b}_K{args.K}"
    else:
        raise "Not implemented for other problems"

    if i is not None:
        args.inst_num = i
    # load test environment
    env.read_test_inst()

    # run random algorithm
    algorithm(args.K, env, problem_type=problem_type, time_limit=args.time_limit*60)


def main_parallel(args):
    s_start = (args.job_num - 1)*args.n_procs + 1
    s_end = args.job_num*args.n_procs + 1
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
    parser.add_argument('--problem', type=str, default='cb', choices=["cb", "sp_normal", "sp_sphere", "kp"])
    parser.add_argument('--inst_num', type=int, default=1)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--time_limit', type=int, default=30)

    # knapsack parameters
    parser.add_argument('--kp_g', type=int, default=5)
    parser.add_argument('--kp_b', type=int, default=5)

    # PARALLEL RUN params
    parser.add_argument('--parallel', type=int, default=0)
    parser.add_argument('--n_procs', type=int, default=8)
    parser.add_argument('--job_num', type=int, default=1)
    args = parser.parse_args()

    if args.parallel:
        main_parallel(args)
    else:
        main(args)