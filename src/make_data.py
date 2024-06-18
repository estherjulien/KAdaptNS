from argparse import ArgumentParser
import numpy as np

"""
MAIN file for creating training data for a benchmark problem
INPUT:  problem         - benchmark problem 
        job_num         - number of job (for parallelization purposes)
        N               - instance size
        K               - number of second-stage decisions (or subsets)
        time_limit      - minutes spent per instance
        num_instances   - number of instances for generating data
OUTPUT: training data
        data saved in src/<benchmark>/data/train_data/inst_results/ 
        metadata saved in src/<benchmark>/data/train_data/inst_results_md/
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default='cb', choices=["cb", "sp_sphere"])
    parser.add_argument('--job_num', type=int, default=1)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--time_limit', type=int, default=10)
    parser.add_argument('--num_instances', type=int, default=-1)
    args = parser.parse_args()

    if args.num_instances == -1:
        args.num_instances = int(np.floor(60/args.time_limit))

    problem_type = f"{args.problem}_N{args.N}_K{args.K}_m{args.time_limit}"

    if args.problem == "cb":
        from cb.problem_functions.environment import ProjectsInstance as Env
        from cb.method.data_gen import data_gen_fun_max as data_gen_fun
        att_series = ["coords", "obj_det", "x_det", "y_det", "obj_stat", "y_stat", "slack", "const_to_z_dist",
                      "const_to_const_dist"]
    elif "sp" in args.problem:
        from sp.method.data_gen import data_gen_fun
        from sp.problem_functions.environment import Graph as Env
        att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist"]

    rng = np.random.RandomState(args.job_num)
    for i in range((args.job_num - 1)*args.num_instances+1, args.job_num*args.num_instances+1):
        # make an instance of the capital budgeting problem
        inst_num = rng.randint(120, 6000)
        env = Env(args.problem, N=args.N, inst_num=inst_num)
        env.make_test_inst(save_env=False)
        # generate data for this instance
        data_gen_fun(args.K, env, att_series=att_series, problem_type=problem_type, time_limit=args.time_limit*60, dg_inst=i)
