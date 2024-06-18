from multiprocessing import Process
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

"""
MAIN file for K-B&B-NodeSelection for a benchmark problem
(ML guided node selection in K-adaptability branch-and-bound algorithm)
INPUT:  -- instance parameters
        problem         - benchmark problem 
        inst_num        - instance id
        N               - instance size
        K               - number of second-stage decisions
        time_limit      - time limit of algorithm (in minutes)
        -- ML model parameters
        K_train         - value of K used to generate training data
        min_train       - num. minutes spent per instance in generating training data
        h_train         - number of hours spend on generating training data
        ct              - classification threshold = from which probability score on the quality is considered good.
        
        -- K-B&B-NodeSelection parameters
        max_level = maximum level up to where ML model is applied for node selection. from then on random

OUTPUT: solution of K-B&B-NodeSelection
        saved in src/<benchmark>/data/results/ml/
"""


def main(args, i=None):
    if args.max_level == -1:
        args.max_level = None

    if args.problem == "cb":
        from cb.method.ml import algorithm
        from cb.problem_functions.environment import ProjectsInstance as Env
        ml_name = f"src/cb/data/ml_model_cb_N10_K{args.K_train}_min{args.min_train}_nodes{args.h_train}_ct{args.ct}_bal_" \
                  f"scp-{args.sc_pre}_scmm{args.sc_min_max}.joblib"
    elif "sp" in args.problem:
        from sp.method.ml import algorithm
        from sp.problem_functions.environment import Graph as Env
        ml_name = f"src/sp/data/ml_model_{args.problem}_N10_K{args.K_train}_min{args.min_train}_nodes{args.h_train}_" \
                  f"ct{args.ct}_bal_scp-{args.sc_pre}_scmm{args.sc_min_max}.joblib"
    else:
        raise "ML node selection for this problem type not implemented"

    problem_type = f"{args.problem}_ml_" \
                   f"ML[N10_K{args.K_train}_m{args.min_train}_nodes{args.h_train}_ct{args.ct}_" \
                   f"scp-{args.sc_pre}_scmm{args.sc_min_max}]_" \
                   f"T[N{args.N}_K{args.K}]_L{args.max_level}"
    if i is not None:
        args.inst_num = i
    # load test environment
    env = Env(args.problem, args.N, args.inst_num)
    env.read_test_inst()

    # run algorithm
    algorithm(args.K, env, max_level=args.max_level, problem_type=problem_type, success_model_name=ml_name,
              time_limit=args.time_limit*60, sc_pre=args.sc_pre, sc_min_max=args.sc_min_max)


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
    parser.add_argument('--problem', type=str, default='cb', choices=["cb", "sp_sphere"])
    parser.add_argument('--inst_num', type=int, default=1)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--time_limit', type=int, default=30)

    # ML params
    parser.add_argument('--K_train', type=int, default=6)
    parser.add_argument('--min_train', type=int, default=10)
    parser.add_argument('--h_train', type=int, default=2)
    parser.add_argument('--ct', type=int, default=5)
    parser.add_argument('--sc_pre', type=str, default="dive", choices=["dive", "alt", "no"])
    parser.add_argument('--sc_min_max', type=int, default=0)

    # K-B&B-NodeSelection param
    parser.add_argument('--max_level', type=int, default=-1)

    # PARALLEL RUN params
    parser.add_argument('--parallel', type=int, default=0)
    parser.add_argument('--n_procs', type=int, default=8)
    parser.add_argument('--job_num', type=int, default=1)
    args = parser.parse_args()

    if args.parallel:
        main_parallel(args)
    else:
        main(args)

