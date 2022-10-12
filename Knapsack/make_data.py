from ProblemFunctions.Env import KnapsackEnv
from Method.SucPredData import data_gen_fun_max

import numpy as np
import sys

if __name__ == "__main__":
    array_num = int(sys.argv[1])
    N = int(sys.argv[2])
    gamma_perc = int(sys.argv[3])
    K = int(sys.argv[4])
    time_limit = int(sys.argv[5])
    perc_label = float(sys.argv[6])
    normalized = float(sys.argv[7])

    if len(sys.argv) == 8:
        num_instances = int(sys.argv[7])
    else:
        num_instances = int(np.floor(60/time_limit))

    problem_type = f"ks_p{int(perc_label*100)}_N{N}_g{gamma_perc}_K{K}"

    for i in np.arange((array_num - 1)*num_instances, array_num*num_instances):
        env = KnapsackEnv(N=N, gamma_perc=float(gamma_perc)/100, inst_num=i)
        data_gen_fun_max(K, env, problem_type=problem_type, time_limit=time_limit*60,
                         perc_label=perc_label, normalized=normalized)
