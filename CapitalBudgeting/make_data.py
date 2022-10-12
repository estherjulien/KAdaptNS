from ProblemFunctions.Env import ProjectsInstance
from Method.SucPredData import data_gen_fun_max

import numpy as np
import sys

"""
MAIN file for creating training data for the capital budgeting problem
INPUT:  job_num = number of job (for parallelization purposes)
        N = instance size
        K = number of second-stage decisions (or subsets)
        time_limit = seconds spent per instance
OUTPUT: training data
        data saved in CapitalBudgeting/Data/Results/TrainData 
        metadata saved in CapitalBudgeting/Data/RunInfo 
"""

if __name__ == "__main__":
    job_num = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    time_limit = int(sys.argv[4])

    if len(sys.argv) == 6:
        num_instances = int(sys.argv[5])
    else:
        num_instances = int(np.floor(60/time_limit))

    # attributes used for capital budgeting
    att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
                  "const_to_const_dist"]

    problem_type = f"cb_N{N}_K{K}_m{time_limit}"

    for i in np.arange((job_num - 1)*num_instances, job_num*num_instances):
        # make an instance of the capital budgeting problem
        env = ProjectsInstance(N=N, inst_num=i)
        # generate data for this instance
        data_gen_fun_max(K, env, att_series=att_series, problem_type=problem_type, time_limit=time_limit*60)
