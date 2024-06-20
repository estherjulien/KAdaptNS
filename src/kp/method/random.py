import os

from kp.problem_functions.functions_milp import *

from datetime import datetime
import gurobipy as gp
import numpy as np
import pickle
import copy
import time

"""
Code for running K-B&B for solving the knapsack problem
(random depth node selection for K-adaptability branch and bound)

INPUT:  K = number of second-stage decisions (or subsets)
        env = instance of the capital budgeting problem
        time_limit = seconds spend per instance - if limit is reached, 
                     the incumbent solution will be used as final solution
OUTPUT: solution to capital budgeting problem
        saved in src/kp/data/results/random/
"""


def algorithm(K, env, time_limit=30*60, print_info=True, problem_type="test"):
    rng = np.random.RandomState(env.inst_num)

    gp_env = gp.Env()
    gp_env.setParam("OutputFlag", 0)
    gp_env.setParam("Threads", 1)
    # Initialize
    iteration = 0
    start_time = time.time()
    # initialization for saving
    inc_thetas_t = dict()
    inc_thetas_n = dict()
    inc_tau = dict()
    inc_x = dict()
    inc_y = dict()
    prune_count = 0
    inc_tot_nodes = dict()
    tot_nodes = 0
    inc_tot_nodes[0] = 0
    mp_time = 0
    sp_time = 0
    # initialization of lower and upper bounds
    theta_i, x_i, y_i = (env.upper_bound, [], [])
    # K-branch and bound algorithm
    now = datetime.now().time()
    xi_new, k_new = None, None

    new_model = True
    # initialize N_set with actual scenario
    tau_i, scen_all = init_k_adapt(K, env, gp_env)
    N_set = [tau_i]

    new_xi_num = 0

    print(f"Instance R {env.inst_num}: started at {now}")
    while (N_set or k_new is not None) and time.time() - start_time < time_limit:
        # MASTER PROBLEM
        if new_model:
            tot_nodes += 1
            # take new node
            new_pass = rng.randint(len(N_set))
            placement = N_set.pop(new_pass)
            tau = {k: scen_all[placement[k]] for k in range(K)}
            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_build(K, tau, env, gp_env)
            mp_time += time.time() - start_mp
        else:
            # make new tau from k_new
            tot_nodes += 1

            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)
            mp_time += time.time() - start_mp

            placement[k_new].append(new_xi_num)
            tau = {k: scen_all[placement[k]] for k in range(K)}

        # prune if theta higher than current robust theta
        if theta - theta_i < 1e-8:
            prune_count += 1
            new_model = True
            continue

        # SUBPROBLEM
        start_sp = time.time()
        zeta, xi = separation_fun(K, x, y, theta, env, tau, gp_env)
        sp_time += time.time() - start_sp

        k_new = None
        # check if robust
        if zeta <= 1e-04:
            if print_info:
                now = datetime.now().time()
                print("Instance R {}: ROBUST at iteration {} ({}) (time {})   :obj = {},    violation = {}".format(
                    env.inst_num, iteration, np.round(time.time()-start_time, 3), now, np.round(theta, 4), np.round(zeta, 4)))
            try:
                env.plot_graph_solutions(K, y, tau, x=x, alg_type=problem_type, tmp=True, it=iteration)
            except:
                pass

            theta_i, x_i, y_i = (copy.deepcopy(theta), copy.deepcopy(x), copy.deepcopy(y))
            tau_i = copy.deepcopy(tau)
            inc_thetas_t[time.time() - start_time] = theta_i
            inc_thetas_n[tot_nodes] = theta_i
            inc_tau[time.time() - start_time] = tau_i
            inc_x[time.time() - start_time] = x_i
            inc_y[time.time() - start_time] = y_i
            prune_count += 1
            new_model = True
            continue
        else:
            new_model = False
            new_xi_num += 1
            scen_all = np.vstack([scen_all, xi])

        full_list = [k for k in range(K) if len(tau[k]) > 0]
        if len(full_list) == 0:
            K_set = [0]
            k_new = 0
        elif len(full_list) == K:
            K_set = np.arange(K)
            k_new = rng.randint(K)
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
            k_new = K_set[-1]

        for k in K_set:
            if k == k_new:
                continue
            # add to node set
            placement_tmp = copy.deepcopy(placement)
            placement_tmp[k].append(new_xi_num)
            N_set.append(placement_tmp)

        iteration += 1
    # termination results
    runtime = time.time() - start_time
    inc_thetas_t[time.time() - start_time] = theta_i
    inc_thetas_n[tot_nodes] = theta_i

    now = datetime.now().time()
    now_nice = f"{now.hour}:{now.minute}:{now.second}"
    print(f"Instance R {env.inst_num}, completed at {now_nice}, solved in {np.round(runtime/60, 3)} minutes")
    results = {"obj": theta_i, "inc_obj_time": inc_thetas_t,
               "inc_obj_nodes": inc_thetas_n, "runtime": time.time() - start_time,
               "tot_nodes": tot_nodes, "mp_time": mp_time, "sp_time": sp_time}

    os.makedirs("src/kp/data/results/random", exist_ok=True)
    with open(f"src/kp/data/results/random/final_results_{problem_type}_s{env.inst_num}.pkl", "wb") as handle:
        pickle.dump(results, handle)

    return results


# Instead of using the scenario of all zeros (if this is in the uncertainty set), another initial scenario is sought.
def init_k_adapt(K, env, gp_env):
    tau = {k: [] for k in range(K)}
    tau[0].append(env.init_uncertainty)

    # run master problem
    theta, x, y, _ = scenario_fun_build(K, tau, env, gp_env)

    # run sub problem
    _, xi_new = separation_fun(K, x, y, theta, env, tau, gp_env)

    # new tau to be saved in N_set
    tau = {k: [] for k in range(K)}
    tau[0].append(0)

    scen_all = xi_new.reshape([1, -1])
    return tau, scen_all
