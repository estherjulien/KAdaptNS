from src.sp.problem_functions.att_functions import *
from src.sp.problem_functions.functions_milp import *

from datetime import datetime
import numpy as np
import joblib
import pickle
import copy
import time
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
Code for running K-B&B-NodeSelection for solving the shortest path problem
(ML guided node selection for K-adaptability branch and bound)

INPUT:  K = number of second-stage decisions (or subsets)
        env = instance of the capital budgeting problem
        att_series = names of attributes used for this problem
        max_level = level in the tree up to where ML node selection is done
        time_limit = seconds spend per instance - if limit is reached, 
                     the incumbent solution will be used as final solution
        thresh = threshold that is applied to ML predictions
        num_runs = number of initial runs done
        sc_pre = scaling method of training data
OUTPUT: solution to shortest path problem
        saved in src/sp/data/results/ml/
"""


def algorithm(K, env, att_series=None, max_level=None, success_model_name=None, time_limit=30 * 60, print_info=True,
              problem_type="test", thresh=None, num_runs=2, sc_pre="dive", sc_min_max=0):
    rng = np.random.RandomState(env.inst_num)
    gp_env = gp.Env()
    gp_env.setParam("OutputFlag", 0)
    gp_env.setParam("Threads", 1)
    if att_series is None:
        att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist"]
    # Initialize
    start_time = time.time()
    # initialization for saving stuff
    inc_thetas_t = dict()
    inc_thetas_n = dict()
    prune_count = 0
    mp_time = 0
    sp_time = 0
    new_model = True

    det_model = scenario_fun_deterministic_build(env, gp_env)

    # initialize N_set
    N_set, tau, scen_all, att_all, att_all_k, zeta_init, theta_start = fill_subsets(K, env, gp_env, att_series)

    N_set_trash = []
    # run a few times to get theta_init, tot_scens_init
    # init random passes
    theta_i = np.inf
    tot_nodes = 0

    if sc_pre == "dive":
        init_theta_list = []
        tot_scens_init_list = []
        for i in np.arange(num_runs):
            theta, tot_scens, new_nodes = random_pass(K, env, tau, gp_env)
            init_theta_list.append(theta)
            tot_scens_init_list.append(tot_scens)
            tot_nodes += new_nodes
            if theta - theta_i > -1e-8:
                continue
            else:   # store incumbent theta
                theta_i = copy.deepcopy(theta)
                inc_thetas_t[time.time() - start_time] = theta_i
                inc_thetas_n[tot_nodes] = theta_i

        theta_init = np.mean(init_theta_list)
        tot_scens_init = np.mean(init_theta_list)
    elif sc_pre == "alt":
        theta_init = theta_start
        tot_scens_init = K ** np.log(env.xi_dim)
    else:
        theta_init = None
        tot_scens_init = None

    # save stuff
    runtime = time.time() - start_time
    inc_thetas_t[runtime] = theta_i
    inc_thetas_n[tot_nodes] = theta_i

    # load ML model
    success_model = joblib.load(success_model_name)
    scales = success_model._scales

    att_index = att_index_maker(env, att_series)

    new_xi_num = len(scen_all) - 1
    from_trash = False
    # K-branch and bound algorithm
    k_new = None
    now = datetime.now().time()
    print("Instance S {}: started at {}".format(env.inst_num, now))
    while (N_set or N_set_trash or k_new is not None) and time.time() - start_time < time_limit:
        # MASTER PROBLEM
        if new_model:
            tot_nodes += 1
            # take new node
            if N_set:
                new_pass = rng.randint(len(N_set))
                placement = N_set.pop(new_pass)
                from_trash = False
            else:
                new_pass = rng.randint(len(N_set_trash))
                placement = N_set_trash.pop(new_pass)
                from_trash = True

            tau = {k: scen_all[placement[k]] for k in np.arange(K)}

            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_build(K, tau, env, gp_env)
            mp_time += time.time() - start_mp
            theta_pre, zeta_pre = [1, 1]
        else:
            theta_pre = copy.copy(theta)
            zeta_pre = copy.copy(zeta)
            # new node from k_new
            tot_nodes += 1
            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)
            mp_time += time.time() - start_mp

            placement[k_new].append(new_xi_num)
            tau = {k: scen_all[placement[k]] for k in np.arange(K)}

        k_new = None
        # prune if theta higher than current robust theta
        if theta - theta_i > -1e-8:
            prune_count += 1
            new_model = True
            continue

        # SUBPROBLEM
        start_sp = time.time()
        zeta, xi = separation_fun(K, x, y, theta, env, placement, gp_env)
        sp_time += time.time() - start_sp

        # check if robust
        if zeta <= 1e-04:
            if print_info:
                now = datetime.now().time()
                print("Instance S {}: ROBUST at iteration {} ({}) (time {})   :obj = {},    violation = {}".format(
                    env.inst_num, tot_nodes, np.round(time.time() - start_time, 3), now, np.round(theta, 4),
                    np.round(zeta, 4)))

            theta_i = copy.deepcopy(theta)
            inc_thetas_t[time.time() - start_time] = theta_i
            inc_thetas_n[tot_nodes] = theta_i
            prune_count += 1
            new_model = True
            continue
        else:
            new_model = False
            new_xi_num += 1
            scen_all = np.vstack([scen_all, xi])

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
        if len(full_list) == 0:
            K_set = [0]
            k_new = 0
        elif len(full_list) == K:
            # predict subset
            if max_level is not None and sum([len(t) for t in tau.values()]) - K >= max_level:
                K_set = np.arange(K)
                k_new = rng.randint(K)
                # add an empty array, for this scenario we don't want to construct attributes
                empty_array = np.empty((1, len(att_all[0])))
                empty_array[:] = np.nan
                att_all = np.vstack([att_all, empty_array])
                att_all_k.append([])
            else:
                # find attribute of new scenario
                scen_att, scen_att_k = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, det_model=det_model)
                att_all = np.vstack([att_all, scen_att])
                att_all_k.append(scen_att_k)

                # STATE FEATURES (based on master and sub problem)
                tot_scens = np.sum([len(t) for t in tau.values()])
                k_att_sel = {k: np.array([att_all_k[p][k] for p in placement[k]]) for k in np.arange(K)}
                tau_att = {k: np.hstack([att_all[placement[k]], k_att_sel[k]]) for k in np.arange(K)}
                tau_s = state_features(theta, zeta, tot_scens, zeta_init, theta_pre, zeta_pre,
                                       depth_i=tot_scens_init, theta_i=theta_init)
                K_set, predictions = predict_subset(K, tau_att, scen_att, scen_att_k, success_model, att_index, tau_s, scales=scales)
                # if prediction is lower than a threshold, we choose another starting point from N_set
                # The node is saved in "thrash"
                if thresh is not None and not from_trash and np.all(predictions < thresh):
                    N_set_trash.append(placement)
                    new_model = True
                    continue
                k_new = K_set[0]
        else:
            # find attributes
            scen_att, scen_att_k = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, det_model=det_model)
            att_all = np.vstack([att_all, scen_att])
            att_all_k.append(scen_att_k)

            # select empty subset
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

    # termination results
    runtime = time.time() - start_time
    inc_thetas_t[time.time() - start_time] = theta_i
    inc_thetas_n[tot_nodes] = theta_i

    now = datetime.now().time()
    now_nice = f"{now.hour}:{now.minute}:{now.second}"
    print(f"Instance SP {env.inst_num}, completed at {now_nice}, solved in {np.round(runtime/60, 3)} minutes")

    results = {"obj": theta_i, "inc_obj_time": inc_thetas_t,
               "inc_obj_nodes": inc_thetas_n, "runtime": time.time() - start_time,
               "tot_nodes": tot_nodes, "mp_time": mp_time, "sp_time": sp_time}

    os.makedirs("src/sp/data/results/ml", exist_ok=True)
    with open(f"src/sp/data/results/ml/final_results_{problem_type}_s{env.inst_num}.pickle", "wb") as handle:
        pickle.dump(results, handle)

    return results


# code for finding one scenario per subset. From this point on, ML is used.
def fill_subsets(K, env, att_series, gp_env, progress=False):
    # Initialize
    start_time = time.time()

    # K-branch and bound algorithm
    new_model = True

    det_model = scenario_fun_deterministic_build(env, gp_env)

    xi_init, att_init, att_init_k = init_scen(K, env, att_series, det_model)

    N_set = [{k: [] for k in np.arange(K)}]
    N_set[0][0].append(0)
    scen_all = xi_init.reshape([1, -1])
    att_all = np.array(att_init).reshape([1, -1])
    att_all_k = [att_init_k]
    new_xi_num = 0
    while True:
        # MASTER PROBLEM
        if new_model:
            # master problem
            placement = N_set.pop(0)
            tau = {k: scen_all[placement[k]] for k in np.arange(K)}
            theta, x, y, model = scenario_fun_build(K, tau, env, gp_env)
        else:
            # NEW NODE from k_new
            # master problem
            placement[k_new].append(new_xi_num)
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)

            tau = {k: scen_all[placement[k]] for k in np.arange(K)}

        # SUBPROBLEM
        zeta, xi = separation_fun(K, x, y, theta, env, tau, gp_env)

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
        if len(full_list) == K:
            N_set.append(placement)
            break

        # check if robust
        if zeta < 1e-04:
            now = datetime.now().time()
            if progress:
                print(
                    "Instance SP {}: ROBUST IN FILLING SUBSETS RUN ({}) (time {})   :obj = {},    violation = {}".format(
                        env.inst_num, np.round(time.time() - start_time, 3), now,
                        np.round(theta, 4), np.round(zeta, 4)))
            break
        else:
            new_model = False
            new_xi_num += 1
            scen_all = np.vstack([scen_all, xi])
            scen_att, scen_att_k = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, det_model=det_model)
            att_all = np.vstack([att_all, scen_att])
            att_all_k.append(scen_att_k)

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
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

    return N_set[::-1], tau, scen_all, att_all, att_all_k, zeta, theta


# Instead of using the scenario of all zeros (if this is in the uncertainty set), another initial scenario is sought.
def init_scen(K, env, att_series, det_model, gp_env):
    tau = {k: [] for k in np.arange(K)}
    tau[0].append(env.init_uncertainty)

    # run master problem
    theta, x, y, _ = scenario_fun_build(K, tau, env, gp_env)

    # run sub problem
    zeta, xi_new = separation_fun(K, x, y, theta, env, tau, gp_env)

    # attributes
    first_att_part, k_att_part = attribute_per_scen(K, xi_new, env, att_series, tau, theta, x, y, det_model=det_model)
    return xi_new, first_att_part, k_att_part


# initial random passes to get information for scaling the features
def random_pass(K, env, tau, gp_env, progress=False):
    rng = np.random.RandomState(env.inst_num)
    # Initialize
    start_time = time.time()
    new_model = True
    tot_nodes = 1
    while True:
        # MASTER PROBLEM
        if new_model:
            # master problem
            theta, x, y, model = scenario_fun_build(K, tau, env, gp_env)
        else:
            tot_nodes += 1
            # NEW NODE from k_new
            tau = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau[k_new])
            try:
                adj_tau_k = np.vstack([adj_tau_k, xi])
            except:
                adj_tau_k = xi.reshape([1, -1])
            tau[k_new] = adj_tau_k

            # master problem
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)

        # SUBPROBLEM
        zeta, xi = separation_fun(K, x, y, theta, env, tau, gp_env)

        # check if robust
        if zeta < 1e-04:
            if progress:
                now = datetime.now().time()
                print(
                    "Instance SPD {}: INIT PASS ROBUST ({}) (time {})   :obj = {},    violation = {}".format(
                        env.inst_num, np.round(time.time() - start_time, 3), now,
                        np.round(theta, 4), np.round(zeta, 4)))
            break
        else:
            new_model = False

        # choose new k randomly
        k_new = rng.randint(K)

    tot_scens = np.sum([len(t) for t in tau.values()])
    return theta, tot_scens, tot_nodes
