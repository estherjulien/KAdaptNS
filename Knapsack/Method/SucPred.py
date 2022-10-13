from ProblemFunctions.att_functions import *
from ProblemFunctions.functions_milp import *

from datetime import datetime
import numpy as np
import joblib
import pickle
import copy
import time


def algorithm(K, env, att_series=None, max_level=5, success_model_name=None, time_limit=30 * 60, print_info=True,
              problem_type="test", thresh=None, num_runs=5):
    if att_series is None:
        att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
                  "const_to_const_dist"]
    # Initialize
    start_time = time.time()
    # initialization for saving stuff
    inc_thetas_t = dict()
    inc_thetas_n = dict()
    prune_count = 0
    mp_time = 0
    sp_time = 0
    new_model = True
    # FOR STATIC ATTRIBUTE
    x_static = None

    det_model = scenario_fun_deterministic_build(env)

    # initialize N_set
    N_set, tau, scen_all, att_all, att_all_k, zeta_init = fill_subsets(K, env, att_series, x_static)

    N_set_trash = []
    # run a few times to get theta_init, tot_scens_init
    # init random passes
    theta_i = np.inf
    init_theta_list = []
    tot_scens_init_list = []
    tot_nodes = 0
    for i in np.arange(num_runs):
        theta, tot_scens, new_nodes = random_pass(K, env, tau)
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

    # save stuff
    runtime = time.time() - start_time
    inc_thetas_t[runtime] = theta_i
    inc_thetas_n[tot_nodes] = theta_i

    if success_model_name is None:
        success_model_name = f"./Data/Models/nn_regr_cb_N10_K2_test_all_norm_D5_W100.h5"
    success_model = joblib.load(success_model_name)

    att_index = att_index_maker(env, att_series)

    new_xi_num = len(scen_all) - 1
    from_trash = False
    # K-branch and bound algorithm
    now = datetime.now().time()
    print("Instance S {}: started at {}".format(env.inst_num, now))
    while (N_set or N_set_trash) and time.time() - start_time < time_limit:
        # MASTER PROBLEM
        if new_model:
            tot_nodes += 1
            # take new node
            if N_set:
                new_pass = np.random.randint(len(N_set))
                placement = N_set.pop(new_pass)
                from_trash = False
            else:
                new_pass = np.random.randint(len(N_set_trash))
                placement = N_set_trash.pop(new_pass)
                from_trash = True

            tau = {k: scen_all[placement[k]] for k in np.arange(K)}

            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_build(K, tau, env)
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

        # prune if theta higher than current robust theta
        if theta - theta_i < 1e-8:
            prune_count += 1
            new_model = True
            continue

        # SUBPROBLEM
        start_sp = time.time()
        zeta, xi = separation_fun(K, x, y, theta, env, placement)
        sp_time += time.time() - start_sp

        # check if robust
        if zeta <= 1e-04:
            if print_info:
                now = datetime.now().time()
                print("Instance S {}: ROBUST at iteration {} ({}) (time {})   :theta = {},    zeta = {}   Xi{},   "
                      "prune count = {}".format(
                    env.inst_num, tot_nodes, np.round(time.time() - start_time, 3), now, np.round(theta, 4),
                    np.round(zeta, 4), [len(t) for t in placement.values()], prune_count))

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
            if sum([len(t) for t in tau.values()]) - K >= max_level:
                K_set = np.arange(K)
                k_new = np.random.randint(K)
                # add nothing
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
                tau_s = state_features(theta, zeta, tot_scens, tot_scens_init, theta_init, zeta_init, theta_pre, zeta_pre)
                K_set, predictions = predict_subset(K, tau_att, scen_att, scen_att_k, success_model, att_index, tau_s)
                if thresh is not None and not from_trash and np.all(predictions < thresh):
                    N_set_trash.append(placement)
                    new_model = True
                    continue
                k_new = K_set[0]
        else:
            # find attributes here too
            scen_att, scen_att_k = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, det_model=det_model)
            att_all = np.vstack([att_all, scen_att])
            att_all_k.append(scen_att_k)

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

    results = {"theta": theta_i, "inc_thetas_t": inc_thetas_t,
               "inc_thetas_n": inc_thetas_n, "runtime": time.time() - start_time,
               "tot_nodes": tot_nodes, "mp_time": mp_time, "sp_time": sp_time}

    with open(f"ShortestPathCluster/Data/Results/Decisions/inst_results/final_results_{problem_type}_inst{env.inst_num}.pickle", "wb") as handle:
        pickle.dump(results, handle)

    return results


def fill_subsets(K, env, att_series, progress=False):
    # Initialize
    start_time = time.time()

    # K-branch and bound algorithm
    new_model = True

    det_model = scenario_fun_deterministic_build(env)

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
            theta, x, y, model = scenario_fun_build(K, tau, env)
        else:
            # NEW NODE from k_new
            # master problem
            placement[k_new].append(new_xi_num)
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)

            tau = {k: scen_all[placement[k]] for k in np.arange(K)}

        # SUBPROBLEM
        zeta, xi = separation_fun(K, x, y, theta, env, tau)

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
        if len(full_list) == K:
            N_set.append(placement)
            break

        # check if robust
        if zeta < 1e-04:
            now = datetime.now().time()
            if progress:
                print(
                    "Instance SP {}: ROBUST IN FILLING SUBSETS RUN ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
                        env.inst_num, np.round(time.time() - start_time, 3), now,
                        np.round(theta, 4), np.round(zeta, 4), [len(t) for t in tau.values()]))
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

    return N_set[::-1], tau, scen_all, att_all, att_all_k, zeta


def init_scen(K, env, att_series, det_model):
    tau = {k: [] for k in np.arange(K)}
    tau[0].append(env.init_uncertainty)

    # run master problem
    theta, x, y, _ = scenario_fun_build(K, tau, env)

    # run sub problem
    zeta, xi_new = separation_fun(K, x, y, theta, env, tau)

    # attributes
    first_att_part, k_att_part = attribute_per_scen(K, xi_new, env, att_series, tau, theta, x, y, det_model=det_model)
    return xi_new, first_att_part, k_att_part


def random_pass(K, env, tau, progress=False):
    # Initialize
    start_time = time.time()
    new_model = True
    tot_nodes = 1
    while True:
        # MASTER PROBLEM
        if new_model:
            # master problem
            theta, x, y, model = scenario_fun_build(K, tau, env)
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
        zeta, xi = separation_fun(K, x, y, theta, env, tau)

        # check if robust
        if zeta < 1e-04:
            if progress:
                now = datetime.now().time()
                print(
                    "Instance SPD {}: INIT PASS ROBUST ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
                        env.inst_num, np.round(time.time() - start_time, 3), now,
                        np.round(theta, 4), np.round(zeta, 4), [len(t) for t in tau.values()]))
            break
        else:
            new_model = False

        # choose new k randomly
        k_new = np.random.randint(K)

    tot_scens = np.sum([len(t) for t in tau.values()])
    return theta, tot_scens, tot_nodes
