from ProblemFunctions.att_functions import *

from joblib import Parallel, delayed
from datetime import datetime
import multiprocessing
import pandas as pd
import numpy as np
import itertools
import pickle
import copy
import time


def data_gen_fun_max(K, env, att_series=None, problem_type="test", time_limit=5*60, perc_label=0.05, normalized=False):
    if att_series is None:
        att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
                      "const_to_const_dist"]
    # Initialize
    start_time = time.time()

    # FILL ALL SUBSETS, store attribute information here!
    tau, tau_att, new_scen, theta_start, zeta_init = fill_subsets(K, env, att_series)
    if zeta_init < 1e-04:
        # solution already robust. Can't be better with this approach.
        return None

    # initialize on all cores, store init theta and tot scens
    theta_init, tot_scens_init, rt_mean, max_depth = init_pass(K, env, tau)

    att_index = att_index_maker(env, att_series)
    # CREATE SUBTREE
    input_data_dict, success_data_dict, tau_dict, _, covered_nodes, level = make_upper_tree(K, env, att_series,
                                                                                            tau, tau_att,
                                                                                            theta_init,
                                                                                            theta_start, zeta_init,
                                                                                            tot_scens_init, att_index,
                                                                                            max_depth=max_depth,
                                                                                            time_limit=time_limit,
                                                                                            rt_mean=rt_mean)

    # RUN RANDOM RUNS FOR SUCCESS DATA
    success_data_dict, num_start_nodes, time_per_node, num_runs_info = random_runs_from_nodes(K, env, tau_dict,
                                                                                              success_data_dict,
                                                                                              covered_nodes=covered_nodes,
                                                                                              level=level,
                                                                                              time_limit=time_limit,
                                                                                              perc_label=perc_label,
                                                                                              normalized=normalized)
    # make input and success data final
    input_data = []
    success_data = []
    for node in input_data_dict.keys():
        try:
            input_data = np.vstack([input_data, input_data_dict[node]])
        except:
            input_data = input_data_dict[node].reshape([1, -1])
        success_data.append(success_data_dict[node])
    success_data = np.array(success_data)

    # termination results
    runtime = time.time() - start_time
    now = datetime.now().time()
    now_nice = f"{now.hour}:{now.minute}:{now.second}"
    Y_10 = np.round(np.quantile(success_data, 0.1), 2)
    Y_mean = np.round(np.mean(success_data), 2)
    Y_90 = np.round(np.quantile(success_data, 0.9), 2)
    print(f"Instance SPD {env.inst_num}, completed at {now_nice}, solved in {runtime}s. "
          f"Data points = {len(success_data)}, Y: [{Y_10},{Y_mean},{Y_90}], R: {num_runs_info}")
    results = {"X": input_data, "Y": success_data}

    with open(f"ShortestPathCluster/Data/Results/TrainData/inst_results/data_results_{problem_type}_"
              f"{env.inst_num}.pickle", "wb") as handle:
        pickle.dump(results, handle)

    # save information
    pd.Series([level, rt_mean, num_start_nodes, tot_scens_init, time_per_node, *num_runs_info, runtime],
              index=["level", "rt_mean", "num_start_nodes", "tot_scens_init", "time_per_node",
                     "num_runs_min", "num_runs_mean", "num_runs_max", "runtime"],
              dtype=float).to_pickle(f"ShortestPathCluster/Data/RunInfo/"
                                     f"run_info_{problem_type}_inst{env.inst_num}.pickle")


def fill_subsets(K, env, att_series):
    # Initialize
    start_time = time.time()

    # K-branch and bound algorithm
    new_model = True

    det_model = scenario_fun_deterministic_build(env)

    tau, tau_att = init_scen(K, env, att_series, det_model)

    while True:
        # MASTER PROBLEM
        if new_model:
            # master problem
            theta, x, y, model = scenario_fun_build(K, tau, env)
        else:
            # NEW NODE from k_new
            tau = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau[k_new])
            try:
                adj_tau_k = np.vstack([adj_tau_k, xi])
            except:
                adj_tau_k = xi.reshape([1, -1])
            tau[k_new] = adj_tau_k

            # NEW ATTRIBUTE NODE
            tau_att = copy.deepcopy(tau_att)
            try:
                tau_att[k_new] = np.vstack([tau_att[k_new], scen_att + scen_att_k[k_new]])
            except:
                tau_att[k_new] = np.array(scen_att + scen_att_k[k_new]).reshape([1, -1])

            # master problem
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)

        # SUBPROBLEM
        zeta, xi = separation_fun(K, x, y, theta, env, tau)

        # check if robust
        if zeta < 1e-04:
            now = datetime.now().time()
            print(
                "Instance SPD {}: ROBUST IN FILLING SUBSETS RUN ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
                    env.inst_num, np.round(time.time() - start_time, 3), now,
                    np.round(theta, 4), np.round(zeta, 4), [len(t) for t in tau.values()]))
            break
        else:
            new_model = False
            scen_att, scen_att_k = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, det_model=det_model)

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
        if len(full_list) == K:
            break
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
            k_new = K_set[-1]

    return tau, tau_att, xi, theta, zeta


def init_scen(K, env, att_series, det_model,):
    tau = {k: [] for k in np.arange(K)}
    tau[0].append(env.init_uncertainty)

    # run master problem
    theta, x, y, _ = scenario_fun_build(K, tau, env)

    # run sub problem
    zeta, xi_new = separation_fun(K, x, y, theta, env, tau)

    # new tau to be saved in N_set
    tau = {k: [] for k in np.arange(K)}
    tau[0] = xi_new.reshape([1, -1])
    tau_att = {k: [] for k in np.arange(K)}

    first_att_part, k_att_part = attribute_per_scen(K, xi_new, env, att_series, tau, theta, x, y, det_model=det_model)
    tau_att[0] = np.array(first_att_part + k_att_part[0]).reshape([1, -1])
    return tau, tau_att


def init_pass(K, env, tau):
    thread_count = multiprocessing.cpu_count()
    init_results = Parallel(n_jobs=-1)(delayed(random_pass)(K, env, tau, progress=False) for i in np.arange(thread_count))

    theta_init = np.mean([res[0] for res in init_results])
    tot_scens_init = np.mean([res[1] for res in init_results])
    rt_mean = np.mean([res[2] for res in init_results])
    max_depth = np.quantile([res[1] for res in init_results], 0.9)

    return theta_init, tot_scens_init, rt_mean, max_depth


def random_pass(K, env, tau, progress=False, time_per_node=None):
    # Initialize
    start_time = time.time()
    new_model = True

    while True:
        # MASTER PROBLEM
        if new_model:
            # master problem
            theta, x, y, model = scenario_fun_build(K, tau, env)
        else:
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

        if time_per_node is not None and time.time() - start_time > time_per_node/5:
            return None
        # choose new k randomly
        k_new = np.random.randint(K)

    runtime = time.time() - start_time
    if time_per_node is None:
        tot_scens = np.sum([len(t) for t in tau.values()])
        return theta, tot_scens, runtime
    else:
        return theta


def make_upper_tree(K, env, att_series, tau, tau_att, theta_init, theta_start, zeta_init,
                   tot_scens_init, att_index, max_depth=5, time_limit=5*60, rt_mean=10):
    # Initialize
    # K-branch and bound algorithm
    start_time = time.time()

    # decide on maximum number of start nodes
    thread_count = multiprocessing.cpu_count()
    start_nodes_max = (time_limit*thread_count)/(5*rt_mean)
    # node information
    node_index = (0,)

    input_data_dict = {}
    success_data_dict = {}
    level_next = {(): (0,), node_index: tuple(np.arange(K))}
    covered_nodes = {1: [(0, )]}
    tau_dict = {node_index: tau}
    tau_att_dict = {node_index: tau_att}
    xi_dict = {(): []}
    theta_dict = {(): theta_start}
    zeta_dict = {(): zeta_init}

    # INITIAL MODEL
    det_model = scenario_fun_deterministic_build(env)
    model = None
    # CREATE UPPER TREE
    while len(level_next):
        node_index = list(level_next.keys())[0]
        if len(node_index) + 2 not in covered_nodes:
            covered_nodes[len(node_index) + 2] = []
            success_data_dict[len(node_index) + 2] = {}
        while len(level_next[node_index]):
            k_new = level_next[node_index][0]
            level_next[node_index] = level_next[node_index][1:]
            new_node_index = node_index + (k_new,)

            # ALGORITHM
            tau = tau_dict[new_node_index]
            tau_att = tau_att_dict[new_node_index]
            # MASTER PROBLEM
            if model is None:
                theta, x, y, model = scenario_fun_build(K, tau, env)
            else:
                theta, x, y = scenario_fun_update_sub_tree(K, new_node_index, xi_dict, env, model)

            # SUBPROBLEM
            zeta, xi = separation_fun(K, x, y, theta, env, tau)
            scen_att, scen_att_k = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, det_model=det_model)

            # check if robust
            if zeta < 1e-04:
                success_data_dict[len(new_node_index)][new_node_index] = theta
                continue

            # STATE DATA
            K_set = np.arange(K)
            tot_scens = np.sum([len(t) for t in tau.values()])
            tau_s = state_features(theta, zeta, tot_scens, tot_scens_init, theta_init, zeta_init,
                                   theta_dict[node_index], zeta_dict[node_index])

            # INPUT DATA
            new_input_data = input_fun(K, tau_s, tau_att, scen_att, scen_att_k, att_index)
            for k in np.arange(K):
                input_data_dict[new_node_index + (k,)] = new_input_data[k]
                covered_nodes[len(new_node_index) + 1].append(new_node_index + (k,))

            level_next[new_node_index] = tuple(K_set)
            xi_dict[new_node_index] = xi
            theta_dict[new_node_index] = theta
            zeta_dict[new_node_index] = zeta

            # NEW ATTRIBUTE NODE
            for k in K_set:
                # NEW NODE
                tau_tmp = copy.deepcopy(tau)
                adj_tau_k = copy.deepcopy(tau_tmp[k])
                try:
                    adj_tau_k = np.vstack([adj_tau_k, xi])
                except:
                    adj_tau_k = xi.reshape([1, -1])
                tau_tmp[k] = adj_tau_k
                # add to node set
                tau_dict[new_node_index + (k,)] = tau_tmp

                # NEW ATTRIBUTE NODE
                tau_att_tmp = copy.deepcopy(tau_att)
                try:
                    tau_att_tmp[k] = np.vstack([tau_att_tmp[k], scen_att + scen_att_k[k]])
                except:
                    tau_att_tmp[k] = np.array(scen_att + scen_att_k[k]).reshape([1, -1])
                tau_att_dict[new_node_index + (k,)] = tau_att_tmp
        # after each set of children
        del level_next[node_index]

        # TERMINATION CONSTRAINTS
        if len(covered_nodes[len(new_node_index) + 1]) > start_nodes_max:
            # delete all data and covered data of last level
            last_level = len(new_node_index) + 1
            nodes_to_del = []
            for node in input_data_dict.keys():
                if len(node) == last_level:
                    nodes_to_del.append(node)
            for node in nodes_to_del:
                del input_data_dict[node]
            del covered_nodes[last_level]
            del success_data_dict[last_level]
            break
        elif len(list(level_next)[0]) > max_depth:
            break
    level = list(covered_nodes.keys())[-1]
    runtime = time.time() - start_time

    return input_data_dict, success_data_dict, tau_dict, runtime, covered_nodes, level


def random_runs_from_nodes(K, env, tau_dict, success_data_dict, covered_nodes, level, time_limit=5*60,
                           perc_label=0.05, normalized=False):
    # find starting nodes
    starting_nodes = []
    for node in covered_nodes[level]:
        if node in success_data_dict[level]:
            continue
        starting_nodes.append(node)

    # actual time per node
    thread_count = multiprocessing.cpu_count()
    time_per_node = (time_limit*thread_count)/len(starting_nodes)

    print(f"Instance SPD {env.inst_num}: K = {K}, N = {env.N}, level = {level}, "
          f"{len(starting_nodes)} starting nodes, each {np.round(time_per_node, 3)}s")
    thetas = Parallel(n_jobs=-1)(delayed(starting_nodes_pass)(K, env, tau_dict[n], time_per_node)
                                     for n in starting_nodes)

    # find best perc_label percent.
    all_thetas = np.array([t for t_list in thetas for t in t_list])
    theta_compete = np.quantile(all_thetas, perc_label)
    # for all robust nodes, check if theta < theta_perc. if so, it's 1
    for level, level_nodes in success_data_dict.items():
        for node, theta in level_nodes.items():
            if theta < theta_compete:
                success_data_dict[level][node] = 1
            else:
                success_data_dict[level][node] = 0

    num_runs = []
    for i, node in enumerate(starting_nodes):
        num_runs.append(len(thetas[i]))
        if len(thetas[i]):
            # success predictions of starting nodes
            success_data_dict[level][node] = np.sum(np.array(thetas[i]) < theta_compete)/len(thetas[i])
        else:
            success_data_dict[level][node] = 0

    # get success predictions of other nodes in upper tree
    success_data_dict = finish_success(K, success_data_dict, covered_nodes, level, normalized=normalized)

    num_runs_info = [np.min(num_runs), np.mean(num_runs), np.max(num_runs)]
    return success_data_dict, len(starting_nodes), time_per_node, num_runs_info


def starting_nodes_pass(K, env, tau, time_per_node):
    start_time = time.time()
    results = []
    while len(results) < 50 and time.time() - start_time < time_per_node:
        new_res = random_pass(K, env, tau, time_per_node=time_per_node)
        if new_res is None:
            continue
        results.append(new_res)
    return results


def finish_success(K, success_data_dict, covered_nodes, max_depth, normalized=False):
    all_levels = np.arange(2, max_depth)[::-1]
    for level in all_levels:
        for parent in covered_nodes[level]:
            if parent in success_data_dict[level]:
                continue
            prob_matrix = np.array([[1 - success_data_dict[level+1][parent + (k,)], success_data_dict[level+1][parent + (k,)]] for k in np.arange(K)])
            choices = itertools.product(*[np.arange(2)] * K)
            suc_prob = 0
            for c in choices:
                if c == (0, 0):
                    continue
                suc_prob += np.prod([prob_matrix[k, c[k]] for k in np.arange(K)])
            success_data_dict[level][parent] = suc_prob

    success_data_dict_final = {}

    if normalized:
        # normalize per parent-child subtree
        all_levels = np.arange(1, max_depth)
        for level in all_levels:
            for parent in covered_nodes[level]:
                child_values = np.zeros(K)
                if parent + (0,) not in success_data_dict[level+1]:
                    continue
                for k in np.arange(K):
                    child_values[k] = success_data_dict[level+1][parent + (k,)]

                if np.sum(child_values) == 0:
                    for k in np.arange(K):
                        success_data_dict_final[parent + (k,)] = 0
                else:
                    for k in np.arange(K):
                        success_data_dict_final[parent + (k,)] = child_values[k]/np.sum(child_values)
    else:
        for level, nodes in success_data_dict.items():
            for node, label in nodes.items():
                success_data_dict_final[node] = label

    return success_data_dict_final
