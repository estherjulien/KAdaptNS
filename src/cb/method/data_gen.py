from cb.problem_functions.att_functions import *

from joblib import Parallel, delayed
from datetime import datetime
import multiprocessing
import gurobipy as gp
import pandas as pd
import numpy as np
import itertools
import pickle
import copy
import time
import os

"""
Code for running generating training data for the capital budgeting problem
(ML guided node selection for K-adaptability branch and bound)

INPUT:  K = number of second-stage decisions (or subsets)
        env = instance of the capital budgeting problem
        att_series = names of attributes used for this problem
        time_limit = seconds spent per instance
        perc_label = percentage of found objective values chosen as best (5%). The corresponding paths are marked as good paths
        num_runs = number of initial runs done
OUTPUT: training data for the capital budgeting problem
        in src/cb/data/train_data/
"""


def data_gen_fun_max(K, env, att_series, problem_type="test", time_limit=5*60, perc_label=0.05, normalized=False, dg_inst=None):
    # Initialize
    start_time = time.time()
    gp_env = gp.Env()
    # FOR STATIC ATTRIBUTE
    print(" Loading static results")
    try:
        x_static = static_solution_rc(env, gp_env)
    except:
        x_static = None

    # FILL ALL SUBSETS. from this point on, the sub tree is generated
    print(" Initializing subsets")
    tau, tau_att, new_scen, theta_start, zeta_init = fill_subsets(K, env, att_series, x_static, gp_env)
    theta_init_alt = theta_start
    if zeta_init < 1e-04:
        # solution already robust. Can't be better with this approach.
        return None

    # initialize on all cores, store init theta and tot scenarios
    print(" Initial pass")
    theta_init, tot_scens_init, rt_mean, max_depth = init_pass(K, env, tau)
    depth_init_alt = K ** np.log(env.xi_dim)
    att_index = att_index_maker(env, att_series)

    # CREATE SUBTREE
    print(" Create subtree")
    input_data_dict, success_data_dict, tau_dict, _, covered_nodes, level, starting_nodes = make_upper_tree(K, env, att_series,
                                                                                            tau, tau_att,
                                                                                            theta_init,
                                                                                            theta_start, zeta_init,
                                                                                            tot_scens_init, att_index,
                                                                                            max_depth=max_depth,
                                                                                            x_static=x_static,
                                                                                            time_limit=time_limit,
                                                                                            rt_mean=rt_mean,
                                                                                            depth_init_alt=depth_init_alt,
                                                                                            theta_init_alt=theta_init_alt)

    # RUN RANDOM RUNS FOR SUCCESS DATA
    print(" Simulation START")
    success_data_dict, num_start_nodes, time_per_node, num_runs_info = random_runs_from_nodes(K, env, tau_dict,
                                                                                              success_data_dict,
                                                                                              covered_nodes=covered_nodes,
                                                                                              level=level,
                                                                                              time_limit=time_limit,
                                                                                              perc_label=perc_label,
                                                                                              normalized=normalized,
                                                                                              starting_nodes=starting_nodes)
    print(" Simulation FINISHED")
    # make input and success data final
    input_data = {"dive": [], "alt": [], "no":[]}
    success_data = []
    for node in input_data_dict.keys():
        try:
            input_data["dive"] = np.vstack([input_data["dive"], input_data_dict[node][0]])
            input_data["alt"] = np.vstack([input_data["alt"], input_data_dict[node][1]])
            input_data["no"] = np.vstack([input_data["no"], input_data_dict[node][2]])
            # input_data = np.vstack([input_data, input_data_dict[node]])
        except:
            input_data["dive"] = input_data_dict[node][0].reshape([1, -1])
            input_data["alt"] = input_data_dict[node][1].reshape([1, -1])
            input_data["no"] = input_data_dict[node][2].reshape([1, -1])
            # input_data = input_data_dict[node].reshape([1, -1])
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

    os.makedirs("src/cb/data/train_data/inst_results", exist_ok=True)
    if dg_inst is None:
        dg_inst = env.inst_num
    with open(f"src/cb/data/train_data/inst_results/data_results_{problem_type}_"
              f"{dg_inst}.pkl", "wb") as handle:
        pickle.dump(results, handle)

    # save information
    os.makedirs("src/cb/data/train_data/inst_results_md", exist_ok=True)
    pd.Series([len(input_data["no"]), level, rt_mean, num_start_nodes, tot_scens_init, time_per_node, *num_runs_info, runtime],
              index=["datapoints", "level", "rt_mean", "num_start_nodes", "tot_scens_init", "time_per_node",
                     "num_runs_min", "num_runs_mean", "num_runs_max", "runtime"],
              dtype=float).to_pickle(f"src/cb/data/train_data/inst_results_md/"
                                     f"run_info_{problem_type}_inst{dg_inst}.pkl")


# code for finding one scenario per subset. From this point on, the sub tree is generated
def fill_subsets(K, env, att_series, x_static, gp_env):
    # Initialize
    start_time = time.time()

    # K-branch and bound algorithm
    new_model = True

    # initialize N_set with actual scenario
    try:
        stat_model = scenario_fun_static_build(env, x_static, gp_env)
    except:
        stat_model = None

    det_model = scenario_fun_deterministic_build(env, gp_env)

    tau, tau_att = init_scen(K, env, att_series, stat_model, det_model, x_static, gp_env)

    while True:
        # MASTER PROBLEM
        if new_model:
            # master problem
            theta, x, y, model = scenario_fun_build(K, tau, env, gp_env)
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
        zeta, xi = separation_fun(K, x, y, theta, env, tau, gp_env)

        # check if robust
        if zeta < 1e-04:
            now = datetime.now().time()
            print(
                "Instance SPD {}: ROBUST IN FILLING SUBSETS RUN ({}) (time {})   :obj = {},    violation = {}".format(
                    env.inst_num, np.round(time.time() - start_time, 3), now,
                    np.round(theta, 4), np.round(zeta, 4)))
            break
        else:
            new_model = False
            scen_att, scen_att_k = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, x_static=x_static,
                                                      stat_model=stat_model, det_model=det_model)

        full_list = [k for k in range(K) if len(tau[k]) > 0]
        if len(full_list) == K:
            break
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
            k_new = K_set[-1]

    return tau, tau_att, xi, theta, zeta


# Instead of using the scenario of all zeros (if this is in the uncertainty set), another initial scenario is sought.
def init_scen(K, env, att_series, stat_model, det_model, x_static, gp_env):
    tau = {k: [] for k in range(K)}
    tau[0].append(env.init_uncertainty)

    # run master problem
    theta, x, y, _ = scenario_fun_build(K, tau, env, gp_env)

    # run sub problem
    zeta, xi_new = separation_fun(K, x, y, theta, env, tau, gp_env)

    # new tau to be saved in N_set
    tau = {k: [] for k in range(K)}
    tau[0] = xi_new.reshape([1, -1])
    tau_att = {k: [] for k in range(K)}

    first_att_part, k_att_part = attribute_per_scen(K, xi_new, env, att_series, tau, theta, x, y, x_static=x_static,
                                                    stat_model=stat_model, det_model=det_model)
    tau_att[0] = np.array(first_att_part + k_att_part[0]).reshape([1, -1])
    return tau, tau_att


# function for running multiple random passes in parallel, one random pass per thread
def init_pass(K, env, tau):
    thread_count = multiprocessing.cpu_count()
    init_results = Parallel(n_jobs=-1)(delayed(random_pass)(K, env, tau, progress=False) for i in range(thread_count))

    theta_init = np.mean([res[0] for res in init_results])
    tot_scens_init = np.mean([res[1] for res in init_results])
    rt_mean = np.mean([res[2] for res in init_results])
    max_depth = np.quantile([res[1] for res in init_results], 0.9)

    return theta_init, tot_scens_init, rt_mean, max_depth


# initial random passes to get information for scaling the features
def random_pass(K, env, tau, gp_env=None, progress=False, time_per_node=None):
    rng = np.random.RandomState(env.inst_num)
    if gp_env is None:
        gp_env = gp.Env()
    # Initialize
    start_time = time.time()
    new_model = True

    while True:
        # MASTER PROBLEM
        if new_model:
            # master problem
            theta, x, y, model = scenario_fun_build(K, tau, env, gp_env)
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

        if time_per_node is not None and time.time() - start_time > time_per_node/5:
            return None
        # choose new k randomly
        k_new = rng.randint(K)

    runtime = time.time() - start_time
    if time_per_node is None:
        tot_scens = np.sum([len(t) for t in tau.values()])
        return theta, tot_scens, runtime
    else:
        return theta


# function for making the sub tree
def make_upper_tree(K, env, att_series, tau, tau_att, theta_init, theta_start, zeta_init,
                    tot_scens_init, att_index, depth_init_alt, theta_init_alt,
                    max_depth=5, x_static=None, time_limit=5*60, rt_mean=10):
    rng = np.random.RandomState(env.inst_num)
    gp_env = gp.Env()
    gp_env.setParam("OutputFlag", 0)
    # Initialize
    start_time = time.time()

    # decide on maximum number of start nodes
    thread_count = multiprocessing.cpu_count()
    start_nodes_max = int(np.floor((time_limit*thread_count)/(15*rt_mean)))

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
    model = None
    try:
        stat_model = scenario_fun_static_build(env, x_static, gp_env)
    except:
        stat_model = None

    det_model = scenario_fun_deterministic_build(env, gp_env)

    # CREATE UPPER TREE BY BREADTH FIRST EXPLORATION
    last_level_done = False
    prev_level_nodes = None
    while len(level_next):
        node_index = list(level_next)[0]
        next_level_happened = False
        if len(node_index) + 2 not in covered_nodes:
            if last_level_done:
                if prev_level_nodes is None:
                    # DETERMINE STARTING NODES
                    last_level = list(covered_nodes)[-1]
                    starting_nodes = []
                    for node in covered_nodes[last_level]:
                        if node in success_data_dict[last_level]:
                            continue
                        starting_nodes.append(node)
                break
            covered_nodes[len(node_index) + 2] = []
            success_data_dict[len(node_index) + 2] = {}
            next_level_happened = True

        # TERMINATION CONSTRAINTS
        if next_level_happened and len(list(level_next)[0]) + 2 >= np.ceil(max_depth):
            last_level_done = True
            print(f"max depth reached = {np.ceil(max_depth)}")

        if next_level_happened and len(node_index) > 0 and (len(covered_nodes[len(node_index) + 1]) -
                                                            len(success_data_dict[len(node_index) + 1]))*K > start_nodes_max:     # and len(covered_nodes[len(new_node_index)]) * K > start_nodes_max:
            last_level_done = True
            last_level = len(node_index) + 2
            prev_level_nodes = covered_nodes[last_level - 1]
            num_from_prev_level = int(np.ceil((len(prev_level_nodes)*K - start_nodes_max)/(1-1/K)/K))
            random_ind = rng.choice(len(prev_level_nodes), int(num_from_prev_level), replace=False)
            prev_level_nodes = {tuple(node) for node in np.array(prev_level_nodes)[random_ind]}
            starting_nodes = []

        while len(level_next[node_index]):
            k_new = level_next[node_index][0]
            level_next[node_index] = level_next[node_index][1:]
            new_node_index = node_index + (k_new,)

            # ALGORITHM
            tau = tau_dict[new_node_index]
            tau_att = tau_att_dict[new_node_index]
            # MASTER PROBLEM
            if model is None:
                theta, x, y, model = scenario_fun_build(K, tau, env, gp_env)
            else:
                theta, x, y = scenario_fun_update_sub_tree(K, new_node_index, xi_dict, env, model)

            # SUBPROBLEM
            zeta, xi = separation_fun(K, x, y, theta, env, tau, gp_env)

            # check if robust
            if zeta < 1e-04:
                success_data_dict[len(new_node_index)][new_node_index] = theta
                if prev_level_nodes is not None and new_node_index not in prev_level_nodes and len(prev_level_nodes):
                    random_choice = rng.choice(len(prev_level_nodes))
                    random_prev_node = tuple(np.array(list(prev_level_nodes))[random_choice])
                    prev_level_nodes.remove(random_prev_node)
                continue

            # CHECK IF WE NEED TO GET THE NEXT NODE (FOR LAST LEVEL)
            if prev_level_nodes is not None:
                if new_node_index not in prev_level_nodes:
                    starting_nodes += [new_node_index + (k,) for k in range(K)]
                else:
                    starting_nodes.append(new_node_index)
                    prev_level_nodes.remove(new_node_index)
                    continue

            # STATE DATA
            K_set = np.arange(K)
            tot_scens = np.sum([len(t) for t in tau.values()])
            tau_s = state_features(theta, zeta, tot_scens, zeta_init,
                                   theta_dict[node_index], zeta_dict[node_index],
                                   depth_i=tot_scens_init, depth_i_alt=depth_init_alt,
                                   theta_i=theta_init, theta_i_alt=theta_init_alt, return_all=True)
            scen_att, scen_att_k = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, x_static=x_static,
                                                      stat_model=stat_model, det_model=det_model)
            # INPUT DATA
            new_input_data = input_fun(K, tau_s, tau_att, scen_att, scen_att_k, att_index, return_all=True)
            for k in range(K):
                features = [data[k] for data in new_input_data]
                input_data_dict[new_node_index + (k,)] = features
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

    # find starting nodes
    level = list(covered_nodes.keys())[-1]
    runtime = time.time() - start_time

    return input_data_dict, success_data_dict, tau_dict, runtime, covered_nodes, level, starting_nodes


# function for running random dives from the last level of the sub tree
def random_runs_from_nodes(K, env, tau_dict, success_data_dict, covered_nodes, starting_nodes, level, time_limit=5*60,
                           perc_label=0.05, normalized=False):
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
            success_data_dict[len(node)][node] = np.sum(np.array(thetas[i]) < theta_compete)/len(thetas[i])
        else:
            success_data_dict[len(node)][node] = 0

    # back propagate success probabilities from last level to the root
    success_data_dict = finish_success(K, success_data_dict, covered_nodes, level, normalized=normalized)

    num_runs_info = [np.min(num_runs), np.mean(num_runs), np.max(num_runs)]
    return success_data_dict, len(starting_nodes), time_per_node, num_runs_info


# function for random dives per starting node
def starting_nodes_pass(K, env, tau, time_per_node):
    start_time = time.time()
    results = []
    gp_env = gp.Env()
    gp_env.setParam("OutputFlag", 0)
    gp_env.setParam("Threads", 1)
    while len(results) < 50 and time.time() - start_time < time_per_node:
        new_res = random_pass(K, env, tau, gp_env, time_per_node=time_per_node)
        if new_res is None:
            continue
        results.append(new_res)
    return results


# back propagate success probabilities from last level to the root
def finish_success(K, success_data_dict, covered_nodes, max_depth, normalized=False):
    all_levels = range(2, max_depth)[::-1]
    for level in all_levels:
        for parent in covered_nodes[level]:
            if parent in success_data_dict[level]:
                continue
            prob_matrix = np.array([[1 - success_data_dict[level+1][parent + (k,)], success_data_dict[level+1][parent + (k,)]] for k in range(K)])
            choices = itertools.product(*[np.arange(2)] * K)
            suc_prob = 0
            for c in choices:
                if np.sum(c) == 0:
                    continue
                suc_prob += np.prod([prob_matrix[k, c[k]] for k in range(K)])
            success_data_dict[level][parent] = suc_prob

    success_data_dict_final = {}

    if normalized:
        # normalize per parent-child subtree
        all_levels = range(1, max_depth)
        for level in all_levels:
            for parent in covered_nodes[level]:
                child_values = np.zeros(K)
                if parent + (0,) not in success_data_dict[level+1]:
                    continue
                for k in range(K):
                    child_values[k] = success_data_dict[level+1][parent + (k,)]

                if np.sum(child_values) == 0:
                    for k in range(K):
                        success_data_dict_final[parent + (k,)] = 0
                else:
                    for k in range(K):
                        success_data_dict_final[parent + (k,)] = child_values[k]/np.sum(child_values)
    else:
        for level, nodes in success_data_dict.items():
            for node, label in nodes.items():
                success_data_dict_final[node] = label

    return success_data_dict_final
