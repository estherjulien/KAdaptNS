from Knapsack.ProblemFunctions.functions_milp import *

import numpy as np


def predict_subset(K, tau_att, scen_att, scen_att_k, success_model, att_index, state_features, nn_used=False):
    X = input_fun(K, state_features, tau_att, scen_att, scen_att_k, att_index)

    if nn_used:
        success_prediction = success_model.predict(X)[:, 0]
    else:
        pred_tmp = success_model.predict_proba(X)
        success_prediction = np.array([i[1] for i in pred_tmp])
    order = np.argsort(success_prediction)
    return order[::-1], success_prediction


def state_features(theta, zeta, depth, depth_i, theta_i, zeta_i, theta_pre, zeta_pre):
    # objective, objective compared to previous, violation
    features = [theta / theta_i, theta / theta_pre, zeta / zeta_i]
    # violation compared to previous
    try:
        features.append(zeta/zeta_pre)
    except ZeroDivisionError:
        features.append(1)
    # depth
    features.append(depth/depth_i)

    return np.array(features)


def input_fun(K, state_features, tau_att, scen_att_pre, scen_att_k, att_index):
    att_num = len(att_index)
    scen_att = {k: np.array(scen_att_pre + scen_att_k[k]) for k in np.arange(K)}

    diff_info = {k: [] for k in np.arange(K)}
    for k in np.arange(K):
        for att_type in np.arange(att_num):
            diff_info[k].append(np.linalg.norm(np.mean(tau_att[k][:, att_index[att_type]], axis=0) - scen_att[k][att_index[att_type]]) / len(att_index[att_type]))

    X = np.array([np.hstack([state_features, diff_info[k]]) for k in np.arange(K)])

    return X


def attribute_per_scen(K, scen, env, att_series, tau, theta, x, y, det_model=None):
    # create list of attributes
    # subset is first value
    sr_att = []
    sr_att_k = {k: [] for k in np.arange(K)}

    if "coords" in att_series:
        for i in np.arange(env.xi_dim):
            sr_att.append(scen[i])

    # based on other problems
    # deterministic problem
    if "obj_det" in att_series or "x_det" in att_series or "y_det" in att_series:
        theta_det, y_det = scenario_fun_deterministic_update(env, scen, det_model)
    if "obj_det" in att_series:
        sr_att.append(theta_det / theta)
    if "y_det" in att_series:
        sr_att += y_det

    # static problem doesn't exist for knapsack

    # k dependent
    if "slack" in att_series:
        slack = slack_fun(K, scen, env, theta, x, y)
        for k in np.arange(K):
            sr_att_k[k] += slack[k]
    if "const_to_z_dist" in att_series:
        c_to_z = const_to_z_fun(K, scen, env, theta, x, y)
        for k in np.arange(K):
            sr_att_k[k] += c_to_z[k]
    if "const_to_const_dist" in att_series:
        c_to_c = const_to_const_fun(K, scen, env, tau)
        for k in np.arange(K):
            sr_att_k[k] += c_to_c[k]
    return sr_att, sr_att_k


def slack_fun(K, scen, env, theta, x, y):
    slack = []
    for k in np.arange(K):
        slack.append(abs(sum(env.cost[i] * (1 - 0.1 * scen[i]) * y[k][i] for i in np.arange(env.N)) + theta))

    slack_final = {k: [] for k in np.arange(K)}
    sum_slack = sum(slack)
    for k in np.arange(K):
        if sum_slack == 0:
            slack_final[k].append(0)
        else:
            slack_final[k].append(slack[k]/sum_slack)

    return slack_final


def const_to_z_fun(K, scen, env, theta, x, y):
    # point-to-plane: https://mathworld.wolfram.com/Point-PlaneDistance.html
    # variable xi
    dist = []
    for k in np.arange(K):
        # CONST 1
        # define coefficients
        coeff_1 = [-0.1 * env.cost[i] * y[k][i] for i in np.arange(env.N)]
        # define constant
        const_1 = sum([env.cost[i] * y[k][i] for i in np.arange(env.N)]) - theta
        # take distance
        dist.append((sum([coeff_1[i] * scen[i] for i in np.arange(env.N)]) + const_1) / (np.sqrt(sum(coeff_1[i] ** 2 for i in np.arange(env.N)))))

    dist_final = {k: [] for k in np.arange(K)}
    sum_dist = sum(dist)
    for k in np.arange(K):
        if sum_dist == 0:
            dist_final[k].append(1)
        else:
            dist_final[k].append(dist[k]/sum_dist)

    return dist_final


def const_to_const_fun(K, scen, env, tau):
    # cosine similarity: https://en.wikipedia.org/wiki/Cosine_similarity
    # variable x and y
    # take minimum distance from existing scenarios
    cos = {k: [] for k in np.arange(K)}
    for k in np.arange(K):
        if len(tau[k]) == 0:
            cos[k].append(0)
            continue
        cos_tmp = []
        scen_vec_0 = [env.cost[i] * (1 - 0.1 * scen[i]) for i in np.arange(env.N)]
        for xi in tau[k]:
            # CONST
            xi_vec_0 = [env.cost[i] * (1 - 0.1 * xi[i]) for i in np.arange(env.N)]
            similarity = (sum([xi_vec_0[i]*scen_vec_0[i] for i in np.arange(env.N)])) / \
                         ((np.sqrt(sum(xi_vec_0[i] ** 2 for i in np.arange(env.N)))) *
                          (np.sqrt(sum(scen_vec_0[i] ** 2 for i in np.arange(env.N)))))
            cos_tmp.append(similarity)

        # select cos with most similarity, so max cos
        cos[k].append(max(cos_tmp))

    return {k: list(np.nan_to_num(cos[k], nan=0.0)) for k in np.arange(K)}



def att_index_maker(env, att_series):
    # create list of attributes
    att_index = []

    if "coords" in att_series:
        # index
        att_index.append(np.arange(env.xi_dim))
    if "obj_det" in att_series:
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 1))
        except:
            att_index.append(np.arange(1))
    if "y_det" in att_series:
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + env.y_dim))
        except:
            att_index.append(np.arange(env.y_dim))

    if "slack" in att_series:
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 1))
        except:
            att_index.append(np.arange(1))
    if "const_to_z_dist" in att_series:
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 1))
        except:
            att_index.append(np.arange(1))
    if "const_to_const_dist" in att_series:
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 1))
        except:
            att_index.append(np.arange(1))

    return att_index
