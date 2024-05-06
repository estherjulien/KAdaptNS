from sp.problem_functions.functions_milp import *
import numpy as np
import copy

"""
Code related to features for the shortest path problem
    1. predict_subset()     - predicting the right subset based on the features
    2. state_features()     - make state features for a node
    3. input_fun()          - combine state and scenario attributes to make input features
    4. attribute_per_scen() - assign attributes to the given scenario
    5. slack_fun()          - get slack attribute values of the given scenario, used in attribute_per_scen()
    6. const_to_z_fun()     - get scenario distance attribute values of the given scenario, used in attribute_per_scen()
    7. const_to_const_fun() - get constraint distance attribute values of the given scenario, used in attribute_per_scen()
    8. att_index_maker()    - to avoid using dataframes, we assign indices to the stored attribute vectors
"""


# predicting the right subset based on the features
def predict_subset(K, tau_att, scen_att, scen_att_k, success_model, att_index, state_features, scales=None):
    X = input_fun(K, state_features, tau_att, scen_att, scen_att_k, att_index)

    if scales is not None and len(scales):
        X = (X - scales[0].values) / (scales[1] - scales[0]).values
        if any((scales[1] - scales[0]).values == 0):
            culprits = np.where((scales[1] - scales[0]).values == 0)[0]
            for i in culprits:
                X[:, i] = 0

    pred_tmp = success_model.predict_proba(X)
    success_prediction = np.array([i[1] for i in pred_tmp])
    order = np.argsort(success_prediction)
    return order[::-1], success_prediction


# make state features for a node
def state_features(theta, zeta, depth, zeta_i, theta_pre, zeta_pre,
                   depth_i=None, depth_i_alt=None,
                   theta_i=None, theta_i_alt=None, return_all=False):
    # objective, objective compared to previous, violation
    if depth_i is None:
        features = [theta, theta / theta_pre, zeta]
    else:
        features = [theta / theta_i, theta / theta_pre, zeta / zeta_i]
    # violation compared to previous
    try:
        features.append(zeta/zeta_pre)
    except ZeroDivisionError:
        features.append(1)
    # depth
    if return_all:
        # DATA GENERATION
        features_alt = copy.deepcopy(features)
        features_no = copy.deepcopy(features)

        features.append(depth/depth_i)
        features_alt.append(depth/depth_i_alt)
        features_no.append(depth)

        features_alt[0] = theta / theta_i_alt
        features_no[0] = theta
        features_no[2] = zeta
        return [np.array(features), np.array(features_alt), np.array(features_no)]
    elif depth_i is None:
        features.append(depth)
    else:
        features.append(depth/depth_i)
    return np.array(features)


# combine state and scenario attributes to make input features
def input_fun(K, state_features, tau_att, scen_att_pre, scen_att_k, att_index, full_list=None, return_all=False):
    if full_list is None:
        full_list = range(K)
    att_num = len(att_index)
    scen_att = {k: np.array(scen_att_pre + scen_att_k[k]) for k in range(K)}

    diff_info = {k: [] for k in np.arange(K)}
    for k in np.arange(K):
        for att_type in np.arange(att_num):
            diff_info[k].append(np.linalg.norm(np.mean(tau_att[k][:, att_index[att_type]], axis=0) - scen_att[k][att_index[att_type]]) / len(att_index[att_type]))

    if return_all:
        X = []
        for st_feat in state_features:
            X.append(np.array([np.hstack([st_feat, diff_info[k]]) for k in full_list]))
    else:
        X = np.array([np.hstack([state_features, diff_info[k]]) for k in full_list])
    return X


# assign attributes to the given scenario
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

    # static problem doesn't exist for shortest path

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


# get slack attribute values of the given scenario, used in attribute_per_scen()
def slack_fun(K, scen, env, theta, x, y):
    slack = []
    for k in np.arange(K):
        slack.append(abs(sum((1 + scen[a] / 2) * env.distances_array[a] * y[k][a] for a in np.arange(env.num_arcs)) - theta))

    slack_final = {k: [] for k in np.arange(K)}
    sum_slack = sum(slack)
    for k in np.arange(K):
        if sum_slack == 0:
            slack_final[k].append(0)
        else:
            slack_final[k].append(slack[k]/sum_slack)

    return slack_final


# get scenario distance attribute values of the given scenario, used in attribute_per_scen()
def const_to_z_fun(K, scen, env, theta, x, y):
    # point-to-plane: https://mathworld.wolfram.com/Point-PlaneDistance.html
    # variable xi
    dist = []
    for k in np.arange(K):
        # CONST 1
        # define coefficients
        coeff_1 = [1 / 2 * env.distances_array[a] * (y[k][a]) for a in np.arange(env.num_arcs)]
        # define constant
        const_1 = sum([env.distances_array[a] * (y[k][a]) for a in np.arange(env.num_arcs)]) - theta
        # take distance
        dist.append((sum([coeff_1[a] * scen[a] for a in np.arange(env.num_arcs)]) + const_1) / (np.sqrt(sum(coeff_1[a] ** 2 for a in np.arange(env.num_arcs)))))

    dist_final = {k: [] for k in np.arange(K)}
    sum_dist = sum(dist)
    for k in np.arange(K):
        if sum_dist == 0:
            dist_final[k].append(1)
        else:
            dist_final[k].append(dist[k]/sum_dist)

    return dist_final


# get constraint distance attribute values of the given scenario, used in attribute_per_scen()
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
        scen_vec_0 = [(1 + scen[a]/2)*env.distances_array[a] for a in np.arange(env.num_arcs)]
        for xi in tau[k]:
            # CONST
            xi_vec_0 = [(1 + xi[a]/2)*env.distances_array[a] for a in np.arange(env.num_arcs)]
            similarity = (sum([xi_vec_0[a]*scen_vec_0[a] for a in np.arange(env.num_arcs)])) / \
                         ((np.sqrt(sum(xi_vec_0[a] ** 2 for a in np.arange(env.num_arcs)))) *
                          (np.sqrt(sum(scen_vec_0[a] ** 2 for a in np.arange(env.num_arcs)))))
            cos_tmp.append(similarity)

        # select cos with most similarity, so max cos
        cos[k].append(max(cos_tmp))

    return {k: list(np.nan_to_num(cos[k], nan=0.0)) for k in np.arange(K)}


# to avoid using dataframes, we assign indices to the stored attribute vectors
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
