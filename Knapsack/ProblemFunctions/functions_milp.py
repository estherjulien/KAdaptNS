import gurobipy as gp
from gurobipy import GRB
from ProblemFunctions.Env import *


def scenario_fun_update(K, k_new, xi_new, env, scen_model):
    # load variables
    y = {k: {i: scen_model.getVarByName(f"y_{k}[{i}]") for i in np.arange(env.N)} for k in np.arange(K)}
    theta = scen_model.getVarByName("theta")

    # add new constraints
    # objective constraint
    scen_model.addConstr((gp.quicksum(env.cost[i]*(1 - 0.5*xi_new[i])*y[k_new][i] for i in np.arange(env.N))) >= theta)
    # scen_model.addConstr((gp.quicksum((env.cost[i] - 2 * xi_new[i])*y[k_new][i] for i in np.arange(env.N))) >= theta)

    # solve
    scen_model.optimize()
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = np.array([var.X for i, var in y[k].items()])
    theta_sol = theta.X
    return theta_sol, [], y_sol, scen_model


def scenario_fun_build(K, tau, env):
    scen_model = gp.Model("Scenario-Based K-Adaptability Problem")
    # variables
    theta = scen_model.addVar(lb=0, ub=-env.lower_bound, name="theta")
    y = dict()
    for k in np.arange(K):
        y[k] = scen_model.addVars(env.y_dim, vtype=GRB.BINARY, name=f"y_{k}")

    # objective function
    scen_model.setObjective(theta, GRB.MAXIMIZE)

    # constraints
    for k in np.arange(K):
        for xi in tau[k]:
            # objective constraint
            scen_model.addConstr(gp.quicksum(env.cost[i]*(1 - 0.5 * xi[i])*y[k][i] for i in np.arange(env.N)) >= theta)
            # scen_model.addConstr(gp.quicksum((env.cost[i] - 2 * xi[i])*y[k][i] for i in np.arange(env.N)) >= theta)
        # budget constraint
        scen_model.addConstr(gp.quicksum(y[k][i]*env.weight[i] for i in np.arange(env.N)) <= env.budget)

    # solve
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = theta.X
    return theta_sol, [], y_sol, scen_model


def separation_fun(K, x, y, theta, env, tau):
    # model
    sep_model = gp.Model("Separation Problem")
    sep_model.Params.OutputFlag = 0
    # variables
    zeta = sep_model.addVar(lb=env.lower_bound, name="zeta")
    xi = sep_model.addVars(env.xi_dim, lb=0, ub=1, name="xi")

    # objective function
    sep_model.setObjective(zeta, GRB.MAXIMIZE)
    # objective constraint
    for k in np.arange(K):
        if len(tau[k]) > 0:
            sep_model.addConstr(zeta <= -(gp.quicksum(env.cost[i]*(1 - 0.5 * xi[i])*y[k][i] for i in np.arange(env.N)) -
                                theta))
            # sep_model.addConstr(zeta <= -(gp.quicksum((env.cost[i] - 2 * xi[i])*y[k][i] for i in np.arange(env.N)) -
            #                     theta))
    # uncertainty set
    sep_model.addConstr(gp.quicksum(xi[i] for i in np.arange(env.N)) <= env.gamma)

    # solve
    sep_model.optimize()
    zeta_sol = zeta.X
    xi_sol = np.array([var.X for i, var in xi.items()])
    return zeta_sol, xi_sol


# SUB TREE MODEL
def scenario_fun_update_sub_tree(K, new_node, xi_dict, env, scen_model=None):
    # use same model and just add new constraint
    # load variables
    y = {k: {i: scen_model.getVarByName(f"y_{k}[{i}]") for i in np.arange(env.N)} for k in np.arange(K)}
    theta = scen_model.getVarByName("theta")

    for node_sec in np.arange(1, len(new_node)):
        xi_new = xi_dict[new_node[:node_sec]]
        k_new = new_node[node_sec]
        # add new constraints
        # objective constraint
        scen_model.addConstr(gp.quicksum(env.cost[i]*(1 - 0.5*xi_new[i])*y[k_new][i] for i in np.arange(env.N))
                             >= theta, name=f"const1_{new_node[:node_sec]}_{k_new}")
        # scen_model.addConstr(gp.quicksum((env.cost[i] - 2*xi_new[i]) *y[k_new][i] for i in np.arange(env.N))
        #                      >= theta, name=f"const1_{new_node[:node_sec]}_{k_new}")
    scen_model.update()

    # solve
    scen_model.optimize()
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = np.array([var.X for i, var in y[k].items()])
    theta_sol = theta.X

    # delete constraints
    for node_sec in np.arange(1, len(new_node)):
        xi_found = new_node[:node_sec]
        k_new = new_node[node_sec]
        scen_model.remove(scen_model.getConstrByName(f"const1_{xi_found}_{k_new}"))

    scen_model.update()

    return theta_sol, [], y_sol


# SCENARIO FUN DETERMINISTIC ATTRIBUTES
def scenario_fun_deterministic_build(env):
    smd = gp.Model("Scenario-Based K-Adaptability Problem")
    # variables
    theta = smd.addVar(lb=0, ub=-env.lower_bound, name="theta")
    y = smd.addVars(env.y_dim, vtype=GRB.BINARY, name="y")

    # objective function
    smd.setObjective(theta, GRB.MAXIMIZE)

    # other constraints
    smd.addConstr(gp.quicksum(y[i]*env.weight[i] for i in np.arange(env.N)) <= env.budget)

    # solve model
    smd.Params.OutputFlag = 0
    smd.optimize()

    return smd


def scenario_fun_deterministic_update(env, scen, smd):
    N = env.N

    y = {i: smd.getVarByName(f"y[{i}]") for i in np.arange(N)}
    theta = smd.getVarByName("theta")

    # constraints
    # objective constraint
    smd.addConstr(gp.quicksum(env.cost[i] * (1 - 0.5 * scen[i]) * y[i] for i in np.arange(env.N)) >= theta, name="new_const_1")
    # smd.addConstr(gp.quicksum((env.cost[i] - 2*scen[i]) * y[i] for i in np.arange(env.N)) >= theta, name="new_const_1")

    smd.update()
    # solve
    smd.optimize()
    y_sol = [var.X for i, var in y.items()]
    theta_sol = theta.X

    # delete new constraint
    smd.remove(smd.getConstrByName("new_const_1"))

    smd.update()

    return theta_sol, [], y_sol

