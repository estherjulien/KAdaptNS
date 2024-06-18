import numpy as np
import gurobipy as gp
from gurobipy import GRB

"""
Code for solving the MILP formulation of the knapsack problem 
of the master problem (given as scenario problem) and subproblem (scenario problem)
"""


# SCENARIO FUN = MASTER PROBLEM
# update master problem with new scenario in specified subset (k_new)
def scenario_fun_update(K, k_new, xi_new, env, scen_model):
    # load variables
    y = {k: {i: scen_model.getVarByName(f"y_{k}[{i}]") for i in range(env.N)} for k in range(K)}
    theta = scen_model.getVarByName("theta")

    # add new constraints
    # objective constraint
    scen_model.addConstr((gp.quicksum(env.cost[i]*(1 - 0.5*xi_new[i])*y[k_new][i] for i in range(env.N))) >= theta)

    # solve
    scen_model.optimize()
    y_sol = dict()
    for k in range(K):
        y_sol[k] = np.array([var.X for i, var in y[k].items()])
    theta_sol = theta.X
    return theta_sol, None, y_sol, scen_model


# build master problem
def scenario_fun_build(K, tau, env, gp_env):
    scen_model = gp.Model("Scenario-Based K-Adaptability Problem", env=gp_env)
    # variables
    theta = scen_model.addVar(lb=0, ub=-env.lower_bound, name="theta")
    y = dict()
    for k in range(K):
        y[k] = scen_model.addVars(env.y_dim, vtype=GRB.BINARY, name=f"y_{k}")

    # objective function
    scen_model.setObjective(theta, GRB.MAXIMIZE)

    # constraints
    for k in range(K):
        for xi in tau[k]:
            # objective constraint
            scen_model.addConstr(gp.quicksum(env.cost[i]*(1 - 0.5 * xi[i])*y[k][i] for i in range(env.N)) >= theta)
        # budget constraint
        scen_model.addConstr(gp.quicksum(y[k][i]*env.weight[i] for i in range(env.N)) <= env.budget)

    # solve
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    y_sol = dict()
    for k in range(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = theta.X
    return theta_sol, None, y_sol, scen_model


# SEPARATION FUN = SUBPROBLEM
def separation_fun(K, x, y, theta, env, tau, gp_env):
    # model
    sep_model = gp.Model("Separation Problem", env=gp_env)
    sep_model.Params.OutputFlag = 0
    # variables
    zeta = sep_model.addVar(lb=env.lower_bound, name="zeta")
    xi = sep_model.addVars(env.xi_dim, lb=0, ub=1, name="xi")

    # objective function
    sep_model.setObjective(zeta, GRB.MAXIMIZE)
    # objective constraint
    for k in range(K):
        if len(tau[k]) > 0:
            sep_model.addConstr(zeta <= -(gp.quicksum(env.cost[i]*(1 - 0.5 * xi[i])*y[k][i] for i in range(env.N)) -
                                theta))
    # uncertainty set
    sep_model.addConstr(gp.quicksum(xi[i] for i in range(env.N)) <= env.gamma)

    # solve
    sep_model.optimize()
    zeta_sol = zeta.X
    xi_sol = np.array([var.X for i, var in xi.items()])
    return zeta_sol, xi_sol

