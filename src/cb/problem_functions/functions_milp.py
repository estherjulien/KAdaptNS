from cb.problem_functions.environment import *
from gurobipy import GRB
import gurobipy as gp
import numpy as np


"""
Code for solving the MILP formulation of the capital budgeting problem 
of the master problem (given as scenario problem) and subproblem (scenario problem)
"""


# SCENARIO FUN = MASTER PROBLEM
# update master problem with new scenario in specified subset (k_new)
def scenario_fun_update(K, k_new, xi_new, env, scen_model, delete_added_cons=False):
    projects = env.projects
    N = env.N

    # load variables
    x_0 = scen_model.getVarByName("x0")
    x = {p: scen_model.getVarByName(f"x[{p}]") for p in range(N)}
    y_0 = {k: scen_model.getVarByName(f"y0[{k}]") for k in range(K)}
    y = {k: {p: scen_model.getVarByName(f"y_{k}[{p}]") for p in range(N)} for k in range(K)}
    theta = scen_model.getVarByName("theta")

    if delete_added_cons:
        name1 = "obj_cons"
        name2 = "budg_cons1"
        name3 = "budg_cons2"
    else:
        name1 = ""
        name2 = ""
        name3 = ""

    # add new constraints
    # objective constraint
    scen_model.addConstr(-(gp.quicksum(rev_fun(projects[p], xi_new) * (x[p] + env.kappa * y[k_new][p])
                                       for p in range(N)) - env.lam * (x_0 + env.mu * y_0[k_new])) <= theta, name=name1)
    # budget constraint
    scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi_new) * (x[p]) for p in range(N))
                         <= env.budget + x_0, name=name2)
    scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi_new) * (x[p] + y[k_new][p]) for p in range(N))
                         <= (env.budget + x_0 + y_0[k_new]), name=name3)

    # solve
    scen_model.optimize()
    x_0_sol = x_0.X
    x_sol = np.array([var.X for i, var in x.items()])
    y_0_sol = np.array([var.X for i, var in y_0.items()])
    y_sol = dict()
    for k in range(K):
        y_sol[k] = np.array([var.X for i, var in y[k].items()])
    theta_sol = theta.X

    if delete_added_cons:
        scen_model.remove(scen_model.getConstrByName(name1))
        scen_model.remove(scen_model.getConstrByName(name2))
        scen_model.remove(scen_model.getConstrByName(name3))
        scen_model.update()

    return theta_sol, [x_0_sol, x_sol], [y_0_sol, y_sol], scen_model


# build master problem
def scenario_fun_build(K, tau, env, gp_env):
    projects = env.projects
    N = env.N
    scen_model = gp.Model("Scenario-Based K-Adaptability Problem", env=gp_env)
    # variables
    theta = scen_model.addVar(lb=-env.lower_bound, ub=0, name="theta")
    x_0 = scen_model.addVar(lb=0, name="x0")
    x = scen_model.addVars(N, vtype=GRB.BINARY, name="x")
    y = dict()
    for k in range(K):
        y[k] = scen_model.addVars(N, vtype=GRB.BINARY, name=f"y_{k}")
    y_0 = scen_model.addVars(K, lb=0, name="y0")

    # objective function
    scen_model.setObjective(theta, GRB.MINIMIZE)

    # constraints
    for k in range(K):
        for xi in tau[k]:
            # objective constraint
            try:
                scen_model.addConstr(-(gp.quicksum(rev_fun(projects[p], xi) * (x[p] + env.kappa*y[k][p])
                                                   for p in range(N)) - env.lam*(x_0 + env.mu*y_0[k])) <= theta)
            except IndexError:
                pass
            # budget constraint
            scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi) * (x[p]) for p in range(N))
                                 <= env.budget + 0)
            scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi) * (x[p] + y[k][p]) for p in range(N))
                                 <= (env.budget + x_0 + y_0[k]))

        # other constraints
        scen_model.addConstrs(x[p] + y[k][p] <= 1 for p in range(N))

    # solve
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    x_0_sol = x_0.X
    x_sol = np.array([var.X for i, var in x.items()])
    y_0_sol = np.array([var.X for i, var in y_0.items()])
    y_sol = dict()
    for k in range(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = theta.X

    return theta_sol, [x_0_sol, x_sol], [y_0_sol, y_sol], scen_model


# SEPARATION FUN = SUBPROBLEM
def separation_fun(K, x_input, y_input, theta, env, tau, gp_env):
    x_0, x = x_input
    y_0, y = y_input
    N = env.N
    projects = env.projects
    # model
    sep_model = gp.Model("Separation Problem", env=gp_env)
    sep_model.Params.OutputFlag = 0
    # variables
    zeta = sep_model.addVar(lb=-env.bigM, name="zeta")
    xi = sep_model.addVars(env.xi_dim, lb=-1, ub=1, name="xi")
    z_index = [(k, i) for k in range(K) for i in [0, 1, 2]]
    z = sep_model.addVars(z_index, name="z", vtype=GRB.BINARY)

    # objective function
    sep_model.setObjective(zeta, GRB.MAXIMIZE)
    # z constraint
    sep_model.addConstrs(gp.quicksum(z[k, l] for l in range(3)) == 1 for k in range(K))
    # objective constraint
    for k in range(K):
        if len(tau[k]) > 0:
            sep_model.addConstr((zeta + env.bigM*z[k, 2] <= -(gp.quicksum(rev_fun(projects[p], xi) *
                                                            (x[p] + env.kappa*y[k][p]) for p in range(N)) -
                                                            env.lam*(x_0 + env.mu*y_0[k])) - theta + env.bigM))
            # budget constraints
            sep_model.addConstr((zeta + env.bigM*z[k, 1] <= gp.quicksum(cost_fun(projects[p], xi) * (x[p])
                                                                        for p in range(N)) - env.budget - x_0 + env.bigM))
            sep_model.addConstr((zeta + env.bigM*z[k, 0] <= gp.quicksum(cost_fun(projects[p], xi) * (x[p] + y[k][p])
                                                                        for p in range(N)) - env.budget - x_0 - y_0[k] + env.bigM))
    # solve
    sep_model.optimize()
    zeta_sol = zeta.X
    xi_sol = np.array([var.X for i, var in xi.items()])

    return zeta_sol, xi_sol


# SUB TREE MODEL (for training data generation)
def scenario_fun_update_sub_tree(K, new_node, xi_dict, env, scen_model=None):
    # use same model and just add new constraint
    projects = env.projects
    N = env.N

    # load variables
    x_0 = scen_model.getVarByName("x0")
    x = {p: scen_model.getVarByName(f"x[{p}]") for p in range(N)}
    y_0 = {k: scen_model.getVarByName(f"y0[{k}]") for k in range(K)}
    y = {k: {p: scen_model.getVarByName(f"y_{k}[{p}]") for p in range(N)} for k in range(K)}
    theta = scen_model.getVarByName("theta")

    for node_sec in range(1, len(new_node)):
        xi_new = xi_dict[new_node[:node_sec]]
        k_new = new_node[node_sec]
        # add new constraints
        # objective constraint
        scen_model.addConstr(-(gp.quicksum(rev_fun(projects[p], xi_new) * (x[p] + env.kappa * y[k_new][p])
                                           for p in range(N)) - env.lam * (x_0 + env.mu * y_0[k_new])) <= theta
                             , name=f"const1_{new_node[:node_sec]}_{k_new}")
        # budget constraint
        scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi_new) * (x[p]) for p in range(N))
                             <= env.budget + x_0, name=f"const2_{new_node[:node_sec]}_{k_new}")
        scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi_new) * (x[p] + y[k_new][p]) for p in range(N))
                             <= (env.budget + x_0 + y_0[k_new]), name=f"const3_{new_node[:node_sec]}_{k_new}")

    scen_model.update()

    # solve
    scen_model.optimize()
    x_0_sol = x_0.X
    x_sol = np.array([var.X for i, var in x.items()])
    y_0_sol = np.array([var.X for i, var in y_0.items()])
    y_sol = dict()
    for k in range(K):
        y_sol[k] = np.array([var.X for i, var in y[k].items()])
    theta_sol = theta.X

    # delete constraints
    for node_sec in range(1, len(new_node)):
        xi_found = new_node[:node_sec]
        k_new = new_node[node_sec]
        scen_model.remove(scen_model.getConstrByName(f"const1_{xi_found}_{k_new}"))
        scen_model.remove(scen_model.getConstrByName(f"const2_{xi_found}_{k_new}"))
        scen_model.remove(scen_model.getConstrByName(f"const3_{xi_found}_{k_new}"))

    scen_model.update()

    return theta_sol, [x_0_sol, x_sol], [y_0_sol, y_sol]


# STATIC SOLUTION ROBUST COUNTERPART (for features)
def static_solution_rc(env, gp_env):
    print("     Making model")
    src = gp.Model("Scenario-Based K-Adaptability Problem", env=gp_env)
    projects = env.projects
    N = env.N
    # variables
    print("     Making variables")
    theta = src.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
    x = src.addVars(N, vtype=GRB.BINARY, name="x")
    x_0 = src.addVar(lb=0, name="x0")
    y = src.addVars(N, vtype=GRB.BINARY, name="y")
    y_0 = src.addVar(lb=0, name="y0")

    # dual variables
    # objective
    d_1_a = src.addVars(env.xi_dim, lb=0)
    d_1_b = src.addVars(env.xi_dim, lb=0)
    # first-stage constraint
    d_2_a = src.addVars(env.xi_dim, lb=0)
    d_2_b = src.addVars(env.xi_dim, lb=0)
    # second-stage constraint
    d_3_a = src.addVars(env.xi_dim, lb=0)
    d_3_b = src.addVars(env.xi_dim, lb=0)

    # objective function
    print("     Setting objective")
    src.setObjective(theta, GRB.MAXIMIZE)

    # CONSTRAINTS
    print("     Adding constraints")
    # deterministic constraints
    src.addConstrs(x[p] + y[p] <= 1 for p in range(N))

    # uncertain constraints
    # objective constraint
    src.addConstr(gp.quicksum(projects[p].rev_nom*(x[p] + env.kappa*y[p]) for p in range(N)) -
                  gp.quicksum(d_1_a[j] + d_1_b[j] for j in range(env.xi_dim))
                  >= theta + env.lam * (x_0 + env.mu * y_0))
    src.addConstrs((1/2*gp.quicksum(projects[p].rev_nom*projects[p].psi[j]*(x[p] + env.kappa*y[p]) for p in range(N))
                   - d_1_a[j] + d_1_b[j] == 0) for j in range(env.xi_dim))
    # first-stage constraint
    src.addConstr(gp.quicksum(projects[p].cost_nom*x[p] for p in range(N)) +
                  gp.quicksum(d_2_a[j] + d_2_b[j] for j in range(env.xi_dim)) <= env.budget + x_0)
    src.addConstrs((1/2*gp.quicksum(projects[p].cost_nom*projects[p].phi[j]*x[p] for p in range(N))
                   + d_2_a[j] - d_2_b[j] == 0) for j in range(env.xi_dim))
    # second-stage constraints
    src.addConstr(gp.quicksum(projects[p].cost_nom*(x[p] + y[p]) for p in range(N)) +
                  gp.quicksum(d_3_a[j] + d_3_b[j] for j in range(env.xi_dim)) <= env.budget + x_0 + y_0)
    src.addConstrs((1/2*gp.quicksum(projects[p].cost_nom*projects[p].phi[j]*x[p] for p in range(N))
                   + d_3_a[j] - d_3_b[j] == 0) for j in range(env.xi_dim))

    # solve
    print("     Solve")
    src.Params.OutputFlag = 0
    src.optimize()
    x_0_sol = x_0.X
    x_sol = np.array([var.X for i, var in x.items()])
    print("     Return results")

    return [x_0_sol, x_sol]


# SCENARIO FUN STATIC ATTRIBUTES (for features)
# build problem
def scenario_fun_static_build(env, x_input, gp_env):
    x_0, x = x_input
    N = env.N
    sms = gp.Model("Scenario-Based K-Adaptability Problem", env=gp_env)
    # variables
    theta = sms.addVar(lb=-env.lower_bound, ub=0, name="theta")
    y_0 = sms.addVar(lb=0, name="y0")
    y = sms.addVars(N, vtype=GRB.BINARY, name="y")

    # objective function
    sms.setObjective(theta, GRB.MINIMIZE)

    # other constraints
    sms.addConstrs(x[p] + y[p] <= 1 for p in range(N))

    # solve
    sms.Params.OutputFlag = 0
    sms.optimize()

    return sms


# update problem with new scenario
def scenario_fun_static_update(env, scen, x_input, sms):
    x_0, x = x_input
    projects = env.projects
    N = env.N

    # old variables
    y = {p: sms.getVarByName(f"y[{p}]") for p in range(N)}
    y_0 = sms.getVarByName("y0")
    theta = sms.getVarByName("theta")

    # constraints
    # objective constraint
    sms.addConstr(-(gp.quicksum(rev_fun(projects[p], scen) * (x[p] + env.kappa*y[p])
                                for p in range(N)) - env.lam*(x_0 + env.mu*y_0)) <= theta, name="new_const_1")
    # budget constraint
    sms.addConstr(gp.quicksum(cost_fun(projects[p], scen) * (x[p]) for p in range(N))
                  <= env.budget + x_0, name="new_const_2")
    sms.addConstr(gp.quicksum(cost_fun(projects[p], scen) * (x[p] + y[p]) for p in range(N))
                  <= (env.budget + x_0 + y_0), name="new_const_3")

    sms.update()
    # solve
    sms.optimize()
    y_0_sol = y_0.X
    y_sol = [var.X for i, var in y.items()]
    theta_sol = theta.X

    # delete new constraint
    sms.remove(sms.getConstrByName("new_const_1"))
    sms.remove(sms.getConstrByName("new_const_2"))
    sms.remove(sms.getConstrByName("new_const_3"))

    sms.update()

    return theta_sol, [y_0_sol, y_sol]


# SCENARIO FUN DETERMINISTIC ATTRIBUTES (for features)
# build problem
def scenario_fun_deterministic_build(env, gp_env):
    smd = gp.Model("Scenario-Based K-Adaptability Problem", env=gp_env)
    projects = env.projects
    N = env.N
    # variables
    theta = smd.addVar(lb=-env.lower_bound, ub=0, name="theta")
    x_0 = smd.addVar(lb=0, name="x0")
    x = smd.addVars(N, vtype=GRB.BINARY, name="x")
    y = smd.addVars(N, vtype=GRB.BINARY, name="y")
    y_0 = smd.addVar(lb=0, name="y0")

    # objective function
    smd.setObjective(theta, GRB.MINIMIZE)

    # other constraints
    smd.addConstrs(x[p] + y[p] <= 1 for p in range(N))

    # solve model
    smd.Params.OutputFlag = 0
    smd.optimize()

    return smd


# update problem with new scenario
def scenario_fun_deterministic_update(env, scen, smd):
    projects = env.projects
    N = env.N

    x = {p: smd.getVarByName(f"x[{p}]") for p in range(N)}
    x_0 = smd.getVarByName("x0")
    y = {p: smd.getVarByName(f"y[{p}]") for p in range(N)}
    y_0 = smd.getVarByName("y0")
    theta = smd.getVarByName("theta")

    # constraints
    # objective constraint
    smd.addConstr(-(gp.quicksum(rev_fun(projects[p], scen) * (x[p] + env.kappa*y[p])
                                for p in range(N)) - env.lam*(x_0 + env.mu*y_0)) <= theta, name="new_const_1")
    # budget constraint
    smd.addConstr(gp.quicksum(cost_fun(projects[p], scen) * (x[p]) for p in range(N))
                  <= env.budget + x_0, name="new_const_2")
    smd.addConstr(gp.quicksum(cost_fun(projects[p], scen) * (x[p] + y[p]) for p in range(N))
                  <= (env.budget + x_0 + y_0), name="new_const_3")

    smd.update()
    # solve
    smd.optimize()
    x_0_sol = x_0.X
    x_sol = [var.X for i, var in x.items()]
    y_0_sol = y_0.X
    y_sol = [var.X for i, var in y.items()]
    theta_sol = theta.X

    # delete new constraint
    smd.remove(smd.getConstrByName("new_const_1"))
    smd.remove(smd.getConstrByName("new_const_2"))
    smd.remove(smd.getConstrByName("new_const_3"))

    smd.update()

    return theta_sol, [x_0_sol, x_sol], [y_0_sol, y_sol]
