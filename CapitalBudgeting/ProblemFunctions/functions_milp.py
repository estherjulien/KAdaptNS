import gurobipy as gp
from gurobipy import GRB
from ProblemFunctions.Env import *


"""
Code for solving the MILP formulation of the capital budgeting problem 
of the master problem (given as scenario problem) and subproblem (scenario problem)
"""


# SCENARIO FUN = MASTER PROBLEM
def scenario_fun_update(K, k_new, xi_new, env, scen_model):
    projects = env.projects
    N = env.N

    # load variables
    x_0 = scen_model.getVarByName("x0")
    x = {p: scen_model.getVarByName(f"x[{p}]") for p in np.arange(N)}
    y_0 = {k: scen_model.getVarByName(f"y0[{k}]") for k in np.arange(K)}
    y = {k: {p: scen_model.getVarByName(f"y_{k}[{p}]") for p in np.arange(N)} for k in np.arange(K)}
    theta = scen_model.getVarByName("theta")

    # add new constraints
    # objective constraint
    scen_model.addConstr(-(gp.quicksum(rev_fun(projects[p], xi_new) * (x[p] + env.kappa * y[k_new][p])
                                       for p in np.arange(N)) - env.lam * (x_0 + env.mu * y_0[k_new])) <= theta)
    # budget constraint
    scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi_new) * (x[p]) for p in np.arange(N))
                         <= env.budget + x_0)
    scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi_new) * (x[p] + y[k_new][p]) for p in np.arange(N))
                         <= (env.budget + x_0 + y_0[k_new]))

    # solve
    scen_model.optimize()
    x_0_sol = x_0.X
    x_sol = np.array([var.X for i, var in x.items()])
    y_0_sol = np.array([var.X for i, var in y_0.items()])
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = np.array([var.X for i, var in y[k].items()])
    theta_sol = theta.X

    return theta_sol, [x_0_sol, x_sol], [y_0_sol, y_sol], scen_model


def scenario_fun_build(K, tau, env):
    projects = env.projects
    N = env.N
    scen_model = gp.Model("Scenario-Based K-Adaptability Problem")
    # variables
    theta = scen_model.addVar(lb=-env.lower_bound, ub=0, name="theta")
    x_0 = scen_model.addVar(lb=0, name="x0")
    x = scen_model.addVars(N, vtype=GRB.BINARY, name="x")
    y = dict()
    for k in np.arange(K):
        y[k] = scen_model.addVars(N, vtype=GRB.BINARY, name=f"y_{k}")
    y_0 = scen_model.addVars(K, lb=0, name="y0")

    # objective function
    scen_model.setObjective(theta, GRB.MINIMIZE)

    # constraints
    for k in np.arange(K):
        for xi in tau[k]:
            # objective constraint
            try:
                scen_model.addConstr(-(gp.quicksum(rev_fun(projects[p], xi) * (x[p] + env.kappa*y[k][p])
                                                   for p in np.arange(N)) - env.lam*(x_0 + env.mu*y_0[k])) <= theta)
            except IndexError:
                pass
            # budget constraint
            scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi) * (x[p]) for p in np.arange(N))
                                 <= env.budget + 0)
            scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi) * (x[p] + y[k][p]) for p in np.arange(N))
                                 <= (env.budget + x_0 + y_0[k]))

        # other constraints
        scen_model.addConstrs(x[p] + y[k][p] <= 1 for p in np.arange(N))

    # solve
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    x_0_sol = x_0.X
    x_sol = np.array([var.X for i, var in x.items()])
    y_0_sol = np.array([var.X for i, var in y_0.items()])
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = theta.X

    return theta_sol, [x_0_sol, x_sol], [y_0_sol, y_sol], scen_model


# SEPARATION FUN = SUBPROBLEM
def separation_fun(K, x_input, y_input, theta, env, tau):
    x_0, x = x_input
    y_0, y = y_input
    N = env.N
    projects = env.projects
    # model
    sep_model = gp.Model("Separation Problem")
    sep_model.Params.OutputFlag = 0
    # variables
    zeta = sep_model.addVar(lb=-env.bigM, name="zeta")
    xi = sep_model.addVars(env.xi_dim, lb=-1, ub=1, name="xi")
    z_index = [(k, i) for k in np.arange(K) for i in [0, 1, 2]]
    z = sep_model.addVars(z_index, name="z", vtype=GRB.BINARY)

    # objective function
    sep_model.setObjective(zeta, GRB.MAXIMIZE)
    # z constraint
    sep_model.addConstrs(gp.quicksum(z[k, l] for l in np.arange(3)) == 1 for k in np.arange(K))
    # objective constraint
    for k in np.arange(K):
        if len(tau[k]) > 0:
            sep_model.addConstr((zeta + env.bigM*z[k, 2] <= -(gp.quicksum(rev_fun(projects[p], xi) *
                                                            (x[p] + env.kappa*y[k][p]) for p in np.arange(N)) -
                                                            env.lam*(x_0 + env.mu*y_0[k])) - theta + env.bigM))
            # budget constraints
            sep_model.addConstr((zeta + env.bigM*z[k, 1] <= gp.quicksum(cost_fun(projects[p], xi) * (x[p])
                                                                        for p in np.arange(N)) - env.budget - x_0 + env.bigM))
            sep_model.addConstr((zeta + env.bigM*z[k, 0] <= gp.quicksum(cost_fun(projects[p], xi) * (x[p] + y[k][p])
                                                                        for p in np.arange(N)) - env.budget - x_0 - y_0[k] + env.bigM))
    # solve
    sep_model.optimize()
    zeta_sol = zeta.X
    xi_sol = np.array([var.X for i, var in xi.items()])

    return zeta_sol, xi_sol


# SUB TREE MODEL
def scenario_fun_update_sub_tree(K, new_node, xi_dict, env, scen_model=None):
    # use same model and just add new constraint
    projects = env.projects
    N = env.N

    # load variables
    x_0 = scen_model.getVarByName("x0")
    x = {p: scen_model.getVarByName(f"x[{p}]") for p in np.arange(N)}
    y_0 = {k: scen_model.getVarByName(f"y0[{k}]") for k in np.arange(K)}
    y = {k: {p: scen_model.getVarByName(f"y_{k}[{p}]") for p in np.arange(N)} for k in np.arange(K)}
    theta = scen_model.getVarByName("theta")

    for node_sec in np.arange(1, len(new_node)):
        xi_new = xi_dict[new_node[:node_sec]]
        k_new = new_node[node_sec]
        # add new constraints
        # objective constraint
        scen_model.addConstr(-(gp.quicksum(rev_fun(projects[p], xi_new) * (x[p] + env.kappa * y[k_new][p])
                                           for p in np.arange(N)) - env.lam * (x_0 + env.mu * y_0[k_new])) <= theta
                             , name=f"const1_{new_node[:node_sec]}_{k_new}")
        # budget constraint
        scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi_new) * (x[p]) for p in np.arange(N))
                             <= env.budget + x_0, name=f"const2_{new_node[:node_sec]}_{k_new}")
        scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi_new) * (x[p] + y[k_new][p]) for p in np.arange(N))
                             <= (env.budget + x_0 + y_0[k_new]), name=f"const3_{new_node[:node_sec]}_{k_new}")

    scen_model.update()

    # solve
    scen_model.optimize()
    x_0_sol = x_0.X
    x_sol = np.array([var.X for i, var in x.items()])
    y_0_sol = np.array([var.X for i, var in y_0.items()])
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = np.array([var.X for i, var in y[k].items()])
    theta_sol = theta.X

    # delete constraints
    for node_sec in np.arange(1, len(new_node)):
        xi_found = new_node[:node_sec]
        k_new = new_node[node_sec]
        scen_model.remove(scen_model.getConstrByName(f"const1_{xi_found}_{k_new}"))
        scen_model.remove(scen_model.getConstrByName(f"const2_{xi_found}_{k_new}"))
        scen_model.remove(scen_model.getConstrByName(f"const3_{xi_found}_{k_new}"))

    scen_model.update()

    return theta_sol, [x_0_sol, x_sol], [y_0_sol, y_sol]


# STATIC SOLUTION ROBUST COUNTERPART
def static_solution_rc(env):
    src = gp.Model("Scenario-Based K-Adaptability Problem")
    projects = env.projects
    N = env.N
    # variables
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
    src.setObjective(theta, GRB.MAXIMIZE)

    # deterministic constraints
    src.addConstrs(x[p] + y[p] <= 1 for p in np.arange(N))

    # uncertain constraints
    # objective constraint
    src.addConstr(gp.quicksum(projects[p].rev_nom*(x[p] + env.kappa*y[p]) for p in np.arange(N)) -
                  gp.quicksum(d_1_a[j] + d_1_b[j] for j in np.arange(env.xi_dim))
                  >= theta + env.lam * (x_0 + env.mu * y_0))
    src.addConstrs((1/2*gp.quicksum(projects[p].rev_nom*projects[p].psi[j]*(x[p] + env.kappa*y[p]) for p in np.arange(N))
                   - d_1_a[j] + d_1_b[j] == 0) for j in np.arange(env.xi_dim))
    # first-stage constraint
    src.addConstr(gp.quicksum(projects[p].cost_nom*x[p] for p in np.arange(N)) +
                  gp.quicksum(d_2_a[j] + d_2_b[j] for j in np.arange(env.xi_dim)) <= env.budget + x_0)
    src.addConstrs((1/2*gp.quicksum(projects[p].cost_nom*projects[p].phi[j]*x[p] for p in np.arange(N))
                   + d_2_a[j] - d_2_b[j] == 0) for j in np.arange(env.xi_dim))
    # second-stage constraints
    src.addConstr(gp.quicksum(projects[p].cost_nom*(x[p] + y[p]) for p in np.arange(N)) +
                  gp.quicksum(d_3_a[j] + d_3_b[j] for j in np.arange(env.xi_dim)) <= env.budget + x_0 + y_0)
    src.addConstrs((1/2*gp.quicksum(projects[p].cost_nom*projects[p].phi[j]*x[p] for p in np.arange(N))
                   + d_3_a[j] - d_3_b[j] == 0) for j in np.arange(env.xi_dim))

    # solve
    src.Params.OutputFlag = 0
    src.optimize()
    x_0_sol = x_0.X
    x_sol = np.array([var.X for i, var in x.items()])

    return [x_0_sol, x_sol]


# SCENARIO FUN STATIC ATTRIBUTES
def scenario_fun_static_build(env, x_input):
    x_0, x = x_input
    N = env.N
    sms = gp.Model("Scenario-Based K-Adaptability Problem")
    # variables
    theta = sms.addVar(lb=-env.lower_bound, ub=0, name="theta")
    y_0 = sms.addVar(lb=0, name="y0")
    y = sms.addVars(N, vtype=GRB.BINARY, name="y")

    # objective function
    sms.setObjective(theta, GRB.MINIMIZE)

    # other constraints
    sms.addConstrs(x[p] + y[p] <= 1 for p in np.arange(N))

    # solve
    sms.Params.OutputFlag = 0
    sms.optimize()

    return sms


def scenario_fun_static_update(env, scen, x_input, sms):
    x_0, x = x_input
    projects = env.projects
    N = env.N

    # old variables
    y = {p: sms.getVarByName(f"y[{p}]") for p in np.arange(N)}
    y_0 = sms.getVarByName("y0")
    theta = sms.getVarByName("theta")

    # constraints
    # objective constraint
    sms.addConstr(-(gp.quicksum(rev_fun(projects[p], scen) * (x[p] + env.kappa*y[p])
                                for p in np.arange(N)) - env.lam*(x_0 + env.mu*y_0)) <= theta, name="new_const_1")
    # budget constraint
    sms.addConstr(gp.quicksum(cost_fun(projects[p], scen) * (x[p]) for p in np.arange(N))
                  <= env.budget + x_0, name="new_const_2")
    sms.addConstr(gp.quicksum(cost_fun(projects[p], scen) * (x[p] + y[p]) for p in np.arange(N))
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


# SCENARIO FUN DETERMINISTIC ATTRIBUTES
def scenario_fun_deterministic_build(env):
    smd = gp.Model("Scenario-Based K-Adaptability Problem")
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
    smd.addConstrs(x[p] + y[p] <= 1 for p in np.arange(N))

    # solve model
    smd.Params.OutputFlag = 0
    smd.optimize()

    return smd


def scenario_fun_deterministic_update(env, scen, smd):
    projects = env.projects
    N = env.N

    x = {p: smd.getVarByName(f"x[{p}]") for p in np.arange(N)}
    x_0 = smd.getVarByName("x0")
    y = {p: smd.getVarByName(f"y[{p}]") for p in np.arange(N)}
    y_0 = smd.getVarByName("y0")
    theta = smd.getVarByName("theta")

    # constraints
    # objective constraint
    smd.addConstr(-(gp.quicksum(rev_fun(projects[p], scen) * (x[p] + env.kappa*y[p])
                                for p in np.arange(N)) - env.lam*(x_0 + env.mu*y_0)) <= theta, name="new_const_1")
    # budget constraint
    smd.addConstr(gp.quicksum(cost_fun(projects[p], scen) * (x[p]) for p in np.arange(N))
                  <= env.budget + x_0, name="new_const_2")
    smd.addConstr(gp.quicksum(cost_fun(projects[p], scen) * (x[p] + y[p]) for p in np.arange(N))
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
