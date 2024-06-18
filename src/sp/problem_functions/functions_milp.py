import numpy as np
import gurobipy as gp
from gurobipy import GRB

"""
Code for solving the MILP formulation of the shortest path problem 
of the master problem (given as scenario problem) and subproblem (scenario problem)
"""


# SCENARIO FUN = MASTER PROBLEM
# update master problem with new scenario in specified subset (k_new)
def scenario_fun_update(K, k_new, xi_new, graph, scen_model=None):
    # use same model and just add new constraint
    y = dict()
    for k in range(K):
        y[k] = {a: scen_model.getVarByName("y_{}[{}]".format(k, a)) for a in range(graph.num_arcs)}
    theta = scen_model.getVarByName("theta")

    scen_model.addConstr(gp.quicksum((1 + xi_new[a] / 2) * graph.distances_array[a] * y[k_new][a]
                                     for a in range(graph.num_arcs)) <= theta)
    scen_model.update()

    # solve model
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    y_sol = dict()
    for k in range(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = scen_model.getVarByName("theta").X

    return theta_sol, None, y_sol, scen_model


# build master problem
def scenario_fun_build(K, tau, graph, gp_env):
    scen_model = gp.Model("Scenario-Based K-Adaptability Problem", gp_env)
    N = graph.N
    # variables
    theta = scen_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
    y = dict()
    for k in range(K):
        y[k] = scen_model.addVars(graph.num_arcs, vtype=GRB.BINARY, name="y_{}".format(k))
    # objective function
    scen_model.setObjective(theta, GRB.MINIMIZE)

    # deterministic constraints
    for k in range(K):
        for j in range(graph.N):
            if j == graph.s:
                scen_model.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_out[j]) >= 1)
                scen_model.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_in[j]) == 0)
                continue
            if j == graph.t:
                scen_model.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_in[j]) >= 1)
                scen_model.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_out[j]) == 0)
                continue
            scen_model.addConstr(
                gp.quicksum(y[k][a] for a in graph.arcs_out[j])
                - gp.quicksum(y[k][a] for a in graph.arcs_in[j]) == 0)

    for k in range(K):
        for xi in tau[k]:
            scen_model.addConstr(gp.quicksum((1 + xi[a] / 2) * graph.distances_array[a] * y[k][a]
                                             for a in range(graph.num_arcs)) <= theta)
    scen_model.update()

    # solve model
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    y_sol = dict()
    for k in range(K):
        # y_sol[k] = {i: var.X for i, var in y[k].items()}
        y_sol[k] = np.array([var.X for i, var in y[k].items()])
    theta_sol = scen_model.getVarByName("theta").X

    return theta_sol, None, y_sol, scen_model


# SEPARATION FUN = SUBPROBLEM
def separation_fun(K, x, y, theta, graph, tau, gp_env):
    sep_model = gp.Model("Separation Problem", env=gp_env)
    sep_model.Params.OutputFlag = 0
    # variables
    zeta = sep_model.addVar(lb=-graph.bigM, name="zeta", vtype=GRB.CONTINUOUS)
    xi = sep_model.addVars(graph.num_arcs, lb=0, ub=1, name="xi", vtype=GRB.CONTINUOUS)

    # objective function
    sep_model.setObjective(zeta, GRB.MAXIMIZE)

    # uncertainty set
    sep_model.addConstr(gp.quicksum(xi[a] for a in range(graph.num_arcs)) <= graph.gamma)

    for k in range(K):
        if len(tau[k]) > 0:
            sep_model.addConstr(zeta <= gp.quicksum((1 + xi[a] / 2) * graph.distances_array[a] * y[k][a]
                                                    for a in range(graph.num_arcs)) - theta)

    # solve
    sep_model.optimize()
    zeta_sol = zeta.X
    xi_sol = np.array([var.X for i, var in xi.items()])
    return zeta_sol, xi_sol


# SUB TREE MODEL (for training data generation)
def scenario_fun_update_sub_tree(K, new_node, xi_dict, graph, scen_model=None):
    # use same model and just add new constraint
    y = dict()
    for k in range(K):
        y[k] = {a: scen_model.getVarByName("y_{}[{}]".format(k, a)) for a in range(graph.num_arcs)}
    theta = scen_model.getVarByName("theta")

    for node_sec in range(1, len(new_node)):
        xi_new = xi_dict[new_node[:node_sec]]
        k_new = new_node[node_sec]
        scen_model.addConstr(gp.quicksum((1 + xi_new[a] / 2) * graph.distances_array[a] * y[k_new][a]
                                         for a in range(graph.num_arcs)) <= theta, name=f"const_{new_node[:node_sec]}_{k_new}")
    scen_model.update()

    # solve model
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    y_sol = dict()
    for k in range(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = scen_model.getVarByName("theta").X

    # delete constraints
    for node_sec in range(1, len(new_node)):
        xi_found = new_node[:node_sec]
        k_new = new_node[node_sec]
        scen_model.remove(scen_model.getConstrByName(f"const_{xi_found}_{k_new}"))
    scen_model.update()

    return theta_sol, None, y_sol


# SCENARIO FUN DETERMINISTIC ATTRIBUTES (for features)
# build problem
def scenario_fun_deterministic_build(graph, gp_env):
    smn = gp.Model("Scenario-Based K-Adaptability Problem", env=gp_env)
    N = graph.N
    # variables
    theta = smn.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
    y = smn.addVars(graph.num_arcs, vtype=GRB.BINARY, name="y")
    # objective function
    smn.setObjective(theta, GRB.MINIMIZE)

    # deterministic constraints
    for j in range(graph.N):
        if j == graph.s:
            smn.addConstr(gp.quicksum(y[a] for a in graph.arcs_out[j]) >= 1)
            smn.addConstr(gp.quicksum(y[a] for a in graph.arcs_in[j]) == 0)
            continue
        if j == graph.t:
            smn.addConstr(gp.quicksum(y[a] for a in graph.arcs_in[j]) >= 1)
            smn.addConstr(gp.quicksum(y[a] for a in graph.arcs_out[j]) == 0)
            continue
        smn.addConstr(
            gp.quicksum(y[a] for a in graph.arcs_out[j])
            - gp.quicksum(y[a] for a in graph.arcs_in[j]) == 0)

    # solve model
    smn.Params.OutputFlag = 0
    smn.optimize()
    return smn


# update problem with new scenario
def scenario_fun_deterministic_update(graph, scen, smn):
    y = {a: smn.getVarByName(f"y[{a}]") for a in range(graph.num_arcs)}
    theta = smn.getVarByName("theta")

    # constraint
    smn.addConstr(gp.quicksum((1 + scen[a] / 2) * graph.distances_array[a] * y[a]
                              for a in range(graph.num_arcs)) <= theta, name="new_const")
    smn.update()
    # solve model
    smn.optimize()
    y_sol = [var.X for i, var in y.items()]
    theta_sol = smn.getVarByName("theta").X

    # delete new constraint
    smn.remove(smn.getConstrByName("new_const"))
    smn.update()

    return theta_sol, y_sol