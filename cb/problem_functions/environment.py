import numpy as np
import os

"""
Code for making a capital budgeting instance class
"""


# class per project where can be invested in
class Project:
    def __init__(self, cost_nom, phi, psi, xi_dim, rev_nom=None):
        self.cost_nom = cost_nom
        if rev_nom is None:
            self.rev_nom = self.cost_nom/5
        else:
            self.rev_nom = rev_nom
        self.phi = phi
        self.psi = psi
        self.xi_dim = xi_dim


class ProjectsInstance:
    def __init__(self, problem, N, inst_num=0, xi_dim=4):
        self.problem = problem
        self.N = N
        self.kappa = 0.8
        self.lam = 0.12
        self.mu = 1.2
        self.xi_dim = xi_dim
        self.x_dim = self.N + 1
        self.y_dim = self.N + 1
        self.inst_num = inst_num
        self.init_uncertainty = np.zeros(self.xi_dim)

    def make_test_inst(self, save_env=True):
        rng = np.random.RandomState(self.inst_num)
        # make cost vector for projects
        cost_nom_vector = rng.uniform(0, 10, self.N)
        self.budget = sum(cost_nom_vector) / 2
        self.max_loan = sum(cost_nom_vector)*1.5 - self.budget

        # define phi and psi, with unit simplex randomness
        phi_vector = dict()
        psi_vector = dict()

        for p in range(self.N):
            x = {0: 0}
            y = {0: 0}
            for i in range(1, self.xi_dim):
                x[i] = rng.uniform(0, 1)
                y[i] = rng.uniform(0, 1)
            x[self.xi_dim] = 1
            y[self.xi_dim] = 1
            x_values = sorted(x.values())
            y_values = sorted(y.values())
            x = dict()
            for i in range(len(x_values)):
                x[i] = x_values[i]
                y[i] = y_values[i]
            phi = dict()
            psi = dict()
            for i in range(1, self.xi_dim+1):
                phi[i-1] = x[i] - x[i-1]
                psi[i-1] = y[i] - y[i-1]
            phi_vector[p] = list(phi.values())
            psi_vector[p] = list(psi.values())

        # define the projects
        projects = dict()
        for p in range(self.N):
            projects[p] = Project(cost_nom_vector[p], np.array(phi_vector[p]), np.array(psi_vector[p]), self.xi_dim)
        self.projects = projects

        self.upper_bound = 0
        self.bigM = sum([(1+1/2)*self.projects[i].cost_nom for i in range(self.N)])
        self.lower_bound = sum([(1+1/2)*self.projects[i].rev_nom for i in range(self.N)])
        if save_env:
            self.write_test_inst()

    def read_test_inst(self):
        inst_path = f"cb/data/instances/cb_env_N{self.N}_s{self.inst_num}.txt"

        with open(inst_path, 'r') as f:
            f_lines = f.readlines()

        inst_info = list(map(lambda x: float(x), f_lines[0].replace('\n', '').split(' ')))
        self.inst_num = int(inst_info[0])
        self.N = int(inst_info[1])
        self.xi_dim = int(inst_info[2])
        self.budget = int(inst_info[3])
        self.max_loan = int(inst_info[4])

        # make projects
        cost_nom_vector = np.array(list(map(lambda x: float(x), f_lines[1].replace('\n', '').split(' '))))
        rev_nom_vector = np.array(list(map(lambda x: float(x), f_lines[2].replace('\n', '').split(' '))))

        phi = dict()
        for i in range(self.N):
            phi[i] = np.array(list(map(lambda x: float(x), f_lines[3 + i].replace('\n', '').split(' '))))

        psi = dict()
        for i in range(self.N):
            psi[i] = np.array(
                list(map(lambda x: float(x), f_lines[3 + self.N + i].replace('\n', '').split(' '))))

        projects = dict()
        for p in range(self.N):
            projects[p] = Project(cost_nom_vector[p], phi[p], psi[p], self.xi_dim, rev_nom_vector[p])
        self.projects = projects
        self.upper_bound = 0
        self.bigM = sum([(1+1/2)*self.projects[i].cost_nom for i in range(self.N)])
        self.lower_bound = sum([(1+1/2)*self.projects[i].rev_nom for i in range(self.N)])

    def write_test_inst(self):
        # different:
        os.makedirs("cb/data/instances", exist_ok=True)
        test_dir = f"cb/data/instances/cb_env_N{self.N}_s{self.inst_num}.txt"
        f = open(test_dir, "w+")
        f.write(str(self.inst_num) + " ")
        f.write(str(self.N) + " ")
        f.write(str(self.xi_dim) + " ")
        f.write(str(self.budget) + " ")
        f.write(str(self.max_loan) + "\n")
        c_bar = np.array([p.cost_nom for p in self.projects.values()])
        f.write(" ".join(c_bar.astype(str)) + "\n")
        r_bar = np.array([p.rev_nom for p in self.projects.values()])
        f.write(" ".join(r_bar.astype(str)) + "\n")
        for p in self.projects.values():
            f.write(" ".join(p.phi.astype(str)) + "\n")
        for p in self.projects.values():
            f.write(" ".join(p.psi.astype(str)) + "\n")
        f.close()


def cost_fun(project, xi):
    return (1 + sum(project.phi[i]*xi[i] for i in range(project.xi_dim))/2)*project.cost_nom


def rev_fun(project, xi):
    return (1 + sum(project.psi[i]*xi[i] for i in range(project.xi_dim))/2)*project.rev_nom




# def make_env(X_param, inst_list=None):
#     N = 10
#     xi_dim = 2
#     env = dict()
#     if inst_list is None:
#         inst_list = X_param.index
#     for i in inst_list:
#         try:
#             cost_array = np.array([X_param.loc[i, "c_{}".format(p)] for p in range(N)])
#         except KeyError:
#             continue
#         try:
#             phi_array = np.array([X_param.loc[i, "phi_{}_{}".format(p, xi)] for p in range(N) for xi in range(xi_dim)]).reshape([N, xi_dim])
#             psi_array = np.array([X_param.loc[i, "psi_{}_{}".format(p, xi)] for p in range(N) for xi in range(xi_dim)]).reshape([N, xi_dim])
#         except ValueError:
#             print(i)
#         env[i] = ProjectsInstance(N, xi_dim, init_cost_vector=cost_array, init_phi_vector=phi_array, init_psi_vector=psi_array)
#     return env
