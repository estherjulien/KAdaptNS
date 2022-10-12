import numpy as np

"""
Code for making a capital budgeting instance class
"""


# class per project where can be invested in
class Project:
    def __init__(self, cost_nom, phi, psi, xi_dim):
        self.cost_nom = cost_nom
        self.rev_nom = self.cost_nom/5
        self.phi = phi
        self.psi = psi
        self.xi_dim = xi_dim


class ProjectsInstance:
    def __init__(self, N, xi_dim=4, init_cost_vector=None, init_phi_vector=None, init_psi_vector=None, inst_num=0):
        self.N = N
        self.kappa = 0.8
        self.lam = 0.12
        self.mu = 1.2
        self.xi_dim = xi_dim
        self.x_dim = self.N + 1
        self.y_dim = self.N + 1
        self.inst_num = inst_num
        self.init_uncertainty = np.zeros(self.xi_dim)

        # make cost vector for projects
        if init_cost_vector is None:
            cost_nom_vector = np.random.uniform(0, 10, N)
        else:
            cost_nom_vector = init_cost_vector
        self.budget = sum(cost_nom_vector) / 2
        self.max_loan = sum(cost_nom_vector)*1.5 - self.budget

        if init_phi_vector is not None and init_psi_vector is not None:
            phi_vector = init_phi_vector
            psi_vector = init_psi_vector
        else:
            # define phi and psi, with unit simplex randomness
            phi_vector = dict()
            psi_vector = dict()

            for p in np.arange(N):
                x = {0: 0}
                y = {0: 0}
                for i in range(1, xi_dim):
                    x[i] = np.random.uniform(0, 1)
                    y[i] = np.random.uniform(0, 1)
                x[xi_dim] = 1
                y[xi_dim] = 1
                x_values = sorted(x.values())
                y_values = sorted(y.values())
                x = dict()
                for i in np.arange(len(x_values)):
                    x[i] = x_values[i]
                    y[i] = y_values[i]
                phi = dict()
                psi = dict()
                for i in range(1, xi_dim+1):
                    phi[i-1] = x[i] - x[i-1]
                    psi[i-1] = y[i] - y[i-1]
                phi_vector[p] = phi
                psi_vector[p] = psi

        # define the projects
        projects = dict()
        for p in np.arange(N):
            projects[p] = Project(cost_nom_vector[p], phi_vector[p], psi_vector[p], self.xi_dim)
        self.projects = projects

        # other stuff
        self.upper_bound = 0
        self.bigM = sum([(1+1/2)*self.projects[i].cost_nom for i in np.arange(self.N)])
        self.lower_bound = sum([(1+1/2)*self.projects[i].rev_nom for i in np.arange(N)])


def cost_fun(project, xi):
    return (1 + sum(project.phi[i]*xi[i] for i in np.arange(project.xi_dim))/2)*project.cost_nom


def rev_fun(project, xi):
    return (1 + sum(project.psi[i]*xi[i] for i in np.arange(project.xi_dim))/2)*project.rev_nom


def make_env(X_param, inst_list=None):
    N = 10
    xi_dim = 2
    env = dict()
    if inst_list is None:
        inst_list = X_param.index
    for i in inst_list:
        try:
            cost_array = np.array([X_param.loc[i, "c_{}".format(p)] for p in np.arange(N)])
        except KeyError:
            continue
        try:
            phi_array = np.array([X_param.loc[i, "phi_{}_{}".format(p, xi)] for p in np.arange(N) for xi in np.arange(xi_dim)]).reshape([N, xi_dim])
            psi_array = np.array([X_param.loc[i, "psi_{}_{}".format(p, xi)] for p in np.arange(N) for xi in np.arange(xi_dim)]).reshape([N, xi_dim])
        except ValueError:
            print(i)
        env[i] = ProjectsInstance(N, xi_dim, init_cost_vector=cost_array, init_phi_vector=phi_array, init_psi_vector=psi_array)
    return env
