import numpy as np
import copy


class KnapsackEnv:
    def __init__(self, N, gamma_perc=None, budget_perc=None, inst_num=0):
        self.N = N
        self.xi_dim = self.N
        self.y_dim = self.N
        self.inst_num = inst_num
        self.init_uncertainty = np.zeros(self.xi_dim)

        self.weight = np.random.uniform(1, 15, self.N)
        self.cost = np.random.uniform(100, 150, self.N)

        # FROM PAPER
        # self.weight = np.random.uniform(100, 1500, self.N)
        # self.cost = np.random.uniform(10000, 15000, self.N)
        # self.budget = 100*self.N

        if budget_perc is not None:
            self.budget_perc = budget_perc
            self.budget = np.floor(self.budget_perc * np.sum(self.weight))

        if gamma_perc is not None:
            self.gamma_perc = gamma_perc
            self.gamma = np.floor(self.gamma_perc*self.N)

        self.upper_bound = 0
        # for sub problem
        self.bigM = sum(self.cost)*1.1
        # for master problem
        self.lower_bound = - self.bigM

    def set_budget(self, budget_perc):
        self.budget_perc = budget_perc
        self.budget = np.floor(self.budget_perc * np.sum(self.weight))

    def set_gamma(self, gamma_perc):
        self.gamma_perc = gamma_perc
        self.gamma = np.floor(self.gamma_perc * self.N)
