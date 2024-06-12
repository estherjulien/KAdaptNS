import numpy as np
import copy
import os

"""
Code for making a knapsack instance class
"""


# class per project where can be invested in
class KnapsackEnv:
    def __init__(self, problem, N, gamma_perc=None, budget_perc=None, inst_num=0):
        self.problem = problem
        self.N = N
        self.xi_dim = self.N
        self.y_dim = self.N
        self.inst_num = inst_num
        self.init_uncertainty = np.zeros(self.xi_dim)

        if budget_perc is not None:
            self.budget_perc = float(budget_perc)/100
        if gamma_perc is not None:
            self.gamma_perc = float(gamma_perc)/100

    def make_test_inst(self, save_env=True):
        rng = np.random.RandomState(self.inst_num)
        self.weight = rng.uniform(1, 15, self.N)
        self.cost = rng.uniform(100, 150, self.N)

        self.upper_bound = 0
        # for sub problem
        self.bigM = sum(self.cost)*1.1
        # for master problem
        self.lower_bound = - self.bigM

        for budget_perc in [5, 15, 35, 50]:
            for gamma_perc in [5, 15, 35, 50]:
                env = copy.deepcopy(self)
                env.set_gamma(gamma_perc=float(gamma_perc)/100)
                env.set_budget(budget_perc=float(budget_perc)/100)
                if save_env:
                    env.write_test_inst()

    def read_test_inst(self):
        inst_path = f"src/kp/data/instances/kp_env_N{self.N}_" \
                    f"g{int(self.gamma_perc*100)}_b{int(self.budget_perc*100)}" \
                    f"_s{self.inst_num}.txt"

        with open(inst_path, 'r') as f:
            f_lines = f.readlines()

        inst_info = list(map(lambda x: float(x), f_lines[0].replace('\n', '').split(' ')))
        self.inst_num = int(inst_info[0])
        self.N = int(inst_info[1])
        self.gamma_perc = float(inst_info[2])
        self.budget_perc = float(inst_info[3])
        self.weight = np.array(list(map(lambda x: float(x), f_lines[1].replace('\n', '').split(' '))))
        self.cost = np.array(list(map(lambda x: float(x), f_lines[2].replace('\n', '').split(' '))))

        self.budget = np.floor(self.budget_perc * np.sum(self.weight))
        self.gamma = np.floor(self.gamma_perc * self.N)

        self.upper_bound = 0
        # for sub problem
        self.bigM = sum(self.cost)*1.1
        # for master problem
        self.lower_bound = - self.bigM

    def write_test_inst(self):
        # different:
        os.makedirs("src/kp/data/instances", exist_ok=True)
        inst_path = f"src/kp/data/instances/kp_env_N{self.N}_" \
                    f"g{int(self.gamma_perc*100)}_b{int(self.budget_perc*100)}" \
                    f"_s{self.inst_num}.txt"
        f = open(inst_path, "w+")
        f.write(str(self.inst_num) + " ")
        f.write(str(self.N) + " ")
        f.write(str(self.gamma_perc) + " ")
        f.write(str(self.budget_perc) + "\n")
        f.write(" ".join(self.weight.astype(str)) + "\n")
        f.write(" ".join(self.cost.astype(str)) + "\n")
        f.close()

    def set_budget(self, budget_perc):
        self.budget_perc = budget_perc
        self.budget = np.floor(self.budget_perc * np.sum(self.weight))

    def set_gamma(self, gamma_perc):
        self.gamma_perc = gamma_perc
        self.gamma = np.floor(self.gamma_perc * self.N)
