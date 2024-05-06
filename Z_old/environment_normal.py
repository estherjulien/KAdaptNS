import numpy as np
import os

"""
Code for making a 2-D graph as a shortest path instance
"""


class Graph:
    def __init__(self, N, inst_num=0, gamma=7, degree=5, throw_away_perc=0.9):  # TODO
        self.N = N
        self.degree = degree
        self.throw_away_perc = throw_away_perc
        self.gamma = gamma
        self.inst_num = inst_num

    def make_test_inst(self, save_env=True):
        self.vertices, init_arcs = self.init_graph()
        self.distances, self.s, self.t = self.update_graph(init_arcs, self.throw_away_perc)

        self.arcs = self.distances > 1e-5
        self.num_arcs = int(self.arcs.sum())
        self.xi_dim = self.num_arcs
        self.x_dim = self.num_arcs
        self.y_dim = self.num_arcs

        self.arcs_array = np.array([[i, j] for i in range(self.N) for j in range(self.N) if self.arcs[i, j]])
        self.distances_array = np.array([self.distances[i, j] for i, j in self.arcs_array])
        self.arcs_in, self.arcs_out = self.in_out_arcs()

        self.bigM = sum(self.distances_array)*3
        self.upper_bound = sum(self.distances_array)*3
        self.init_uncertainty = np.zeros(self.num_arcs)

        if save_env:
            self.write_test_inst()

    def read_test_instance(self):
        """ Load instances data. """
        inst_path = f"cb/data/instances/cb_env_N{self.N}_s{self.inst_num}.txt"

        with open(inst_path, 'r') as f:
            f_lines = f.readlines()

        inst_info = list(map(lambda x: float(x), f_lines[0].replace('\n', '').split(' ')))
        self.inst_num = int(inst_info[0])
        self.N = int(inst_info[1])
        self.num_arcs = int(inst_info[2])
        self.s = int(inst_info[3])
        self.t = int(inst_info[4])

        # store inst data
        self.distances_array = np.array(list(map(lambda x: float(x), f_lines[1].replace('\n', '').split(' '))))
        arcs_in = dict()
        for i in range(self.N):
            arcs_in[i] = np.array(list(map(lambda x: float(x), f_lines[2 + i].replace('\n', '').split(' '))))
        self.arcs_in = arcs_in

        arcs_out = dict()
        for i in range(self.N):
            arcs_out[i] = np.array(
                list(map(lambda x: float(x), f_lines[2 + self.N + i].replace('\n', '').split(' '))))
        self.arcs_out = arcs_out

    def write_test_inst(self):
        # different:
        os.makedirs("sp/data/instances", exist_ok=True)
        test_dir = f"sp/data/instances/sp_normal_env_N{self.N}_s{self.inst_num}.txt"
        f = open(test_dir, "w+")
        f.write(str(self.inst_num) + " ")
        f.write(str(self.N) + " ")
        f.write(str(self.num_arcs) + " ")
        f.write(str(self.s) + " ")
        f.write(str(self.t) + "\n")
        f.write(" ".join(self.distances_array.astype(str)) + "\n")
        for arcs_in in self.arcs_in.values():
            f.write(" ".join(np.array(arcs_in).astype(str)) + "\n")
        for arcs_out in self.arcs_out.values():
            f.write(" ".join(np.array(arcs_out).astype(str)) + "\n")
        for verts in self.vertices:
            f.write(" ".join(verts.astype(str)) + "\n")
        f.close()

    def set_gamma(self, gamma):
        self.gamma = gamma

    def vertices_fun(self):
        rng = np.random.RandomState(self.inst_num)
        vertices_set = np.zeros([self.N, 2], dtype=np.float)
        # start and terminal node are the first and last one, on 0,0 and 10,10, respectively
        for n in range(self.N):
            x, y = rng.uniform(0, 10, 2)
            vertices_set[n] = [x, y]
        return vertices_set

    def init_graph(self):
        vertices = self.vertices_fun()
        N = self.N
        # make initial arcs
        arcs = np.ones([N, N], dtype=np.float)
        for i in range(N):
            for j in range(N):
                if i == j:
                    arcs[i, j] = 0

        return vertices, arcs

    def update_graph(self, arcs, throw_away_perc):
        arc_dict = dict()
        for i in range(self.N):
            for j in range(self.N):
                if arcs[i, j] < 1e-5:
                    continue
                x_i, y_i = self.vertices[i]
                x_j, y_j = self.vertices[j]
                distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                arcs[i, j] = distance
                arc_dict[(i, j)] = distance
        # delete long arcs (first sort and then sth percent)
        arc_dict_order = {k: v for k, v in sorted(arc_dict.items(), key=lambda item: -item[1])}
        arc_list_order = list(arc_dict_order)
        # first ones are s and t!
        s, t = arc_list_order[0]
        # delete "throw_away_perc" longest arcs
        throw_away_num = np.floor(len(arc_dict_order)*throw_away_perc)
        in_degrees = {i: sum([arcs[:, i] > 1e-5][0]) for i in range(self.N)}
        out_degrees = {i: sum([arcs[i, :] > 1e-5][0]) for i in range(self.N)}
        del_arc = 0
        while del_arc < throw_away_num:
            # check here if when you delete this, degree out and in will be >= 1 and total degree >= min_degree
            try:
                i, j = arc_list_order[del_arc]
            except KeyError:
                break
            # check in degree of j
            if in_degrees[j] <= self.degree:
                del_arc += 1
                continue
            # check out degree of i
            if out_degrees[i] <= self.degree:
                del_arc += 1
                continue
            # check each time if you can delete this arc based on connected graph
            arcs[i, j] = 0
            in_degrees[j] -= 1
            out_degrees[i] -= 1
            del_arc += 1
        return arcs, s, t

    def in_out_arcs(self):
        arcs_in = {i: [] for i in range(self.N)}
        arcs_out = {i: [] for i in range(self.N)}
        for a in range(self.num_arcs):
            i, j = self.arcs_array[a]
            arcs_in[j].append(a)
            arcs_out[i].append(a)
        return arcs_in, arcs_out
