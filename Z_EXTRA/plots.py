import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import pickle as pkl
import os


def plot_cb(K, N, num_inst, K_train=6, sc_pre="dive", sc_min_max=0):
    results = [{}, {}]
    alg_types = ["Random", "ML"]
    num_algs = len(results)
    insts = []
    for i in range(1, num_inst+1):
        try:
            results[0][i] = pkl.load(open(f"cb/data/results/random/final_results_cb_random_N{N}_K{K}_s{i}.pkl", "rb"))
            results[1][i] = pkl.load(open(f"cb/data/results/ml/final_results_cb_ml_ML[N10_K{K_train}_m10_nodes2_ct5_"
                                          f"scp-{sc_pre}_scmm{sc_min_max}]_T[N{N}_K{K}]_LNone_s{i}.pkl", "rb"))

            if results[1][i]["runtime"] > 31*60:
                continue
            insts.append(i)
        except:
            print(i, N, K)
            continue
    # PLOT RESULTS OVER RUNTIME
    t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 31*60+15, 15)])
    num_grids = len(t_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=t_grid, columns=insts, dtype=float))

    for a in np.arange(num_algs):
        for i in insts:
            # random
            theta_final = results[0][i]["theta"]
            t_alg = np.zeros(num_grids)
            for t, theta in results[a][i]["inc_thetas_t"].items():
                t_alg[t_grid > t] = theta/theta_final
            obj[a][i] = t_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(t_grid, random_10, random_90, alpha=0.5)
        plt.plot(t_grid, avg_random, label=alg_types[a])
    plt.xlim([0, 31*60])
    plt.ylim([.8, 1.05])
    plt.xlabel("Runtime (sec)")
    plt.ylabel("Relative Objective")

    os.makedirs("Z_EXTRA/figs", exist_ok=True)
    plt.savefig(f"Z_EXTRA/figs/plot_obj_rt_CB_n{N}_xi4_Ktr{K_train}_{num_inst}_scp-{sc_pre}_scmm{sc_min_max}.png")
    plt.savefig(f"Z_EXTRA/figs/plot_obj_rt_CB_n{N}_xi4_Ktr{K_train}_{num_inst}_scp-{sc_pre}_scmm{sc_min_max}.pdf")
    plt.close()

    # PLOT RESULTS OVER NODES
    n_grid_max = np.max([results[a][i]["tot_nodes"] for a in np.arange(num_algs) for i in insts])
    n_grid = np.arange(0, n_grid_max+10, 10)
    # n_grid = np.arange(0, 1000+10, 10)
    num_grids = len(n_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=n_grid, columns=insts, dtype=float))

    for a in np.arange(num_algs):
        for i in insts:
            # random
            n_alg = np.zeros(num_grids)
            theta_final = results[0][i]["theta"]
            for n, theta in results[a][i]["inc_thetas_n"].items():
                n_alg[n_grid > n] = theta/theta_final
            obj[a][i] = n_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(n_grid, random_10, random_90, alpha=0.5)
        plt.plot(n_grid, avg_random, label=alg_types[a])
    plt.ylim([.8, 1.05])
    plt.xlabel("Nodes")
    plt.ylabel("Relative Objective")
    plt.savefig(f"Z_EXTRA/figs/plot_obj_nodes_CB_n{N}_xi4_Ktr{K_train}_{num_inst}_scp-{sc_pre}_scmm{sc_min_max}.png")
    plt.savefig(f"Z_EXTRA/figs/plot_obj_nodes_CB_n{N}_xi4_Ktr{K_train}_{num_inst}_scp-{sc_pre}_scmm{sc_min_max}.pdf")

    plt.close()

    return obj


def plot_cb_heur(K, N, num_inst, K_train=6):
    results = [{}, {}]
    alg_types = ["Random", "ML"]
    num_algs = len(results)
    insts = []
    for i in range(1, num_inst+1):
        try:
            results[0][i] = pkl.load(open(f"cb/data/results/random/final_results_cb_random_N{N}_K{K}_s{i}.pkl", "rb"))
            results[1][i] = pkl.load(open(f"cb/data/results/heuristic/final_results_cb_heuristic_N{N}_K{K}_s{i}.pkl", "rb"))

            if results[1][i]["runtime"] > 31*60:
                continue
            insts.append(i)
        except:
            print(i, N, K)
            continue
    # PLOT RESULTS OVER RUNTIME
    t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 31*60+15, 15)])
    num_grids = len(t_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=t_grid, columns=insts, dtype=float))

    for a in np.arange(num_algs):
        for i in insts:
            # random
            theta_final = results[0][i]["theta"]
            t_alg = np.zeros(num_grids)
            for t, theta in results[a][i]["inc_thetas_t"].items():
                t_alg[t_grid > t] = theta/theta_final
            obj[a][i] = t_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    cols = ["black", "darkgreen"]
    labels = ["K-B&B", "K-B&B-Heuristic"]
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(t_grid, random_10, random_90, alpha=0.5, color=cols[a])
        plt.plot(t_grid, avg_random, label=labels[a], color=cols[a])
    plt.xlim([0, 31*60])
    plt.ylim([.8, 1.05])
    plt.xlabel("Runtime (sec)")
    plt.ylabel("Relative OFV")
    plt.legend()

    os.makedirs("Z_EXTRA/figs", exist_ok=True)
    plt.savefig(f"Z_EXTRA/figs/plot_obj_rt_CB_n{N}_xi4_Ktr{K_train}_{num_inst}_heur.png")
    plt.savefig(f"Z_EXTRA/figs/plot_obj_rt_CB_n{N}_xi4_Ktr{K_train}_{num_inst}_heur.pdf")
    plt.close()

    # PLOT RESULTS OVER NODES
    n_grid_max = np.max([results[a][i]["tot_nodes"] for a in np.arange(num_algs) for i in insts])
    n_grid = np.arange(0, n_grid_max+10, 10)
    # n_grid = np.arange(0, 1000+10, 10)
    num_grids = len(n_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=n_grid, columns=insts, dtype=float))

    for a in np.arange(num_algs):
        for i in insts:
            # random
            n_alg = np.zeros(num_grids)
            theta_final = results[0][i]["theta"]
            for n, theta in results[a][i]["inc_thetas_n"].items():
                n_alg[n_grid > n] = theta/theta_final
            obj[a][i] = n_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(n_grid, random_10, random_90, alpha=0.5)
        plt.plot(n_grid, avg_random, label=alg_types[a])
    plt.ylim([.8, 1.05])
    plt.xlabel("Nodes")
    plt.ylabel("Relative Objective")
    plt.savefig(f"Z_EXTRA/figs/plot_obj_nodes_CB_n{N}_xi4_Ktr{K_train}_{num_inst}_heur.png")
    plt.savefig(f"Z_EXTRA/figs/plot_obj_nodes_CB_n{N}_xi4_Ktr{K_train}_{num_inst}_heur.pdf")

    plt.close()

    return obj



def plot_sp_sphere(K, N, num_inst, pre_sc_data, sc_min_max):
    results = [{}, {}]
    alg_types = ["Random", "ML"]
    num_algs = len(results)
    insts = []
    for i in range(1, num_inst+1):
        try:
            with open(f"data/sp/results/random/res_SP_sphere_n{N}_s{i}_K{K}.pkl", "rb") as handle:
                results[0][i] = pkl.load(handle)

            with open(f"data/sp/results/ml/res_SP_sphere_n{N}_s{i}_prscd-{pre_sc_data}_scmm-{sc_min_max}_K{K}.pkl", "rb") as handle:
                results[1][i] = pkl.load(handle)
            if results[1][i]["runtime"] > 32*60:
                continue
            if len(results[1][i]["inc_obj_t"]) <= 2:
                continue
            insts.append(i)
        except:
            print(i, N, K, pre_sc_data, sc_min_max)
            continue
    # PLOT RESULTS OVER RUNTIME
    t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 60*60+15, 15)])
    num_grids = len(t_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=t_grid, columns=insts, dtype=float))

    for a in np.arange(num_algs):
        for i in insts:
            # random
            theta_final = results[0][i]["obj"]
            t_alg = np.zeros(num_grids)
            for t, theta in results[a][i]["inc_obj_t"].items():
                t_alg[t_grid > t] = theta/theta_final
            obj[a][i] = t_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(t_grid, random_10, random_90, alpha=0.5)
        plt.plot(t_grid, avg_random, label=alg_types[a])
    plt.xlim([0, 31*60])
    plt.ylim(0.98, 1.04)
    plt.xlabel("Runtime (sec)")
    plt.ylabel("Relative Objective")
    plt.legend(loc=1)

    os.makedirs("Z_EXTRA/figs", exist_ok=True)
    plt.savefig(f"Z_EXTRA/figs/plot_obj_rt_SP_sphere_n{N}_xi4_prscd-{pre_sc_data}_scmm-{sc_min_max}_K{K}_{num_inst}.png")
    plt.savefig(f"Z_EXTRA/figs/plot_obj_rt_SP_sphere_n{N}_xi4_prscd-{pre_sc_data}_scmm-{sc_min_max}_K{K}_{num_inst}.pdf")
    plt.close()

    # PLOT RESULTS OVER NODES
    n_grid_max = np.max([results[a][i]["tot_nodes"] for a in np.arange(num_algs) for i in insts])
    n_grid = np.arange(0, n_grid_max+10, 10)
    # n_grid = np.arange(0, 1000+10, 10)
    num_grids = len(n_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=n_grid, columns=insts, dtype=float))

    for a in np.arange(num_algs):
        for i in insts:
            # random
            n_alg = np.zeros(num_grids)
            theta_final = results[0][i]["obj"]
            for n, theta in results[a][i]["inc_obj_n"].items():
                n_alg[n_grid > n] = theta/theta_final
            obj[a][i] = n_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(n_grid, random_10, random_90, alpha=0.5)
        plt.plot(n_grid, avg_random, label=alg_types[a])

    plt.xlabel("Nodes")
    plt.ylabel("Relative Objective")
    plt.ylim(0.98, 1.04)
    plt.legend(loc=1)
    plt.savefig(f"Z_EXTRA/figs/plot_obj_nodes_SP_sphere_n{N}_xi4_prscd-{pre_sc_data}_scmm-{sc_min_max}_K{K}_{num_inst}")
    plt.savefig(f"Z_EXTRA/figs/plot_obj_nodes_SP_sphere_n{N}_xi4_prscd-{pre_sc_data}_scmm-{sc_min_max}_K{K}_{num_inst}")
    plt.close()

    return obj


if __name__ == "__main__":
    # for scp in ["dive", "alt", "no"]:
    #     for scmm in [0, 1]:
            # plot_cb(6, 10, 8, sc_pre=scp, sc_min_max=scmm)
    plot_cb_heur(6, 10, 16)