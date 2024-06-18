## Experimental results data
Welcome to the data directory accompanying the results of our paper titled **"Machine Learning for _K_-adaptability in
Two-stage Robust Optimization"**. This repository contains all the datasets referenced in the publication. 

### Structure of the Repository
The directory contains the raw datasets of the results of the **Experiments** section. Each dataset is provided in the standard format CSV. The results are grouped per benchmark: capital budgeting (**cb/**), knapsack problem (**kp**), shortest path (**sp_normal/**), and shortest path on a sphere (**sp_sphere/**).

Per benchmark, we have the following files (if applicable).
- `<benchmark>_results_ns-random.csv`: includes the results of `K-B&B` (i.e., random node selection). 
- `<benchmark>_EXP1-4_ns-ml.csv`: includes the results of `K-B&B-NodeSelection` (i.e., ml node selection) for EXP 1 to 4.
- `<benchmark>_EXP5-6_ns-ml_mlmodel-<other benchmark>.csv`: includes the results of `K-B&B-NodeSelection` (i.e., ml node selection) for EXP 5 to 6, where the ml model is trained on data of an alternative benchmark.
- `<benchmark>_tuning_Ltest_epsilon_ns-ml.csv`: includes the results of `K-B&B-NodeSelection` for different values of Ltest and epsilon.
- `<benchmark>_tuning_T_ns-ml.csv`: includes the results of `K-B&B-NodeSelection` for different values of number of hours spent on generating training data (T).

Each dataset has the following labels (if applicable): 
- `inst`: instance number.
- `N`: instance size.
- `K`: number of second-stage decisions, parameter of the _K_-adaptability algorithm.
- `K_train`: K used for generating training data.
- `K_test`: K used for solving benchmark instance.
- `L_test`: layer in branch-and-bound tree up to apply ml instructed node selection.
- `epsilon`: classification threshold.
- `T`: number of hours spent on generating training data for training the ML model used.
- `final_time`: runtime of algorithm (in seconds). 
- `final_obj`: objective function value obtained at termination of algorithm. 
- `time_<i>`: runtime for `i`-th incumbent solution is found.
- `obj_<i>`: objective function value for `i`-th incumbent solution is found.