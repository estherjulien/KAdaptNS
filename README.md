[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Machine Learning for K-adaptability in Two-stage Robust Optimization

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[Machine Learning for K-adaptability in Two-stage Robust Optimization](https://doi.org/10.1287/ijoc.2022.0314) by E. Julien, K. Postek, and S. I. Birbil. 

**Important: This code is being developed on an on-going basis at 
https://github.com/estherjulien/KAdaptNS. Please go there if you would like to
get a more recent version or would like support**

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2022.0314

https://doi.org/10.1287/ijoc.2022.0314.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{julien2024machine,
  author =        {Julien, Esther and Postek, Krzysztof and Birbil, S. Ilker},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Machine Learning for K-adaptability in Two-stage Robust Optimization}},
  year =          {2024},
  doi =           {10.1287/ijoc.2022.0314.cd},
  url =           {https://github.com/INFORMSJoC/2022.0314},
  note =          {Available for download at https://github.com/INFORMSJoC/2022.0314},
}  
```

## Description

This directory contains the code for the _K_-adaptability branch-and-bound (_K_-B&B) algorithm, with random node selection and ML-informed node selection (i.e., _K_-B&B-NodeSelection). _K_-adaptability is a solution approach for two-stage robust optimization problems.

This directory contains the folders `src` and `results`:
- `src`: includes the source code of the paper. This folder is organized as follows: 
  - `src/<benchmark>/`: contains code adapted to a benchmark: **capital budgeting** (`cb`), **knapsack** (`kp`), and **shortest path** (`sp`).
  - `src/make_data.py`: run file to make ML training data.
  - `src/make_instances.py`: run file to generate test instances.
  - `src/run_ml.py`: run file to execute _K_-B&B-NodeSelection, i.e., ML-informed algorithm.
  - `src/run_random.py`: run file to execute _K_-B&B with default random node-selection.
  - `src/train_rf.py`: run file to train ML model with generated data.
- `results`: contains the raw results files. See `results/README.md` for a description of this folder.

## Dependencies
The following Python (3.10.13) packages are required to run this code: 
- `gurobi 10.0.3`
- `numpy 1.26.0`
- `pandas 2.1.1`
- `scikit-learn 1.3.0`

Note that `gurobi` is licensed software. You can install this via the instructions given [here](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python), and look [here](https://support.gurobi.com/hc/en-us/articles/12872879801105-How-do-I-retrieve-and-set-up-a-Gurobi-license) for how to obtain a license.

## Run experiments
This section includes commands to run the code for a benchmark.

**_Generate test instances_**
```commandline
python src/make_instances.py --problem <> --num_instances <> --N <>
```
- `problem` is the problem benchmark. Choose `cb`, `kp`, `sp_normal`, or `sp_sphere`.
- `num_instances` are the number of instances per problem and instance size.
- `N` is the instance size. E.g., for capital budgeting (`cb`), this is the number of projects.

Example: 
```commandline
python src/make_instances.py --problem cb --num_instances 10 --N 10
```

_**Run K-B&B (random node selection)**_
```commandline
python src/run_random.py --problem <> --inst_num <> --N <> --K <> --time_limit <>
```
- `inst_num` is the instance id.
- `K` is the _K_-adaptability parameter, i.e., number of different second-stage decisions.
- `time_limit` is the time limit (in minutes) of the algorithm.

Example:
```commandline
python src/run_random.py --problem cb --inst_num 1 --N 10 --K 4 --time_limit 30
```

**_Generate training data (preprocessing for K-B&B-NodeSelection)_**
```commandline
python src/make_data.py --problem <> --job_num <> --N <> --K <> --time_limit <> --num_instances <>
```
- `job_num` is the job id.
- `time_limit` is the total time limit (in minutes) spent on generating training data for one instance.
- `num_instances` is the number of instances based on which data is generated during one job in the time limit.

Example:
```commandline
python src/make_data.py --problem cb --job_num 1 --N 10 --K 4 --time_limit 10 --num_instances 12
```

**_Train random forest_**
```commandline
python src/train_rf.py --problem <> --N <> --K <> --min_train <> --h_train <> --ct <>
```
- `min_train` is the data generation duration of one instance.
- `h_train` is the total number of hours used to generate data
- `ct` is the classification threshold.

Example:
```commandline
python src/train_rf.py --problem cb --N 10 --K 4 --min_train 10 --h_train 2 --ct 5
```

**_Execute K-B&B-NodeSelection_**
```commandline
python src/run_ml.py --problem <> --inst_num <> --N <> --K <> --K_train <> --min_train <> --h_train <> --max_level <>
```
- `K_train` is the value of K used to generate training data.
- `max_level` is the level in the B&B tree upto which ML-informed node selection is applied.

Example: 
```commandline
python src/run_ml.py --problem cb --inst_num 1 --N 10 --K 4 --K_train 4 --min_train 10 --h_train 2 --max_level 40
```

[//]: # (## Ongoing Development)

[//]: # ()
[//]: # (This code is being developed on an on-going basis at the author's)

[//]: # ([Github page]&#40;https://github.com/estherjulien/KAdaptNS&#41;.)

[//]: # ()
[//]: # (## Support)

[//]: # ()
[//]: # (For support in using this software, submit an)

[//]: # ([issue]&#40;https://github.com/tkralphs/JoCTemplate/issues/new&#41;.)
