# Code for the paper:
> **Machine Learning for K-adaptability in Two-stage Robust Optimization**  
> *Esther Julien, Krzysztof Postek, S.Ilker Birbil*

To run the code, the following packages have to be installed: `numpy`, `scikit-learn`,
`gurobipy`, and `joblib`.

## Make instances
For making instances for the capital budgeting problem, RUN: 
```bash
python CapitalBudgeting/make_instances.py <num_instances> <N> 
```
where `num_instances` is the number of different generated instances; `N` is the instance size (number of possible projects to invest in)

and for the shortest path problem, RUN: 
```bash
python ShortestPath/make_instances.py <num_instances> <N> <sphere>
```
where `N` is the number of nodes in the graph; `sphere` is boolean: 1 if 3-D graph on a sphere instances, and 0 for 2-D graph instances.

## Solving K-adaptability
### K-B&B algorithm
The K-B&B algorithm is given in ```<problem_type>/Method/Random```, where `problem_type` is `CapitalBudgeting` if 
we solve the capital budgeting problem, and `ShortestPath` if we solve the shortest path problem.

RUN in terminal:
```bash
python <problem_type>/run_random.py <i> <N> <K>
```
where `i` is the instance number; `N` is the instance size; `K` is the number of subsets.

EXAMPLE: 
```bash
python CapitalBudgeting/run_random.py 0 10 4
```

### K-B&B-NodeSelection algorithm
The K-B&B-NodeSelection algorithm is given in ```<problem_type>/Method/SucPred```, where `problem_type` is `CapitalBudgeting` if 
we solve the capital budgeting problem, and `ShortestPath` if we solve the shortest path problem.

RUN:
```bash
python <problem_type>/run_suc_pred.py <i> <N> <K> <K_ML> <minutes> <hours> <ct> <max_level>
```
where `K_ML` is the value of `K` used for creating the training data; `minutes` is the number of minutes spent per 
instance for generating training data; `hours` is the total  number of hours spend on creating the training data set;
`ct` is the classification threshold for determining the class of a data point corresponding to its success probability; 
`max_level` is the level in the tree up to where ML guided node selection is used.

EXAMPLE: 
```bash
python <problem_type>/run_suc_pred.py 0 10 4 6 10 2 5 40
```

### Generate training data
The methods of generating training data are given in `<problem_type>/Method/SucPredData.py`.
RUN:
```bash
python <problem_type>/make_data.py <job_num> <N> <K_ML> <time_limit>
```

EXAMPLE:
```bash
python <problem_type>/make_data.py 1 10 4 5
```


#### Train random forest
In the folder `MLTraining` you can find the code for training a random forest. 

### Preprocessed data

The `<problem_type>/Data` folder consists of:

- `Results`:

  - `Results/Decisions/inst_results` has the results of K-B&B and K-B&B-NodeSelection

  - `Results/TrainData/inst_results` has the results of generating training data

- `Instances` for some example test instances (of the smallest size)

- `RunInfo` on metadata of generating training data
