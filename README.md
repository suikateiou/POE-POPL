# Pareto-Optimal Estimation and Policy Learning on Short-term and Long-term Treatment Effects

This is the source code of Pareto-Optimal Estimation (POE) and Pareto-Optimal Policy Learning (POPL), a method to determine the appropriate treatment value that would achieve Pareto optimality between short-term and long-term outcomes.


## Organization

```
├── long_term                      
│   ├── dataset                    # datasets
│   ├── weighted_sum               # intial solutions for POE
│   ├── cpmtl                      # Pareto-optimal solutions for POE
│   ├── policy_initial             # intial solutions for POPL
│   └── policy_pareto              # Pareto-optimal solutions for POPL
├── pareto
│   ├── datasets
│   │   └── ensemble.py            # loading dataset
│   ├── networks
│   │   ├── MutualInfor
│   │   │   ├── mi_estimators.py   # mutual information module
│   │   │   └── module.py          # networks for POE
│   │   ├── multi_lenet.py         # base networks
│   │   └── policynet.py           # networks for POPL
│   ├── optim                      # optimizers for Pareto exploration
│   │   ├── hvp_solver.py
│   │   ├── kkt_solver.py
│   │   ├── linalg_solver.py
│   │   ├── min_norm_solver.py
│   │   └── policy_hvp_solver.py
├── weighted_sum.py                # initially training POE 
├── cpmtl.py                       # Pareto optimization for POE
├── policy_initial.py              # initially training POPL
├── policy_pareto.py               # Pareto optimization for POPL
└── requirements.txt               # requirements for conda environment
```





## Environment
- Ubuntu 18.04.4
- CUDA 11.7
- Conda 4.9.2



## Installation

Dependencies are listed and you can install them via Pip.
```
pip install -r requirements.txt
```



## Run the code

To get the intial solutions of POE:
```
python weighted_sum.py --dataset simulation --bs 64 --lr 1e-4 --num_epochs 20 --seed 42
```

To get the Pareto-optimal solutions of POE:
```
python cpmtl.py --dataset simulation --bs 64 --lr 1e-4 --num_steps 20 --seed 42
```

To get the intial solutions of POPL:
```
python policy_initial.py --dataset simulation --bs 64 --lr 1e-4 --num_epochs 20 --seed 42
```

To get the Pareto-optimal solutions of POPL:
```
python policy_pareto.py --dataset simulation --bs 64 --lr 1e-4 --num_steps 20 --seed 42
```


There are several arguments to be assigned:

|   Argument  |                     Definition                      |
| :---------: | :-------------------------------------------------: |
|   dataset   |     dataset name (simulation, ihdp, jobs, twins)    |
|      bs     |                    batch size                       |
|      lr     |                  learning rate                      |
|     seed    |                   random seed                       |
|  num_epochs |          number of training epochs in POE           |
|  num_steps  |          number of training epochs in POPL          |
|  num_tasks  | number of multiple tasks (3 for POE and 2 for POPL) |







## Contact
If you have any questions, you can contact us by wangyingrong@zju.edu.cn.
