import optuna

from optuna.samplers import TPESampler

import math

def objective(trial):
  
    tree_depth = trial.suggest_int('tree_depth',1, 15)
    min_n = trial.suggest_int('min_n',2, 200)
    loss_reduction = trial.suggest_float("loss_reduction", 1e-8, 5.0, log=True)
    sample_size = trial.suggest_float("sample_size", 0.2, 1.0)
    mtry = trial.suggest_int("mtry", 1, 50)
    learn_rate = trial.suggest_float("learn_rate", 1e-8, 5.0, log=True)
    
    out = r.get_hyperparams(
      trees = 1000,
      tree_depth = tree_depth,
      min_n = min_n, 
      loss_reduction = loss_reduction,            
      sample_size = sample_size, 
      mtry = mtry,         
      learn_rate = learn_rate  
      )
      
    return out

study = optuna.create_study(
  direction="minimize", 
  sampler=TPESampler(seed=42)
  )
  
study.optimize(objective, n_trials=50)

print("The best found Accuracy = {}".format(round(study.best_value,3)))
