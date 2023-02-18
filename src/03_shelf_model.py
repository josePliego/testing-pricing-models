import pandas as pd
import optuna
import optuna.integration.lightgbm as lgb
import json
from lightgbm import early_stopping
from lightgbm import log_evaluation
from sklearn.model_selection import GroupShuffleSplit
import warnings
# from matplotlib import pyplot as plt

source_dir = '/Users/josbop/Documents/Duke/testing-pricing-models/'

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning, module="optuna.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

dt = pd.read_csv(source_dir + "data/raw/train.csv")

cat_cols = []
for col in dt.columns:
    coltype = dt[col].dtype
    if coltype == 'object' or coltype.name == 'category':
        dt[col] = dt[col].astype('category')
        cat_cols.append(col)

splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
split = splitter.split(dt, groups=dt['id'])
trn_idx, val_idx = next(split)
dt_trn = dt.iloc[trn_idx]
dt_val = dt.iloc[val_idx]

dt_trn.to_pickle(source_dir + 'data/processed/dt_trn.pkl')
dt_val.to_pickle(source_dir + 'data/processed/dt_val.pkl')

trnDataset = lgb.Dataset(
    dt_trn.drop(['loss', 'id'], axis=1),
    label=dt_trn.loss
)
valDataset = lgb.Dataset(
    dt_val.drop(['loss', 'id'], axis=1),
    label=dt_val.loss
)

param = {
    'objective': 'regression_l1',
    'metric': 'l1',
    'seed': 42,
    'verbosity': -1,
    'feature_pre_filter': False,
}

model = lgb.train(
    params=param,
    train_set=trnDataset,
    valid_sets=valDataset,
    num_boost_round=1000,
    callbacks=[log_evaluation(-1), early_stopping(100)],
    optuna_seed=42
)

print(model.params)

with open(source_dir + 'output/models/01_shelf_params.json', 'w') as f:
    json.dump(model.params, f)

# To load parameters:
# with open(source_dir + 'output/models/01_shelf_params.json') as f:
#     json.load(f)

# joblib.dump(tuner, source_dir + 'output/models/first_study.pkl')

# lgbm.train(
#     params=study.best_params,
#     train_set=trnDataset,
#
# )

# https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258

# def objective(trial):
#
#     param = {
#         'objective': 'regression',
#         'metric': 'rmse',
#         'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
#         'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
#         'num_leaves': trial.suggest_int('num_leaves', 2, 256),
#         'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
#         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#         'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
#         'learning_rate': trial.suggest_float('learning_rate', 0.05, 1),
#         'num_threads': 4,
#         'seed': 42,
#         'verbosity': -1,
#         'feature_pre_filter': False,
#     }
#
#     gbm = lgb.train(
#         param,
#         X_train,
#         valid_sets=X_val,
#         num_boost_round=1000,
#         callbacks=[log_evaluation(100), early_stopping(100)]
#     )
#     return gbm.best_score['valid_0']['rmse']
#
#
# sampler = optuna.samplers.TPESampler(n_startup_trials=1, multivariate=True, seed=42)
# study = optuna.create_study(direction='minimize', sampler=sampler)
# study.optimize(objective, n_trials=50, n_jobs=4)
#
# print('Number of finished trials:', len(study.trials))

# TODO: first find good starting points with Optuna implementation, then tune as usual
# TODO: REMOVE ID COLUMN!!!!!!!
# TODO: for portfolio, compare one-hot encoding vs LightGBM categorical techniques
# 30 trials: 1867.54
# 50 trials: 1849.03
