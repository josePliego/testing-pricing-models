import pandas as pd
import joblib
import optuna
import json
import lightgbm as lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation
import warnings

source_dir = '/Users/josbop/Documents/Duke/testing-pricing-models/'

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning, module="optuna.*")
optuna.logging.set_verbosity(optuna.logging.INFO)

trn_path = source_dir + 'data/processed/dt_trn.bin'
val_path = source_dir + 'data/processed/dt_val.bin'
dt_trn = pd.read_pickle(source_dir + 'data/processed/dt_trn.pkl')
dt_val = pd.read_pickle(source_dir + 'data/processed/dt_val.pkl')

trnDataset = lgb.Dataset(
    dt_trn.drop(['loss', 'id'], axis=1),
    label=dt_trn.loss
)
valDataset = lgb.Dataset(
    dt_val.drop(['loss', 'id'], axis=1),
    label=dt_val.loss
)

with open(source_dir + 'output/models/01_shelf_params.json') as f:
    shelf_params = json.load(f)


def objective(trial):
    param = {
        'objective': 'regression_l1',
        'metric': 'l1',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 512),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.0, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.0, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 10.0, log=True),
        'cat_l2': trial.suggest_float('cat_l2', 1e-8, 20.0, log=True),
        'cat_smooth': trial.suggest_float('cat_smooth', 1e-8, 20.0, log=True),
        'num_threads': 0,
        'seed': 42,
        'verbosity': -1,
        'feature_pre_filter': False,
    }

    gbm = lgb.train(
        params=param,
        train_set=trnDataset,
        valid_sets=valDataset,
        num_boost_round=1000,
        callbacks=[log_evaluation(-1), early_stopping(100)]
    )

    return gbm.best_score['valid_0']['l1']


sampler = optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True, seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.enqueue_trial(shelf_params)
study.optimize(objective, n_trials=30, n_jobs=2)

optuna.visualization.matplotlib.plot_param_importances(study)
optuna.visualization.matplotlib.plot_slice(study, params=['learning_rate', 'bagging_fraction'])
optuna.visualization.matplotlib.plot_contour(study)
optuna.visualization.matplotlib.plot_optimization_history(study)

joblib.dump(study, source_dir + 'output/models/shelf_tune/shelf_tune_study.pkl')
study = joblib.load(source_dir + 'output/models/shelf_tune/shelf_tune_study.pkl')

best_params = study.best_params

params = {
    'objective': 'regression_l1',
    'metric': 'l1',
    'lambda_l1': best_params['lambda_l1'],
    'lambda_l2': best_params['lambda_l2'],
    'num_leaves': best_params['num_leaves'],
    'feature_fraction': best_params['feature_fraction'],
    'bagging_fraction': best_params['bagging_fraction'],
    'bagging_freq': best_params['bagging_freq'],
    'min_child_samples': best_params['min_child_samples'],
    'learning_rate': best_params['learning_rate'],
    'cat_l2': best_params['cat_l2'],
    'cat_smooth': best_params['cat_smooth'],
    'num_threads': 4,
    'seed': 42,
    'verbosity': -1,
}

final_model = lgb.train(
    params=params,
    train_set=trnDataset,
    valid_sets=valDataset,
    num_boost_round=2000,
    callbacks=[log_evaluation(100), early_stopping(100)]
)

final_model.save_model(source_dir + 'output/models/shelf_tune/shelf_tune.txt')

pd.DataFrame(
    {'var': trnDataset.feature_name, 'imp': final_model.feature_importance()}
).to_csv(source_dir + 'output/models/shelf_tune/shelf_tune_imp.csv')

# Best score: 1137.9023741125961
