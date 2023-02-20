from src import SOURCE_DIR
import pandas as pd
import lightgbm as lgb
import optuna
import json
import warnings
from lightgbm import log_evaluation, early_stopping
import joblib

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning, module="optuna.*")
optuna.logging.set_verbosity(optuna.logging.INFO)

iterable = ['_' + str(i) + '_' + str(j) for i in range(5) for j in range(5)]

hyperparams = dict()

df_par = pd.DataFrame()
for i in iterable:
    with open(SOURCE_DIR + 'output/models/test_reduced/params_model' + i + '.json') as f:
        df_par = pd.concat([df_par, pd.DataFrame([json.load(f)])])

initial_params = {
    'lambda_l1': 0.1175,
    'lambda_l2': 0.3618,
    'num_leaves': 166.48,
    'feature_fraction': 0.48,
    'bagging_fraction': 0.9,
    'bagging_freq': 2,
    'min_child_samples': 30
}

dt_trn = pd.read_pickle(SOURCE_DIR + 'data/processed/dt_trn.pkl')
dt_val = pd.read_pickle(SOURCE_DIR + 'data/processed/dt_test.pkl')

trn_dataset = lgb.Dataset(
    dt_trn.drop(['id', 'loss', 'cat112', 'cat116'], axis=1),
    label=dt_trn.loss
)

val_dataset = lgb.Dataset(
    dt_val.drop(['id', 'loss', 'cat112', 'cat116'], axis=1),
    label=dt_val.loss
)


def objective(trial: optuna.trial):
    params = {
        'objective': 'regression_l1',
        'metric': 'l1',
        'learning_rate': 0.02,
        'num_threads': 6,
        'seed': 42,
        'verbosity': -1,
        'feature_pre_filter': False,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 2, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 40, 300),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.8),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 6),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 150)
    }

    gbm = lgb.train(
        params=params,
        train_set=trn_dataset,
        valid_sets=val_dataset,
        num_boost_round=2000,
        callbacks=[log_evaluation(100), early_stopping(100)]
    )

    return gbm.best_score['valid_0']['l1']


sampler = optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True, seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.enqueue_trial(initial_params)
study.optimize(objective, n_trials=30, n_jobs=1)
joblib.dump(study, SOURCE_DIR + 'output/models/final_tune.pkl')

study = joblib.load(SOURCE_DIR + 'output/models/final_tune.pkl')

opt_params = {
    'objective': 'regression_l1',
    'metric': 'l1',
    'learning_rate': 0.01,
    'num_threads': 6,
    'seed': 42,
    'verbosity': -1,
    'feature_pre_filter': False,
    'lambda_l1': study.best_params['lambda_l1'],
    'lambda_l2': study.best_params['lambda_l2'],
    'num_leaves': study.best_params['num_leaves'],
    'feature_fraction': study.best_params['feature_fraction'],
    'bagging_fraction': study.best_params['bagging_fraction'],
    'bagging_freq': study.best_params['bagging_freq'],
    'min_child_samples': study.best_params['min_child_samples']
}

final_model = lgb.train(
    params=opt_params,
    train_set=trn_dataset,
    valid_sets=val_dataset,
    num_boost_round=5000,
    callbacks=[log_evaluation(100), early_stopping(100)]
)

dt_test = pd.read_csv(SOURCE_DIR + 'data/raw/test.csv')

for col in dt_test.columns:
    col_type = dt_test[col].dtype
    if col_type == 'object' or col_type.name == 'category':
        dt_test[col] = dt_test[col].astype('category')

yhat = final_model.predict(dt_test.drop(['id', 'cat112', 'cat116'], axis=1))
output = pd.DataFrame({'id': dt_test.id, 'loss': yhat})
output.to_csv(SOURCE_DIR + 'output/kaggle_submission.csv', index=False)
