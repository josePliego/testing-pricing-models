import json
import lightgbm
from lightgbm import early_stopping
from lightgbm import log_evaluation
from optuna.integration import lightgbm as lgb
from sklearn.model_selection import KFold


def fit_nested_model(dt_trn, dt_val, model_name, model_path, optuna_seed, nfolds=3):
    trn_dataset = lgb.Dataset(
        dt_trn.drop(['loss', 'id'], axis=1),
        label=dt_trn.loss
    )
    val_dataset = lgb.Dataset(
        dt_val.drop(['loss', 'id'], axis=1),
        label=dt_val.loss
    )

    params = {
        'objective': 'regression_l1',
        'metric': 'l1',
        'learning_rate': 0.1,
        'num_threads': 0,
        'seed': 42,
        'verbosity': -1,
        'feature_pre_filter': False
    }

    tuner = lgb.LightGBMTunerCV(
        params=params,
        train_set=trn_dataset,
        folds=KFold(n_splits=nfolds, random_state=42, shuffle=True),
        num_boost_round=1000,
        callbacks=[early_stopping(100), log_evaluation(0)],
        seed=42,
        optuna_seed=optuna_seed
    )

    tuner.run()

    with open(model_path + 'params_' + model_name + '.json', 'w') as f:
        json.dump(tuner.best_params, f)

    with open(model_path + 'params_' + model_name + '.json') as f:
        best_params = json.load(f)

    opt_params = {
        'objective': 'regression_l1',
        'metric': 'l1',
        'learning_rate': 0.02,
        'num_threads': 0,
        'seed': 42,
        'verbosity': -1,
        'lambda_l1': best_params['lambda_l1'],
        'lambda_l2': best_params['lambda_l2'],
        'num_leaves': best_params['num_leaves'],
        'feature_fraction': best_params['feature_fraction'],
        'bagging_fraction': best_params['bagging_fraction'],
        'bagging_freq': best_params['bagging_freq'],
        'min_child_samples': best_params['min_child_samples']
    }

    model = lightgbm.train(
        params=opt_params,
        train_set=trn_dataset,
        valid_sets=val_dataset,
        num_boost_round=2500,
        callbacks=[log_evaluation(0), early_stopping(100)]
    )

    model.save_model(model_path + model_name + '.txt')

    return model.best_score['valid_0']['l1']


def fit_nonnested_model(dt_trn, dt_val, model_name, model_path, optuna_seed):
    trn_dataset = lgb.Dataset(
        dt_trn.drop(['loss', 'id'], axis=1),
        label=dt_trn.loss
    )
    val_dataset = lgb.Dataset(
        dt_val.drop(['loss', 'id'], axis=1),
        label=dt_val.loss
    )

    params = {
        'objective': 'regression_l1',
        'metric': 'l1',
        'learning_rate': 0.1,
        'num_threads': 0,
        'seed': 42,
        'verbosity': -1,
        'feature_pre_filter': False
    }

    tuner = lgb.LightGBMTuner(
        params=params,
        train_set=trn_dataset,
        valid_sets=val_dataset,
        num_boost_round=1000,
        callbacks=[early_stopping(100), log_evaluation(0)],
        optuna_seed=optuna_seed
    )

    tuner.run()

    with open(model_path + 'params_' + model_name + '.json', 'w') as f:
        json.dump(tuner.best_params, f)

    with open(model_path + 'params_' + model_name + '.json') as f:
        best_params = json.load(f)

    opt_params = {
        'objective': 'regression_l1',
        'metric': 'l1',
        'learning_rate': 0.02,
        'num_threads': 0,
        'seed': 42,
        'verbosity': -1,
        'lambda_l1': best_params['lambda_l1'],
        'lambda_l2': best_params['lambda_l2'],
        'num_leaves': best_params['num_leaves'],
        'feature_fraction': best_params['feature_fraction'],
        'bagging_fraction': best_params['bagging_fraction'],
        'bagging_freq': best_params['bagging_freq'],
        'min_child_samples': best_params['min_child_samples']
    }

    model = lightgbm.train(
        params=opt_params,
        train_set=trn_dataset,
        valid_sets=val_dataset,
        num_boost_round=2500,
        callbacks=[log_evaluation(0), early_stopping(100)]
    )

    model.save_model(model_path + model_name + '.txt')

    return model.best_score['valid_0']['l1']
