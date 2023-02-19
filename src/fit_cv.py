import pandas as pd
import optuna
import joblib
import warnings
import lightgbm as lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation


def make_data(df_trn, df_val):
    trnDataset = lgb.Dataset(
        df_trn.drop(['loss', 'id'], axis=1),
        label=df_trn.loss
    )
    valDataset = lgb.Dataset(
        df_val.drop(['loss', 'id'], axis=1),
        label=df_val.loss
    )

    return trnDataset, valDataset


def tune_model(trnDataset, valDataset, param, n_trials=30):

    def objective(trial):
        gbm = lgb.train(
            params=param,
            train_set=trnDataset,
            valid_sets=valDataset,
            num_boost_round=1000,
            callbacks=[log_evaluation(1000), early_stopping(100)]
        )
