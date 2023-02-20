import lightgbm as lgb
import pandas as pd
import numpy as np
from src import SOURCE_DIR


def mae(y, yhat):
    return np.mean(abs(y - yhat))


iterable = [
    '_' + str(i) + '_' + str(j) for i in range(5) for j in range(5)
]


def extract_performance(ext, models='test_full'):
    dt_val_path = SOURCE_DIR + 'data/processed/cv_test/dt_val' + ext + '.pkl'
    model_path = SOURCE_DIR + 'output/models/' + models + '/model' + ext + '.txt'
    model = lgb.Booster(model_file=model_path)
    dt_val = pd.read_pickle(dt_val_path)
    if models == 'test_reduced':
        dt_val = dt_val.drop(['cat116', 'cat112'], axis=1)
    yhat = model.predict(dt_val.drop(['id', 'loss'], axis=1))

    print(f"Evalutating model {model_path} on data {dt_val_path}.")

    return mae(dt_val.loss, yhat)


performances_full = [extract_performance(i) for i in iterable]

df_full = pd.DataFrame(
    {
        'model': ['model' + i for i in iterable],
        'type': 'full',
        'performance': performances_full
    }
)

performances_reduced = [extract_performance(i, 'test_reduced') for i in iterable]

df_reduced = pd.DataFrame(
    {
        'model': ['model' + i for i in iterable],
        'type': 'reduced',
        'performance': performances_reduced
    }
)

pd.concat([df_full, df_reduced]).to_csv(
    SOURCE_DIR + 'output/full_vs_reduced.csv'
)