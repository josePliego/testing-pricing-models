import lightgbm as lgb
import pandas as pd
import numpy as np
from src import SOURCE_DIR


# model = lgb.Booster(model_file=SOURCE_DIR + 'output/models/nested_full/model_0_0.txt')
# dt_val = pd.read_pickle(SOURCE_DIR + 'data/processed/cv/dt_val_0_0.pkl')
# valDataset = lgb.Dataset(
#     dt_val.drop(['id', 'loss'], axis=1),
#     label=dt_val.loss
# )


def mae(y, yhat):
    return np.mean(abs(y - yhat))


# yhat = model.predict(dt_val.drop(['id', 'loss'], axis=1))
# print(mae(dt_val.loss, yhat))
# nested_model_paths = SOURCE_DIR + 'output/models/nested_full/'
iterable = [
    '_' + str(i) + '_' + str(j) for i in range(3) for j in range(3)
]

nested_importances = dict()


def extract_performance(ext, models='nested_full', importance_list=None):
    dt_val_path = SOURCE_DIR + 'data/processed/cv/dt_val' + ext + '.pkl'
    model_path = SOURCE_DIR + 'output/models/' + models + '/model' + ext + '.txt'
    model = lgb.Booster(model_file=model_path)
    dt_val = pd.read_pickle(dt_val_path)
    yhat = model.predict(dt_val.drop(['id', 'loss'], axis=1))

    if importance_list is not None:
        importances = dict()
        importances['features'] = model.feature_name()
        importances['importance'] = model.feature_importance()
        importance_list['model' + ext] = importances

    return mae(dt_val.loss, yhat)


performances_nested = [
    extract_performance(i, importance_list=nested_importances) for i in iterable
]

df_nested = pd.DataFrame(
    {
        'model': ['model' + i for i in iterable],
        'type': 'nested_full',
        'performance': performances_nested
    }
)

dt_importance = pd.DataFrame()

for i in nested_importances.keys():
    new_df = pd.DataFrame(
        {
            'model': i,
            'type': 'nested_full',
            'var': nested_importances[i]['features'],
            'importance': nested_importances[i]['importance']
        }
    )

    dt_importance = pd.concat([dt_importance, new_df])

dt_importance.to_csv(SOURCE_DIR + 'output/nested_importance.csv')

performances_nonnested = [extract_performance(i, 'nonnested') for i in iterable]

df_nonnested = pd.DataFrame(
    {
        'model': ['model' + i for i in iterable],
        'type': 'nonnested',
        'performance': performances_nonnested
    }
)

pd.concat([df_nested, df_nonnested]).to_csv(
    SOURCE_DIR + 'output/nested_vs_not.csv'
)



# Importances
