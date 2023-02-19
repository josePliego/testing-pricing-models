import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

source_dir = '/Users/josbop/Documents/Duke/testing-pricing-models/'

dt = pd.read_csv(source_dir + "data/raw/train.csv")

for col in dt.columns:
    col_type = dt[col].dtype
    if col_type == 'object' or col_type.name == 'category':
        dt[col] = dt[col].astype('category')

splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
split = splitter.split(dt, groups=dt['id'])
trn_idx, test_idx = next(split)
dt_trn = dt.iloc[trn_idx]
dt_test = dt.iloc[test_idx]

dt_trn.to_pickle(source_dir + 'data/processed/dt_trn.pkl')
dt_test.to_pickle(source_dir + 'data/processed/dt_test.pkl')
