import pandas as pd
from sklearn.model_selection import RepeatedKFold

source_dir = '//'

dt = pd.read_csv(source_dir + "data/raw/train.csv")

cat_cols = []
for col in dt.columns:
    coltype = dt[col].dtype
    if coltype == 'object' or coltype.name == 'category':
        dt[col] = dt[col].astype('category')
        cat_cols.append(col)

splitter = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
split = splitter.split(dt)

for n, (trn_idx, val_idx) in enumerate(split):
    print(f'Saving repeat {n // 5} fold {n % 5}')
    rep = n//5
    fold = n % 5
    trn = dt.iloc[trn_idx]
    val = dt.iloc[val_idx]
    trn.to_pickle(
        source_dir + 'data/processed/dt_trn_' + str(rep) + '_' + str(fold) + '.pkl'
    )
