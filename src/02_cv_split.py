import pandas as pd
from sklearn.model_selection import RepeatedKFold

source_dir = '/Users/josbop/Documents/Duke/testing-pricing-models/'

dt = pd.read_pickle(source_dir + "data/processed/dt_trn.pkl")

splitter = RepeatedKFold(n_splits=3, n_repeats=3, random_state=42)
split = splitter.split(dt)

for n, (trn_idx, val_idx) in enumerate(split):
    print(f'Saving repeat {n // 3} fold {n % 3}')
    rep = n//3
    fold = n % 3
    trn = dt.iloc[trn_idx]
    val = dt.iloc[val_idx]
    trn.to_pickle(
        source_dir + 'data/processed/cv/dt_trn_' + str(rep) + '_' + str(fold) + '.pkl'
    )
    val.to_pickle(
        source_dir + 'data/processed/cv/dt_val_' + str(rep) + '_' + str(fold) + '.pkl'
    )
    