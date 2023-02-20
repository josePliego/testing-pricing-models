import pandas as pd
from sklearn.model_selection import RepeatedKFold

source_dir = '/Users/josbop/Documents/Duke/testing-pricing-models/'

dt = pd.read_pickle(source_dir + "data/processed/dt_trn.pkl")

splitter = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
split = splitter.split(dt)

for n, (trn_idx, val_idx) in enumerate(split):
    print(f'Saving repeat {n // 5} fold {n % 5}')
    rep = n//5
    fold = n % 5
    trn = dt.iloc[trn_idx]
    val = dt.iloc[val_idx]
    trn.to_pickle(
        source_dir + 'data/processed/cv_test/dt_trn_' + str(rep) + '_' + str(fold) + '.pkl'
    )
    val.to_pickle(
        source_dir + 'data/processed/cv_test/dt_val_' + str(rep) + '_' + str(fold) + '.pkl'
    )
    