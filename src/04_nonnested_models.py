from src import SOURCE_DIR
from src.tuners import fit_nonnested_model
import pandas as pd
import warnings
import optuna
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning, module="optuna.*")
optuna.logging.set_verbosity(optuna.logging.INFO)

iterable = ['_' + str(i) + '_' + str(j) for i in range(3) for j in range(3)]

model_path = SOURCE_DIR + 'output/models/nonnested/'
data_path = SOURCE_DIR + 'data/processed/cv/'


def iterate(ext):
    dt_trn = pd.read_pickle(data_path + 'dt_trn' + ext + '.pkl')
    dt_val = pd.read_pickle(data_path + 'dt_val' + ext + '.pkl')

    model_name = 'model' + ext

    fit_nonnested_model(
        dt_trn, dt_val, model_name, model_path, optuna_seed=iterable.index(ext)
    )

    return True


if __name__ == '__main__':
    with Pool(5) as pool:
        for name in pool.map(iterate, iterable):
            print(f'Fitted {name}')