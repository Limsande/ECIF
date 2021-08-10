"""
Reproduces the best model from the paper, i.e. ECIF6:LD-GBT, or
ECIF8.5::LD-GBT if --cross-validate is given.
"""

import pandas as pd
import pickle
import sys

from datetime import datetime
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_validate
from scipy.stats import pearsonr
from math import sqrt


def load_data(cutoff: float) -> DataFrame:
    """
    Loads descriptors and binding data for given cutoff.
    """
    # Load ECIF
    ecif = pd.read_csv(f'Descriptors/ECIF_{cutoff}.csv')
    # Load ligand descriptors
    ligand_descriptors = pd.read_csv("Descriptors/RDKit_Descriptors.csv")
    # Load binding affinity data
    binding_data = pd.read_csv("Descriptors/BindingData.csv")

    # Merge descriptors
    ecif = ecif.merge(ligand_descriptors, left_on="PDB", right_on="PDB")
    ecif = ecif.merge(binding_data, left_on="PDB", right_on="PDB")
    
    return ecif


def split_train_test(ecif: DataFrame) -> (DataFrame, DataFrame, DataFrame, DataFrame):
    """
    Splits given data into training and test set according to column "SET".
    Returns x_train, y_train, x_test, y_test.
    """
    x_train = ecif[ecif["SET"] == "Train"][list(ecif.columns)[1:-2]]
    y_train = ecif[ecif["SET"] == "Train"]["pK"]

    x_test = ecif[ecif["SET"] == "Test"][list(ecif.columns)[1:-2]]
    y_test = ecif[ecif["SET"] == "Test"]["pK"]

    return x_train, y_train, x_test, y_test


def eval(GBT: GradientBoostingRegressor, x_test, y_test):
    """
    Evaluates the model on given data. Returns Pearson's R and mean squared error.
    """
    y_pred_GBT = GBT.predict(x_test)
    return pearsonr(y_test, y_pred_GBT)[0], sqrt(mean_squared_error(y_test,y_pred_GBT))


def pearsonr_score(y_train, y_test) -> float:
    """Wrapper to be used with cross_validate."""
    return pearsonr(y_train, y_test)[0]


def main(do_cross_val: bool):
    if do_cross_val:
        cutoff = 8.5
    else:
        cutoff = 6.0

    print(f'Loading data for cutoff={cutoff}...')
    ecif = load_data(cutoff)
    x_train, y_train, x_test, y_test = split_train_test(ecif)
    print(f'Done. Loaded descriptors: {len(x_train)} train, {len(x_test)} test.')
    
    GBT = GradientBoostingRegressor(
            random_state=1206,
            n_estimators=20000,
            max_features="sqrt",
            max_depth=8,
            min_samples_split=3,
            learning_rate=0.005,
            loss="ls",
            subsample=0.7
        )
        
    if cross_val:
        print('Training model with 10-fold cross validation...')
        scoring_funcs = {
            'mse': make_scorer(mean_squared_error, greater_is_better=False),
            'pearsonr': make_scorer(pearsonr_score)
        }
        start_time = datetime.now()
        scores = cross_validate(GBT, x_train, y_train, scoring=scoring_funcs, cv=10)
    else:
        print('Training model...')
        start_time = datetime.now()
        GBT.fit(x_train, y_train)  
    
    elapsed_time = str(datetime.now() - start_time).split('.')[0]  # Remove microseconds
    print(f'Done. Took {elapsed_time}.')
    
    if cross_val:
        pearson = scores['test_pearsonr'].mean()
        mse = scores['test_mse'].mean() * (-1)  # sign flipped in cross-val because maximization
        print('Pearson correlation coefficient for GBT:', pearson, '(mean accross all splits)')
        print('RMSE for GBT:', mse, '(mean accross all splits)')
    else:
        print('Evaluating model on test set...')
        pearson, mse = eval(GBT, x_test, y_test)
        print('Pearson correlation coefficient for GBT:', pearson)
        print('RMSE for GBT:', mse)

    pickle.dump(GBT, open(f'ECIF{cutoff}_LD_GBT.pkl', 'wb'))
    print(f'Saved model to ECIF{cutoff}_LD_GBT_cv.pkl. Bye')

if __name__ == '__main__':
    cross_val = False
    if len(sys.argv[1:]) == 1 and sys.argv[1] == '--cross-val':
        cross_val = True
    elif len(sys.argv[1:]) > 0:
        sys.exit('Usage: reproduce_best_model [--cross-val]')

    main(do_cross_val=cross_val)

