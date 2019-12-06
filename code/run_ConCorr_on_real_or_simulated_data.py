""" This code reproduces the results from the paper 'causal regularization' from Dominik Janzing, NeurIPS 2019, where
the algorithm ConCorr (for 'Confounder Correction') has been shown as proof of concept, but not necessarily as a method
to be used in practice.
"""


from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sc
import compare_regression_errors as cp


def normalize_data(x):
    """Normalizes numpy data matrix to data with unit variance.
    Note that this is not a "whitening" transformation because the correlations between
    the variables are kept.
    :param x: (n x d) data matrix
    :return:
    """
    Conxx = np.linalg.inv(np.diag(np.diag(np.cov(x))))
    Normalization = sc.linalg.sqrtm(Conxx)
    return np.dot(Normalization,x)


def read_data(filename):
    """Reads data files as input for ConCorr.
    :param filename:
    :return: pandas data frame samples x (d+1) where d denotes the number of input features.
    The (d+1)th dimension is for the output.
    """
    df = pd.read_csv(
        os.path.join(DATA_PATH, filename),
        sep=';',
        header=0,
        dtype=object)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column])
    return df


def ConCorr_and_cv(df):
    """Compares regression errors from ConCorr to the error with cross validation.
    Both Concorr and CV are computed with Ridge and Lasso regression.

    :param df: pandas data frame of dimension nx total_dim, where n is the number of
    samples and d = total_dim-1 the number of features X_j. The (d+1)th dimension is for
    the target variable Y.
    :return: 4-dimensional vector containing the relative squared errors (as defined in the NeurIPS paper) of
    (1) ConCorr with Lasso, (2) cross-validated Lasso, (3) ConCorr with Ridge, (4) cross-validated Ridge
    """
    # convert data frame to matrix x and vector y
    total_dim = df.shape[1]
    x = df.iloc[:,0:total_dim-1].transpose()
    y = df.iloc[:,total_dim-1].transpose()

    # ask whether features should be normalized
    nor = input('normalize data? y/n ')
    if nor == 'y':
        x = normalize_data(x)

    # ask which of the features should be dropped (as a method to generate confounding)
    components_dropped = input('which components should be dropped? Provide list with elements from 0 to ' + str(total_dim-2) + '  ')
    components_dropped = list(map(int, components_dropped.split()))
    components_kept = [i for i in range(total_dim-1) if i not in components_dropped]

    a_full_lasso = cp.cv_lasso(x,y)
    a_full_ridge = cp.cv_ridge(x,y)


    a_lasso = a_full_lasso[components_kept]
    a_ridge = a_full_ridge[components_kept]

    x_reduced = np.take(x,components_kept,axis=0)

    rel_error_concorr_lasso = cp.relative_error(cp.concorr_lasso(x_reduced, y), a_lasso)
    rel_error_concorr_ridge = cp.relative_error(cp.concorr_ridge(x_reduced, y), a_ridge)

    rel_error_cvlasso = cp.relative_error(cp.cv_lasso(x_reduced,y),a_lasso)
    rel_error_cvridge = cp.relative_error(cp.cv_ridge(x_reduced,y),a_ridge)

    rel_error_unregularized = cp.relative_error(cp.unregularized_regression(x_reduced,y),a_lasso)

    print('rel error lasso with ConCorr: ')
    print(rel_error_concorr_lasso)
    print('rel error lasso with cross validation: ')
    print(rel_error_cvlasso)

    print('rel error ridge with ConCorr: ')
    print(rel_error_concorr_ridge)
    print('rel error lasso with cross validation: ')
    print(rel_error_cvridge)


    print('rel error unreg: ')
    print(rel_error_unregularized)
    return [rel_error_concorr_lasso, rel_error_cvlasso, rel_error_concorr_ridge, rel_error_cvridge]
    

def try_wine_data():
    filename = './data_taste_of_wine/winequality-red.csv'
    df = pd.read_csv(filename, sep=';', header=0, dtype=object)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column])
    ConCorr_and_cv(df)


def try_optical_data():

    data_path = './data_from_optical_device/'
    file_list = os.listdir(data_path)
    print(file_list)
    file_list = [f for f in file_list if 'confounder' in f and 'random' not in f and 'READ' not in f]
    print(file_list)
    n_files = len(file_list)
    print(n_files)
    rel_concorr_lasso = np.zeros(n_files)
    rel_cvlasso = np.zeros(n_files)
    rel_concorr_ridge = np.zeros(n_files)
    rel_cvridge = np.zeros(n_files)
    for j in range(11):
        print(file_list[j])
        df = pd.read_csv(
        os.path.join(data_path, file_list[j]),
        sep=' ',
        header=None,
        dtype=object)
        for column in df.columns:
            df[column] = pd.to_numeric(df[column])
        [rel_concorr_lasso[j], rel_cvlasso[j], rel_concorr_ridge[j], rel_cvridge[j]] = ConCorr_and_cv(df)
    plt.plot(rel_cvlasso, rel_concorr_lasso, 'bo')
    grid = [i/100 for i in range(100)]
    plt.plot(grid,grid,'r.')
    plt.xlabel('relative squared error cv', fontsize=18)
    plt.ylabel('relative squared error ConCorr', fontsize=18)
    plt.show()

    plt.plot(rel_cvridge, rel_concorr_ridge, 'bo')
    grid = [i/100 for i in range(100)]
    plt.plot(grid,grid,'r.')
    plt.xlabel('relative squared error cv', fontsize=18)
    plt.ylabel('relative squared error ConCorr', fontsize=18)
    plt.show()

def main():
    """Reproduces all experiments reported in the NeurIPS paper.
    All outputs are either provided in the command line or as figures.
    :return: none.
    """
    print()
    print('Note that the main intention of the algorithm ConCorr is a proof of concept that sometimes causal statements')
    print('require stronger regularization than optimal statistical predictability. It is likely that ConCorr is not')
    print('appropriate for many real data sets since it relies on a highly idealized toy model of confounding')
    print()
    print('Which experiment from the NeurIPS paper you want to reproduce?')
    print('Section 4.1, simulated data (1)')
    print('Section 4.2, real data from the optical device (2)')
    print('Section 4.2, real data on taste of wine (3)')
    answer = input('choose one of the three options ')
    if answer == '1':
        cp.main()
    elif answer == '2':
        try_optical_data()
    else: try_wine_data()

if __name__ == '__main__':
    main()
