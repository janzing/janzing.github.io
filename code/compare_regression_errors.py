"""
This program explores causal regularization using simulated data.
It tests to what extent regularization via the ConCorr algorithm in
the NeurIPS 2019 paper "Causal Regularization" by Janzing
partially removes the effect of confounders (common causes) in multivariate linear
causal models.

The estimation of the regression vector a from (X,Y) samples is spoiled by the multivariate
common cause Z, but regularizing the regression (Lasso or Ridge) can improve
the estimation of the regression vector a.
The program outputs a scatter plot that plots the error of the regression vector obtained via
unregularized regression to the one obtained by one of the following methods:
(1) standard cross validated Ridge
(2) standard cross validated Lasso
(3) ConCorr with Ridge
(4) ConCorr with Lasso
"""

from __future__ import division
import estimate_confounding as est
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
print(linear_model.__file__)
from sklearn.linear_model import RidgeCV


# specify parameters
D = 30 # dimension of the cause X
L = D  # dimension of confounder Z
K = D  # sparsity controlling parameter for the causal influence, induces sparsity for K << D
N = 50  # number of samples
N_RUNS = 100  # number of runs

# specify bounds for random parameter chosen in each run
f_max = 5
a_max = 1
c_max = 1 # confounding parameter
e_max = 0 # noise of the cause


def independent_source_model(sdt_a, sdt_c, sdt_e, sdt_f):
    """
    Generate samples from (X,Y) (simulated confounding). 
    Obtained by a slight modification of the independent sources model
    described in Janzing & Schoelkopf: Detecting non-causal artifacts in
    multivariate linear regression models, ICML 2018.
    X = M*Z + E
    Y = a^T*X + c^T*Z + F
    with
    Z: is the common cause of X and Y
    E, F: independent noise terms (E and F don't exist in the ICML paper)

    Args:
        std_a: standard deviation of coefficients of causal vector a
        sdt_c: standard deviation of coefficients of vector c
        sdt_e: standard deviation of E (i.e. noise level for X)
        sdt_f: standard deviation of F (i.e. noise level for Y)

    Returns:
        list [x,y,a,conf_strength] where
        x: data matrix with format (n x d)
        y: vector (n x 1)
        a: vector of causal regression coefficients 
        conf_str: confounding strength beta as defined in Janzing & Schoelkopf 2018.

    a, M, c are drawn at random from Gaussians   

    """

    # generate regression vectors and mixing matrix
    a = np.random.normal(0, sdt_a, D)  # draw structure coefficients at random
    a[K:D] = np.zeros(D - K)  # set parts of them to zero to obtain sparse causal model
    M = np.random.normal(0, 1, (D, L))  # draw mixing matrix at random
    c = np.random.normal(0, sdt_c, L)  # draw confounding vector at random

    # compute confounding strength beta
    correlation_confound = np.dot(M, c)
    see = np.identity(D) * (sdt_e**2)  # define covariance matrix of noise E
    sxx = see + np.dot(M, M.transpose())
    conf_vector = np.dot(np.linalg.inv(sxx), correlation_confound)
    conf_str = sqlength(conf_vector) / (sqlength(a) + sqlength(conf_vector))

    # generate data
    z = np.random.normal(0, 1, (L, N))
    e = np.random.normal(0, sdt_e, (D, N))
    f = np.random.normal(0, sdt_f, (1, N))
    x = np.dot(M, z) + e
    y = np.dot(a, x) + np.dot(c, z) + f
    return [x, y, a, conf_str]

def find_alpha_for_given_sqlength_lasso(sqlen,x,y):
    """ Find the regularization parameter alpha such that the regression vector has a certain length.
    :param sqlen: desired squared length of the regression vector
    :param x: input data matrix n x d (number of samples x dimension)
    :param y: output data vector with dimension n (number of samples)
    :return: regularization parameter alpha to be used for Lasso regression
    """
    return sc.optimize.root(sqlength_diff_lasso,0,args=(sqlen,x,y)).x

def find_alpha_for_given_sqlength_ridge(sqlen,x,y):
    """ Find the regularization parameter alpha such that the regression vector has a certain length.
       :param sqlen: desired squared length of the regression vector
       :param x: input data matrix n x d (number of samples x dimension)
       :param y: output data vector with dimension n (number of samples)
       :return: regularization parameter alpha to be used for Ridge regression
       """
    return sc.optimize.root(sqlength_diff_ridge,0,args=(sqlen,x,y)).x

def sqlength_diff_lasso(alpha,sqlen,x,y):
    """Computes the difference between the desired squared length of the regression vector obtained by
    Lasso for the given regularization parameter to the desired squared length
    :param alpha: regularization parameter
    :param sqlen: desired squared length
    :param x: input data matrix n x d (number of samples x dimension)
    :param y: youtput data vector with dimension n (number of samples)
    :return: difference of squared lengths
    """
    if alpha !=0:
        clf = linear_model.Lasso(alpha)
        clf.fit(x.transpose(),y.transpose())
        return sqlength(clf.coef_) - sqlen
    else:
        reg = LinearRegression().fit(x.transpose(), y.transpose())
        return sqlength(reg.coef_) - sqlen


def sqlength_diff_ridge(alpha,sqlen,x,y):
    """Computes the difference between the desired squared length of the regression vector obtained by
        Ridge for the given regularization parameter to the desired squared length
        :param alpha: regularization parameter
        :param sqlen: desired squared length
        :param x: input data matrix n x d (number of samples x dimension)
        :param y: youtput data vector with dimension n (number of samples)
        :return: difference of squared lengths
        """
    if alpha !=0:
        clf = linear_model.Ridge(alpha)
        clf.fit(x.transpose(),y.transpose())
        return sqlength(clf.coef_) - sqlen
    else:
        reg = LinearRegression().fit(x.transpose(), y.transpose())
        return sqlength(reg.coef_) - sqlen


def sqlength(v):
    """
    Compute the squared length of a vector.

    Args: vector
    Returns: squared length
    """
    return np.sum(v**2)

def ridge(alpha,x,y):
    """
    Applies standard ridge regression in  scikit for fixed regularization parameter.
    :param alpha: regularization parameter
    :param x: input data matrix
    :param y: target data vector
    :return: vector of regression coefficients
    """
    clf =  linear_model.Ridge(alpha)
    clf.fit(x.transpose(),y.transpose())
    return clf.coef_


def lasso(alpha,x,y):
    """
    Applies standard ridge regression in  scikit for fixed regularization parameter.
    :param alpha: regularization parameter
    :param x: input data matrix
    :param y: target data vector
    :return: vector of regression coefficients
    """
    clf =  linear_model.Lasso(alpha)
    clf.fit(x.transpose(),y.transpose())
    return clf.coef_


def concorr_lasso(x,y):
    """
    Algorithm ConCorr using Lasso as described in the NeurIPS paper
    :param x: values of X, nxd matrix with sample size n and dimension d
    :param y: vector in R^n, values of Y
    :return: estimated regression vector
    """
    Cxx = np.cov(x)
    unregularized_reg_vector = unregularized_regression(x,y)
    sqlen =  (1-est.estimate_beta(Cxx, unregularized_reg_vector))*sqlength(unregularized_reg_vector)
    alpha = find_alpha_for_given_sqlength_lasso(sqlen,x,y)
    return lasso(alpha,x,y)


def concorr_ridge(x,y):
    """
    Algorithm ConCorr using Ridge as described in the NeurIPS paper
    :param x: values of X, nxd matrix with sample size n and dimension d
    :param y: vector in R^n, values of Y
    :return: estimated regression vector
    """
    Cxx = np.cov(x)
    unregularized_reg_vector = unregularized_regression(x,y)
    sqlen =  (1-est.estimate_beta(Cxx, unregularized_reg_vector))*sqlength(unregularized_reg_vector)
    alpha = find_alpha_for_given_sqlength_ridge(sqlen,x,y)
    return ridge(alpha,x,y)


def cv_ridge(x,y):
    """
    Apply cross-validated Ridge from scikit
    :param x: values of X, nxd matrix with sample size n and dimension d
    :param y: vector in R^n, values of Y
    :return: estimated regression vector
    """
    clf = RidgeCV().fit(x.transpose(), y.transpose().ravel())
    print('penalty = ' + str(clf.alpha_))
    return clf.coef_


def cv_lasso(x,y):
    """
    Apply cross-validated Lasso from scikit
    :param x: values of X, nxd matrix with sample size n and dimension d
    :param y: vector in R^n, values of Y
    :return: estimated regression vector
    """
    clf = LassoCV(eps=0.00000000001,tol=0.000001, max_iter=1000000).fit(x.transpose(), y.transpose().ravel())
    print('penalty = ' + str(clf.alpha_))
    return clf.coef_

def unregularized_regression(x,y):
    """
    Compute ordinary least square regression vector
    :param x: values of X, nxd matrix with sample size n and dimension d
    :param y: vector in R^n, values of Y
    :return: estimated regression vector
    """
    d = x.shape[0]
    return np.dot(np.linalg.inv(np.cov(x)),np.cov(x,y)[0:d,d])  


def relative_error(est_vector,true_vector):
    """ Compute relative error for one vector relative to a given "true" one
    :param est_vector: vector in R^d that is supposed to be close to the second
    :param true_vector: vector in R^d considered the ground truth
    :return: relative error (between 0 and 1)
    """
    error = sqlength(est_vector - true_vector)
    return error / (error + sqlength(true_vector))

def successes_and_failures(art_strength,rel_error):
    """ Computes fraction of successes and failures for the N_RUNS runs.
    One run is considered success if rel_error is below both simple baselines:
    First, the baseline 1/2 which can be achieved by choosing the trivial regression vector 0
    and second, the unregularized (OLS) regression vector.

    :param art_strength: vector of strengths of artifacts for all runs
    :param rel_error: vector of relative error for all runs
    :return: two numbers in [0,1] being the two fractions.
    """
    n_runs = len(art_strength)
    successes = sum(1 for j in range(n_runs) if rel_error[j] < min(art_strength[j],1/2) - 0.05 )/n_runs
    failures = sum(1 for j in range(n_runs) if rel_error[j] > min(art_strength[j],1/2) + 0.05 )/n_runs
    return [successes, failures]

def main():
    """
    Plot the relative error of the unregularized regression vector
    versus as in Figure 2 of the NeurIPS paper.
    (1) cross-validated Ridge
    (2) cross-validated Lasso
    (3) ConCorr with Ridge
    (4) Concorr with Lasso
    Print the fraction of times the chosen algorithm performs significantly
    better (see definition of successes_and_failures) and the fraction
    of times it performs worse than the unregularized baseline.
    """

    # ask whether cross-validation or concorr should be used, with Lasso or Ridge
    answer = input('ridge cv / lasso cv / ridge concorr / lasso concorr ? rcv /lcv / rcorr / lcorr  ')

    art_strength = np.zeros(N_RUNS)
    rel_error = np.zeros(N_RUNS)
    for j in range(N_RUNS):
        print(j)
        # generate random parameters
        sdt_a = np.random.uniform(0,a_max)
        sdt_c = np.random.uniform(0,c_max)
        sdt_e = np.random.uniform(0,e_max)
        sdt_f = np.random.uniform(0,f_max)
        print('theta = ' + str((sdt_c/sdt_a)**2))  # parameter related to confounding strength
        # generate data
        [x, y, a, conf] = independent_source_model(sdt_a,sdt_c, sdt_e, sdt_f)
        # compute relative error of OLS caused by overfitting and confounding
        art_strength[j] = relative_error(unregularized_regression(x,y),a)

        # compute error for one of the 4 above options
        if answer == 'rcv' :
            rel_error[j] = relative_error(cv_ridge(x,y),a)
        elif answer == 'lcv' :
            rel_error[j] = relative_error(cv_lasso(x,y),a)
        elif answer == 'rcorr' :
            rel_error[j] = relative_error(concorr_ridge(x,y),a)
        elif answer == 'lcorr' :
            rel_error[j] = relative_error(concorr_lasso(x,y),a)
      
    plt.plot(art_strength,rel_error,'bo')
    grid = [i/100 for i in range(100)]
    one_half = [1/2 for i in range(100)]

    plt.plot(grid,grid,'r.')
    plt.plot(grid,one_half,'g.')
    plt.xlabel('relative sq. error unregularized', fontsize=18)
    plt.ylabel('relative sq. error', fontsize=18)
    success_failures = successes_and_failures(art_strength,rel_error)
    print('end')
    plt.show()

    print('performance:' +  str(success_failures))

if __name__ == '__main__':
    main()
