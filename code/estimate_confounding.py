"""
This software estimates the strength of confounding (or overfitting) in a multivariate linear regression model.
Re-implementation of the R-code provided for the ICML 2018 paper 'Detecting non-causal artifacts in multivariate
linear regression models' by Dominik Janzing and Bernhard Schoelkopf. 
Uses the maximum likelihood estimation described in eq. (10).

Given d potential causes X_1,...,X_d and a target variable Y, the vector
lsq_reg obtained by ordinary unregularized least square regression
decomposes into

lsq_reg = coeff_causal +  coeff_conf

where coeff_causal describes the causal influences of X on Y and coeff_conf is
due to the hidden common cause

"""

from __future__ import division
import numpy as np
import scipy as sc
from scipy.optimize import minimize


def density(linear_map,vector):
    """
    Computes the probability distribution of directions in R^d
    when a linear map is applied to an isotropic distribution.

    :param: linear_map: non-singular matrix of format (d x d)
            vector: vector in R^d (length irrelevant)

    :return: probability density for obtaining the direction of 'vector'
       when linear_map is applied to the uniform distribution on the unit sphere
       using eq.(8) in the ICML paper

    """
    d = vector.shape[0]  # dimension
    vector = vector/np.sqrt(sum(vector**2)) # normalize vector
    vector_in = np.dot(np.linalg.inv(linear_map),vector) # compute pre-image under linear map
    stretch_factor = np.sqrt(sum(vector_in**2))
    return 1/(np.linalg.det(linear_map)*(stretch_factor**d))


def estimate_beta(Cxx,lsq_reg):
    """Estimate confounding strength beta.
    :param Cxx: covariance matrix of causes
    :param lsq_reg: unregularized regression vector
    :return: estimated value of beta
    """
    d = len(lsq_reg) # dimension
    theta_est = estimate_theta(Cxx,lsq_reg)
    Tinv = np.matrix.trace(np.linalg.inv(Cxx))/d
    beta_est = 1/(1+1/(0.001+ Tinv *theta_est)) # eq.(11) of ICML paper
    print('estimated beta: ')
    print(beta_est)
    return beta_est

def estimate_theta(Cxx,lsq_reg):
    """Estimate parameter theta related to confounding strength.
    :param Cxx: covariance matrix of causes
    :param lsq_reg: unregularized regression vector
    :return: estimated values of beta
    """
    theta_est = sc.optimize.minimize(loglikelihood, 0, bounds = [(0,None)],args=(Cxx,lsq_reg)).x
    print('theta_est = ' + str(theta_est))
    return theta_est

def loglikelihood(theta,cov,vector):
    """Computes the loglikelihood of theta for a given covariance matrix and

    :param theta: parameter related to confounding strength
    :param cov: covariance matrix of the causes
    :param vector: unregularized regression vector
    :return: loglikelihood of the direction corresponding to the vector
    """
    d = vector.shape[0]
    matrix_squared = np.identity(d) + theta*np.linalg.inv(cov) # equation (7) in the ICML paper
    matrix = sc.linalg.sqrtm(matrix_squared)
    return - np.log(density(matrix,vector))

