import numpy as np
import matplotlib.pyplot as plt
import argparse
from SOAPUtils import CentralSOAP

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


def get_data(input):
    x_init, y_init = map(np.load, input[:2])
    mask = (y_init > 0.05) & (y_init < 14.9)
    return x_init[mask].astype(np.float32), y_init[mask].astype(np.float32)


def display_result(res_skopt):
    pass



def OptimizeGPR(training_files,test_files,n_calls):

    ### hyperparameter space for SOAP descriptor generator
    ### not optimizing for n_max and l_max as the higher they are lower the errors
    space  = [
        Real(7.0, 10.0, name='r_cut'), ### cutoff for structure environment
        Real(0.1, 1.2, name='sigma'),  ### width of the gaussians for atoms
        ### 1 is linear regression with SOAP, 2 includes squares of SOAP
        ### so 2nd order polynomial regression with no cross terms, 3 is 3rd order
        Integer(1,3, name='poly_order')
    ]

    @use_named_args(space)
    def objective(**params):
        r_cut = params['r_cut']
        n_max = 10
        l_max = 10
        sigma = params['sigma']
        poly_order = params['poly_order']

        ### points are inputs, energies are targets
        pts_12_train, en_train = get_data(training_files)
        pts_12_test, en_test = get_data(test_files)

        soap_hyperparameters = [r_cut,n_max,l_max,sigma,poly_order]
        bp = CentralSOAP(pts_12_train,soap_hyperparameters)
        bp.GaussianProcessRegression(en_train)
        mae = bp.infer_GPR(pts_12_test,en_test)

        return mae

    ## Bayesian optimization
    result = gp_minimize(
        func=objective,
        dimensions=space,
        acq_func="EI",       # the acquisition function
        n_calls=n_calls,
        n_initial_points=n_calls//10 + 1, # initial random evaluations
        random_state=42
    )

    return result


def OptimizeLinear(training_files,test_files,n_calls):
    space  = [
        Real(7.0, 10.0, name='r_cut'), ### cutoff for structure environment
        Real(0.1, 1.2, name='sigma'),  ### width of the gaussians for atoms
        ### 1 is linear regression with SOAP, 2 includes squares of SOAP
        ### so 2nd order polynomial regression with no cross terms, 3 is 3rd order
        Integer(1,3, name='poly_order'),
        Real(1e-3,1e-1, name='regularizer') ### lambda for linear ridge regression
    ]

    @use_named_args(space)
    def objective(**params):
        r_cut = params['r_cut']
        n_max = 10
        l_max = 10
        sigma = params['sigma']
        poly_order = params['poly_order']

        ### points are inputs, energies are targets
        pts_12_train, en_train = get_data(training_files)
        pts_12_test, en_test = get_data(test_files)

        soap_hyperparameters = [r_cut,n_max,l_max,sigma,poly_order]
        bp = CentralSOAP(pts_12_train,soap_hyperparameters)
        bp.LinearRegression(en_train,params['regularizer'])
        mae = bp.infer_LR(pts_12_test,en_test)

        return mae

    ## Bayesian optimization
    result = gp_minimize(
        func=objective,
        dimensions=space,
        acq_func="EI",       # the acquisition function
        n_calls=n_calls,
        n_initial_points=n_calls//10 + 1, # initial random evaluations
        random_state=42
    )

    return result


def OptimizeKernel(training_files,test_files,n_calls):
    ### hyperparameter space for SOAP descriptor generator
    ### not optimizing for n_max and l_max as the higher they are lower the errors

    space  = [
        ### SOAP Hyperparameters ###
        Real(7.0, 10.0, name='r_cut'), ### cutoff for structure environment
        Real(0.1, 1.2, name='sigma'),  ### width of the gaussians for atoms
        ### Regression Hyperparameters ###
        Real(1e-5, 1e-3, name='regularizer') ### lambda for kernel ridge regression
    ]

    @use_named_args(space)
    def objective(**params):
        r_cut = params['r_cut']
        n_max = 10
        l_max = 10
        sigma = params['sigma']
        regularizer = params['regularizer']
        poly_order = 1

        ### points are inputs, energies are targets
        pts_12_train, en_train = get_data(training_files)
        pts_12_test, en_test = get_data(test_files)

        soap_hyperparameters = [r_cut,n_max,l_max,sigma,poly_order]
        bp = CentralSOAP(pts_12_train,soap_hyperparameters)
        bp.KernelRidgeRegression(en_train,regularizer)
        mae = bp.infer_KRR(pts_12_test,en_test)

        return mae


    ## Bayesian optimization
    result = gp_minimize(
        func=objective,
        dimensions=space,
        acq_func="EI",       # the acquisition function
        n_calls=n_calls,
        n_initial_points=n_calls//10 + 1, # initial random evaluations
        random_state=42
    )


    return result
