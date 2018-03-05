# !/usr/bin/python
# -*- coding: utf-8 -*-
"""Code for network optimizers.
"""
import autograd.numpy as np
from autograd.misc.flatten import flatten_func


def adam(grad, init_params, callback=None, num_iters=100, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8, m=None,
         v=None, offset=None):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms.

    :param grad: The gradient function.
    :param init_params: The initial parameters.
    :param callback: A callback function to run each iteration.
    :param num_iters: The number of iterations to run for.
    :param step_size: The step_size
    :param b1: Exponential decay rate of first moment.
    :param b2: Exponential decay rate of second moment.
    :param eps: Small term added for stability.
    :param m: The current first moment.
    :param v: The current second moment.
    :param offset: What iteration number to start with
    :return:
    """
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    if m is None:
        m = np.zeros(len(x))
    if v is None:
        v = np.zeros(len(x))
    if offset is None:
        offset = 0
    for i in range(num_iters):
        cur_iter = i + offset
        g = flattened_grad(x, cur_iter)
        if callback:
            callback(unflatten(x), cur_iter, unflatten(g))
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(cur_iter + 1))    # Bias correction.
        vhat = v / (1 - b2**(cur_iter + 1))
        x -= step_size*mhat/(np.sqrt(vhat) + eps)
    return unflatten(x), m, v, cur_iter


def opt_params(graph_iters):
    """The optimizer parameters for the first set of experiments

    :param graph_iters: The number of iterations we can potentially graph for allocating memory (int > 0).
    :return: A dictionary of the parameters.
    """
    data = {
        'global_seed': 7,  # A global seed to use (integer).

        # Define data set for training and testing.
        'n_data': 10,  # The number of data points for the training data set (integer).
        'n_data_val': 10000,  # The number of data points for the validation data set (integer).
        'n_data_test': 10000,  # The number of data points for the testing data set (integer).

        # Define information about the optimization procedure and networks.
        'init_scale': 0.01,  # The scale (positive float) for the hypernetwork initialization.
        'batch_size': 10,  # The number of hyperparameters to sample per batch (integer).
        'num_iters': graph_iters,  # The number of iterations to do the optimization for (integer).
        'num_iters_hypernet': 1,  # The number of iterations to optimize the hypernetwork for (integer).
        'num_iters_hyper': 1,  # The number of iterations to optimize the hyperparameter for (integer).
        'step_size_hypernet': 0.0001,  # The step size for the hypernetwork optimizer (positive float).
        'step_size_hyper': 0.01,  # The step size for the hyperparameter optimizer (positive float).

        'graph_mod': 100  # How many iterations to wait between each graph of the loss (integer).
    }
    return data
