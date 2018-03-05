# !/usr/bin/python
# -*- coding: utf-8 -*-
"""Code to prepare a neural network.
"""
from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.scipy.stats.norm as norm
from autograd.util import flatten
import autograd.numpy.random as npr


def init_random_params(var, layer_sizes, rs):
    """Build a list of (weights, biases) tuples, one for each layer.

    :param var: The variance (positive float) to initialize the weights with.
    :param layer_sizes: The sizes [(integer)] of layers to create.
    :param rs: A numpy random state.
    :return: An array [([float], [float])] of the initial weights.
    """
    assert var > 0
    return [(rs.randn(in_size, out_size) * var,   # weight matrix
             rs.randn(out_size) * var)            # bias vector
            for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])]


def relu(x):
    """A rectified linear unit.

    :param x: The input for the ReLU unit.
    :return: The activation of the ReLU unit.
    """
    return np.maximum(x, 0)


def identity(x):
    """A unit with no activation.

    :param x: The input for the identity unit.
    :return: The activation of the identity unit.
    """
    return x


def nn_predict(weights, inputs, final_activation):
    """Retrieve the prediction of some neural network on some inputs.

    :param weights: The parameters ([[float]]) of the neural network.
    :param inputs: The inputs ([float]) of the neural network.
    :param final_activation: A function ([float] -> [float]) to use as the activation on the final layer
    :return: The outputs ([float]) of the neural network.
    """
    outputs = None
    for W, b in weights:
        outputs = np.dot(inputs, W) + b
        inputs = relu(outputs)
    return final_activation(outputs)


def pred_loss(weights, inputs, targets, noise_var=0.1):
    """The loss of the neural networks predictions.

    :param weights: The parameters ([[float]]) of the neural network.
    :param inputs: The inputs ([float]) of the neural network.
    :param targets: The targets ([float]) of the neural network.
    :param noise_var: The variance (positive float) for the loss.
    :return: The mean of the losses for the predictions.
    """
    assert noise_var > 0
    predictions = nn_predict(weights, inputs, identity)
    return np.mean(norm.logpdf(predictions, targets, noise_var))


def log_gaussian(weights, var):
    """Find the log probability of the weights given some centered, spherical Gaussian prior.

    :param weights: The parameters ([[float]]) of the neural network.
    :param var: The variance (positive float) of the Gaussian distribution.
    :return: The log probability (float) of the weights of the neural network.
    """
    assert var > 0
    flat_params, _ = flatten(weights)
    return -var * np.linalg.norm(flat_params, 2)  # np.mean(norm.logpdf(flat_params, 0, var))


def get_loss_functions(unflatten_vector_to_network_weights, sample_hypers, hyper_loss, batch_size,
                       train_inputs, train_targets, test_inputs, test_targets, valid_inputs, valid_targets,
                       global_seed):
    """Get the loss functions for an experiment.

    :param unflatten_vector_to_network_weights:  A function to convert a weight vector to a weight tensor.
    :param sample_hypers: Sample a set of hyperparameters given a current hyperparameter.
    :param hyper_loss: The regularization term added to the training loss.
    :param batch_size: The batch size for data points.
    :param train_inputs: The training inputs.
    :param train_targets: The training targets.
    :param test_inputs: The testing inputs.
    :param test_targets: The testing targets.
    :param valid_inputs: The validation inputs.
    :param valid_targets: The validation targets.
    :param global_seed: A random seem.
    :return: The training/validation/testing loss functions for a fixed hyperparameter/data point with fixed
    hyperparameter.  Also the best-response function is returned.
    """
    def hypernet(hyper_weights, hyper):
        """A hypernetwork which takes in hyperparameters and outputs neural network weights.

        :param hyper_weights: The weights ([[float]]) of the hypernetwork.
        :param hyper: The hyperparameter (float) input to the hypernetwork.
        :return: The weights ([float)] of the neural network to use for predictions.
        """
        flat_elementary_weights = nn_predict(hyper_weights, hyper, identity)
        return unflatten_vector_to_network_weights(flat_elementary_weights.T)

    def train_objective(weights, hyper, seed):
        """The objective for training a neural network.

        :param weights: The weights ([[float]]) of the neural network.
        :param hyper: The hyperparameter (float) input to the hypernetwork.
        :param seed: The seed (integer) for sampling.
        :return: The training loss (float).
        """
        return -pred_loss(weights, train_inputs, train_targets) + hyper_loss(weights, hyper)

    def valid_objective(weights, hyper, seed):
        """The objective for validating a neural network.

        :param weights: The weights ([[float]]) of the neural network.
        :param hyper: The hyperparameter (float) input to the hypernetwork.
        :param seed: The seed (integer) for sampling a hyperparameter.
        :return: The validation loss (float).
        """
        return -pred_loss(weights, valid_inputs, valid_targets)

    def test_objective(weights, hyper, seed):
        """The objective for testing a neural network.

        :param weights: The weights ([[float]]) of the neural network.
        :param hyper: The hyperparameter (float) input to the hypernetwork.
        :param seed: The seed (integer) for sampling a hyperparameter.
        :return: The testing loss (float).
        """
        return -pred_loss(weights, test_inputs, test_targets)

    def hyper_train_objective(hyper_weights, hyper):
        """The objective to use for training the hypernetwork, which takes a hyperparameter.

        :param hyper_weights: The weights ([[float]]) of the hypernetwork.
        :param hyper: The hyperparameter (float) input to the hypernetwork.
        :return: The hypernetwork's training objective (float).
        """
        weights = hypernet(hyper_weights, hyper)
        return train_objective(weights, hyper, global_seed)

    def hyper_valid_objective(hyper_weights, hyper):
        """The objective to use for validating the hypernetwork, which takes a hyperparameter.

        :param hyper_weights: The weights ([[float]]) of the hypernetwork.
        :param hyper: The hyperparameter (float) input to the hypernetwork.
        :return: The hypernetwork's validation objective (float).
        """
        weights = hypernet(hyper_weights, hyper)
        return valid_objective(weights, hyper, global_seed)

    def hyper_test_objective(hyper_weights, hyper):
        """The objective to use for testing the hypernetwork, which takes a hyperparameter.

        :param hyper_weights: The weights ([[float]]) of the hypernetwork.
        :param hyper: The hyperparameter (float) input to the hypernetwork.
        :return: The hypernetwork's testing objective (float).
        """
        weights = hypernet(hyper_weights, hyper)
        return test_objective(weights, hyper, global_seed)

    def hyper_train_stochastic_objective(hyper, hyper_weights, seed):
        """The objective to use for training the hypernetwork, which samples a hyperparameter.

        :param hyper_weights: The weights ([[float]]) of the hypernetwork.
        :param hyper: The hyperparameter (float) input to the hypernetwork.
        :param seed: The seed (integer) for sampling a hyperparameter.
        :return: The hypernetwork's objective (float).
        """
        rs = npr.RandomState(seed)
        return np.mean([hyper_train_objective(hyper_weights, sample_hypers(hyper, rs)) for _ in range(batch_size)])

    return (hypernet, train_objective, valid_objective, test_objective, hyper_train_objective, hyper_valid_objective,
            hyper_test_objective, hyper_train_stochastic_objective)
