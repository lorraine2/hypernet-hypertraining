# !/usr/bin/python
# -*- coding: utf-8 -*-
"""Run the main body of this code to execute experiments."""
from __future__ import absolute_import
from __future__ import print_function
import os
import os.path
import autograd.numpy as np
import autograd.numpy.random as npr
import pickle
from autograd import grad
from autograd.misc.flatten import flatten
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from code.plotting import create_figure_and_axs, setup_ax_and_save, full_extent
from code.optimizers import adam, opt_params
from code.data_loader import load_mnist
from code.neural_network import init_random_params, identity, nn_predict, pred_loss


def log_gaussian(weights, var):
    """Find the log probability of the weights given some centered, spherical Gaussian prior.

    :param weights: The parameters ([[float]]) of the neural network.
    :param var: The variance (positive float) of the Gaussian distribution.
    :return: The log probability (float) of the weights of the neural network.
    """
    norm_sum = 0.0
    for i in range(10):
        norm_sum += np.linalg.norm(weights[0][0][:, i] * np.exp(var[i]), 2)
    return -norm_sum / 10.0


def experiment(train_data, valid_data, init_scale, num_iters_hypernet, step_size_hypernet, step_size, num_iters,
               batch_size_data, global_seed=0):
    """Run the second experiment, which consists of fitting a hypernetwork, which outputs neural network parameters.
    These neural network parameters try to fit the training data with some additional loss for the hyperparameters.
    We try to optimize the hyperparameters given the learned neural network response through the hypernetwork.
    We observe how the hypernetwork performs on the training and testing, by graphing it against the true loss.
    The true loss is found by training a neural network to convergence at a discrete number of points.

    :param train_data: The training data.
    :param valid_data: The testing data.
    :param init_scale: The scale (positive float) for the hypernetwork initialization.
    :param num_iters_hypernet: The number of iterations (integer) to run the hypernetwork optimizer for.
    :param step_size_hypernet: The step size (positive float) for the hypernetwork optimizer.
    :param step_size: The step size (positive float) for the loss approximation optimizer.
    :param num_iters: The number of iterations (integer) to run the optimization for.
    :param batch_size_data: The number of data points (integer) for a batch.
    :param global_seed: The seed (integer) to use when choosing a constant seed.
    :return: None, but saves pictures.
    """
    assert init_scale > 0
    assert step_size_hypernet > 0, step_size > 0
    assert num_iters > 0, num_iters_hypernet > 0

    def hyper_loss(weights, hyper):
        """Find the loss for neural network that is dependant on the hyperparameter.

        :param weights: The weights ([[float]]) of the neural network.
        :param hyper: The hyperparameter (float) input to the hypernetwork.
        :return: The loss (float) of network dependant on the hyperparameter.
        """
        return -log_gaussian(weights, hyper)

    train_inputs, train_targets = train_data
    valid_inputs, valid_target = valid_data
    batch_ind, feature_ind = 0, 1
    elementary_input_size = np.shape(train_inputs)[feature_ind]
    elementary_output_size = np.shape(train_targets)[feature_ind]
    elementary_layer_sizes = [elementary_input_size, elementary_output_size]
    num_hypers = 10  # The dimensionality of the hyperparameter space (integer).
    batch_size_elementary = 100  # The number of elementary data points to sample (i.e not hyperparameters).

    # Define neural network and function to turn a vector into its weight structure.
    example_elementary_params = init_random_params(init_scale, elementary_layer_sizes, npr.RandomState(global_seed))
    flat_elementary_params, unflatten_vector_to_network_weights = flatten(example_elementary_params)
    num_elementary_params = len(flat_elementary_params)

    rs_train = npr.RandomState(global_seed)

    def train_objective(weights, hyper, seed):
        """The objective for training a neural network.

        :param weights: The weights ([[float]]) of the neural network.
        :param hyper: The hyperparameter (float) input to the hypernetwork.
        :param seed: The seed (integer) for sampling.
        :return: The training loss (float).
        """
        idx = rs_train.randint(len(train_inputs) - batch_size_elementary)
        return -pred_loss(weights, train_inputs[idx:idx+batch_size_elementary],
                          train_targets[idx:idx+batch_size_elementary]) + hyper_loss(weights, hyper)

    def valid_objective(weights, hyper, seed):
        """The objective for validating a neural network.

        :param weights: The weights ([[float]]) of the neural network.
        :param hyper: The hyperparameter (float) input to the hypernetwork.
        :param seed: The seed (integer) for sampling a hyperparameter.
        :return: The validation loss (float).
        """
        return -pred_loss(weights, valid_inputs, valid_target)

    # TODO: Rename valid_objective to prediction loss, and move train objective into data generator block

    pickle_name = 'learn_vs_true_loss_scatter.pickle'
    if not os.path.isfile(pickle_name):  # Train a neural network from scratch with different hyperparameter values.
        real_step_size = 0.0001  # The step size to use to find the real loss (float).
        real_num_iters = 1000  # The number of iterations to use to find the real loss (integer).
        num_data = 2 ** 10 * 10
        data_inputs, data_target_params, data_target_loss = [], [], []
        for i in range(num_data):
            hyper_train = rs_train.rand(num_hypers) * 6.0 - 3.0
            print("Optimizing network parameters: ", i)
            init_params = init_random_params(init_scale, elementary_layer_sizes, npr.RandomState(global_seed))

            def cur_obj(w, seed):
                """The current objective function of the neural network.

                :param w: The weights ([float]) of the neural network.
                :param seed: The seed (integer) for sampling a hyperparameter.
                :return: The current objective value (float).
                """
                return train_objective(w, hyper_train, seed)

            optimized_params, _, _, _ = adam(grad(cur_obj), init_params, step_size=real_step_size, num_iters=real_num_iters)
            loss = valid_objective(optimized_params, hyper_train, global_seed)
            data_inputs += [hyper_train]
            flatten_opt_param, unflatten_vector_to_network_weights = flatten(optimized_params)
            data_target_params += [flatten_opt_param]
            data_target_loss += [loss]
        data_inputs = np.array(data_inputs)
        data_target_params = np.array(data_target_params)
        data_target_loss = np.array(data_target_loss)

        with open(pickle_name, 'wb') as handle:
            pickle.dump({'inputs': data_inputs, 'target_params': data_target_params, 'target_loss': data_target_loss},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(pickle_name, 'rb') as handle:
        pickle_data = pickle.load(handle)
        data_inputs = pickle_data['inputs']
        data_target_params = pickle_data['target_params']
        data_target_loss = pickle_data['target_loss']

    batch_size_sample = batch_size_data
    train_ind, valid_ind = batch_size_data, batch_size_data
    data_inputs_train, data_inputs_valid = data_inputs[:train_ind], data_inputs[valid_ind:]
    data_target_params_train, _ = data_target_params[:train_ind], data_target_params[valid_ind:]
    data_target_loss_train, data_target_loss_valid = data_target_loss[:train_ind], data_target_loss[valid_ind:]

    # New training for lambda, W, and lambda, Loss
    weight_layer_sizes = [num_hypers, num_elementary_params]
    init_weight_params = init_random_params(init_scale, weight_layer_sizes, npr.RandomState(global_seed))

    def train_weight_objective_loss(weights, seed):
        """The objective for training a neural network.

        :param weights: The weights ([[float]]) of the neural network.
        :param seed: The seed (integer) for sampling.
        :return: The training loss (float).
        """
        local_data_inputs = [rs_train.rand(num_hypers) * 6.0 - 3.0 for _ in range(batch_size_sample)]
        losses = [train_objective(unflatten_vector_to_network_weights(nn_predict(weights, np.array([local_data_input]),
                                                                                 identity)[0]),
                                  local_data_input, global_seed)
                  for local_data_input in local_data_inputs]
        return np.mean(np.array(losses))

    def callback_weights_loss(weights, opt_iteration, g):
        """A callback for optimization.

        :param weights: The hypernetwork weights
        :param opt_iteration: The optimization iteration
        :param g: The gradient.
        :return: None
        """
        if opt_iteration % 10 == 0:
            print("Sampled Valid Loss Target: ", opt_iteration, ", Loss: ", train_weight_objective_loss(weights, 0))

    weight_params_loss, _, _, _ = adam(grad(train_weight_objective_loss), init_weight_params,
                                       step_size=step_size_hypernet, num_iters=num_iters_hypernet + 100,
                                       callback=callback_weights_loss)

    init_weight_params = init_random_params(init_scale, weight_layer_sizes, npr.RandomState(global_seed))

    def train_weight_objective_loss_target(weights, seed):
        """The objective for training a neural network.

        :param weights: The weights ([[float]]) of the neural network.
        :param seed: The seed (integer) for sampling.
        :return: The training loss (float).
        """
        idx = rs_train.randint(np.maximum(len(data_inputs_train) - batch_size_data, 1))
        local_data_inputs = data_inputs_train[idx:idx + batch_size_data]
        losses = [train_objective(unflatten_vector_to_network_weights(nn_predict(weights, np.array([local_data_input]),
                                                                                 identity)[0]),
                                  local_data_input, global_seed)
                  for local_data_input in local_data_inputs]
        return np.mean(np.array(losses))

    def callback_weights_loss_target(weights, opt_iteration, g):
        """A callback for optimization.

        :param weights: The hypernetwork weights
        :param opt_iteration: The optimization iteration
        :param g: The gradient.
        :return: None
        """
        if opt_iteration % 10 == 0:
            print("Fixed Valid Loss Target: ", opt_iteration, ", Loss: ",
                  train_weight_objective_loss_target(weights, 0))

    weight_params_loss_target, _, _, _ = adam(grad(train_weight_objective_loss_target), init_weight_params,
                                              step_size=step_size_hypernet, num_iters=num_iters_hypernet,
                                              callback=callback_weights_loss_target)

    print("Preparing the data for plotting...")
    kernel = RBF()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
    gp.fit(data_inputs_train, data_target_loss_train)
    gp_loss_predictions, sigma = gp.predict(data_inputs_valid, return_std=True)

    def hypernet_loss(weights, local_data_input):
        """Find the loss for the hypernetwork.

        :param weights: The hypernet weights
        :param local_data_input: A hyperparameter.
        :return: None
        """
        weight_predictions_valid = nn_predict(weights, [local_data_input], identity)
        weight_predictions_valid = unflatten_vector_to_network_weights(weight_predictions_valid[0])
        return valid_objective(weight_predictions_valid, None, global_seed)

    loss_weight_predictions_loss = [hypernet_loss(weight_params_loss, data_input)
                                    for data_input in data_inputs_valid]
    loss_weight_predictions_loss_target = [hypernet_loss(weight_params_loss_target, data_input)
                                           for data_input in data_inputs_valid]

    fig, axs = create_figure_and_axs(fig_width=21, fig_height=7, num_cols=3, ms_size=34)

    print("Drawing the scatter plot...")
    min_v, max_v = 0.6, 1.1
    axs[0].hexbin(data_target_loss_valid, gp_loss_predictions, extent=[min_v, max_v, min_v, max_v], cmap='Reds',
                  mincnt=1)
    axs[1].hexbin(data_target_loss_valid, loss_weight_predictions_loss_target, extent=[min_v, max_v, min_v, max_v],
                  cmap='Greens', mincnt=1)
    axs[2].hexbin(data_target_loss_valid, loss_weight_predictions_loss, extent=[min_v, max_v, min_v, max_v],
                  cmap='Blues', mincnt=1)

    print("____________________________________________________________________________")
    print("Number of train data points: ", batch_size_data)
    print("GP Predicted Best: ", np.min(gp_loss_predictions), ", Actual Result: ",
          data_target_loss_valid[np.argmin(gp_loss_predictions)])
    print("Fixed Hypernet Predicted Best: ", np.min(loss_weight_predictions_loss_target),
          ", Actual Result: ", data_target_loss_valid[np.argmin(loss_weight_predictions_loss_target)])
    print("Stochastic Hypernet Predicted Best: ", np.min(loss_weight_predictions_loss),
          ", Actual Result: ", data_target_loss_valid[np.argmin(loss_weight_predictions_loss)])
    print("Actual Best: ", np.min(data_target_loss_valid))
    print("____________________________________________________________________________")

    orient_line = np.linspace(min_v, max_v, 100)
    for ax in axs:
        ax.plot(orient_line, orient_line, color='k')
        ax.set_xlim([min_v, max_v])
        ax.set_ylim([min_v, max_v])

    # axs[0].set_title('GP Mean')
    # axs[1].set_title('Hyper-train fixed')
    # axs[2].set_title('Hyper-train')

    axs[0].set_ylabel('Inferred Loss')

    #axs[1].set_xlabel('True loss')

    axs[1].set_yticks([])
    axs[2].set_yticks([])

    axs[0].set_xticks([.7, .8, .9, 1.0])
    axs[1].set_xticks([.7, .8, .9, 1.0])
    axs[2].set_xticks([.7, .8, .9, 1.0])
    axs[0].set_yticks([.7, .8, .9, 1.0])
    setup_ax_and_save(axs, fig, 'learn_vs_true_loss_scatter', do_xticks=False, do_yticks=False, y_mod=750.0, dpi=300)
    for key, ax in enumerate(axs):
        #if key is 0:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig('figures/ax' + str(key) + '_scatter.png', bbox_inches=extent.expanded(1.32, 1.15))
        fig.savefig('figures/ax' + str(key) + '_scatter.pdf', bbox_inches=extent.expanded(1.32, 1.15))
        #else:
        #extent = full_extent(ax, do_yticks=False).transformed(fig.dpi_scale_trans.inverted())
        #fig.savefig('figures/ax' + str(key) + '_scatter.png', bbox_inches=extent.expanded(1.0, 1.15))

    print("Drawing the histograms...")
    [ax.cla() for ax in axs]

    bins = 50
    axs[0].hist(gp_loss_predictions - data_target_loss_valid, bins=bins, color='r', normed=True, edgecolor='r')
    axs[1].hist(loss_weight_predictions_loss_target - data_target_loss_valid, bins=bins, color='g', normed=True,
                edgecolor='g')
    axs[2].hist(loss_weight_predictions_loss - data_target_loss_valid, bins=bins, color='b', normed=True, edgecolor='b')

    axs[0].set_ylabel('Frequency')
    axs[1].set_xlabel('Inferred - true loss')

    y_min, y_max = 10e32, -10e32
    for ax in axs:
        ylim = ax.get_ylim()
        y_min, y_max = np.minimum(y_min, ylim[0]), np.maximum(y_max, ylim[1])
    x_min, x_max = -0.35, 0.6
    for ax in axs:
        ax.set_xlim([x_min, x_max]), ax.set_ylim([y_min, y_max])
        ax.axvline(0, ymax=1.0, linestyle='--', color='Black')

    setup_ax_and_save(axs, fig, 'learn_vs_true_loss_hist', do_xticks=False)
    for key, ax in enumerate(axs):
        extent = full_extent(ax).transformed(fig.dpi_scale_trans.inverted())
        if key is 0:
            fig.savefig('figures/ax' + str(key) + '_hist.png', bbox_inches=extent) #.expand(1.32, 1.15))
            fig.savefig('figures/ax' + str(key) + '_hist.pdf', bbox_inches=extent)
        else:
            fig.savefig('figures/ax' + str(key) + '_hist.png', bbox_inches=extent)
            fig.savefig('figures/ax' + str(key) + '_hist.pdf', bbox_inches=extent)


if __name__ == '__main__':
    params = opt_params(None)

    n_data, n_data_val, n_data_test = 50000, params['n_data_val'], params['n_data_test']
    _, train_images, train_labels, test_images, test_labels = load_mnist()
    train_data = (train_images[:n_data], train_labels[:n_data])
    valid_data = (train_images[n_data:n_data + n_data_val], train_labels[n_data:n_data + n_data_val])
    test_data = (test_images[:n_data_test], test_labels[:n_data_test])

    # Define information about the optimization procedure and networks.
    init_scale = 0.00001  # The scale (positive float) for the hypernetwork initialization.
    num_iters = 5000  # The number of iterations to do the optimization for (integer).
    step_size = 0.0001  # The step size for the hyperparameter optimizer (positive float).
    num_iters_hypernet = 500  # The number of iterations to optimize the hypernetwork for (integer).

    batch_size_data = 25  # [10, 25, 100, 250, 1000, 25]
    experiment(train_data, valid_data, init_scale, num_iters_hypernet, params['step_size_hypernet'], step_size,
               num_iters, batch_size_data, params['global_seed'])
