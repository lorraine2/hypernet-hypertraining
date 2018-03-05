# !/usr/bin/python
# -*- coding: utf-8 -*-
"""Run the main body of this code to execute experiments."""
from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import sgd
from code.plotting import create_figure_and_axs, setup_ax_and_save
from code.optimizers import adam, opt_params
from code.data_loader import load_mnist
from code.neural_network import init_random_params, log_gaussian, get_loss_functions

graph_iters = 100000  # The number of iterations to do the optimization for (integer).
log_likelihoods = np.zeros(graph_iters)  # An array of the log likelihoods (test loss) for each iteration.
train_performance = np.zeros(graph_iters)  # An array of the training performance (no regularization) for each
# iteration.
valid_loss, test_loss = np.zeros(graph_iters), np.zeros(graph_iters)  # An array of the validation/test losses.
grad_norms_hypernet, grad_norms_hyper = np.zeros(graph_iters), np.zeros(graph_iters)  # An array of the hypernet
# gradient norm for each iteration.
train_hypers = np.zeros(graph_iters) # An array of the hyperparameter for each hyperparameter iteration.
global_opt_iteration, global_hyperopt_iteration = 0, 0  # A record of the current hypernet/hyperparameter iteration.
hyper_cur = None  # A record of the current hyperparameter
# NOTE - THESE HAVE GLOBAL SCOPE, SO THE CALLBACK CAN MODIFY THEM!!!


def experiment(train_data, valid_data, test_data, init_scale, batch_size, num_iters_hypernet, step_size_hypernet,
                 num_iters_hyper, step_size_hyper, num_iters, graph_mod, global_seed=0):
    """Run the second experiment, which consists of fitting a hypernetwork, which outputs neural network parameters.
    These neural network parameters try to fit the training data with some additional loss for the hyperparameters.
    We try to optimize the hyperparameters given the learned neural network response through the hypernetwork.
    We observe how the hypernetwork performs on the training and testing, by graphing it against the true loss.
    The true loss is found by training a neural network to convergence at a discrete number of points.

    :param train_data: The training data which is a tuple of (train_input, train_target).
    :param valid_data: The testing data which is a tuple of (valid_input, valid_target).
    :param test_data: The testing data which is a tuple of (test_input, test_target).
    :param init_scale: The scale (positive float) for the hypernetwork initialization.
    :param batch_size: The number of hyperparameters to sample for each iteration.
    :param num_iters_hypernet: The number of iterations (integer) to run the hypernetwork optimizer for.
    :param step_size_hypernet: The step size (positive float) for the hypernetwork optimizer.
    :param num_iters_hyper: The number of iterations (integer) to run the hyperparameter optimizer for.
    :param step_size_hyper: The step size (positive float) for the hypernetwork optimizer.
    :param num_iters: The number of iterations (integer) to run the optimization for.
    :param graph_mod: How many iterations (integer) to weight between each graph of the loss.
    :param global_seed: The seed (integer) to use when choosing a constant seed.
    :return: None.
    """
    assert init_scale > 0
    assert step_size_hypernet > 0 and step_size_hyper > 0
    assert num_iters > 0 and num_iters_hypernet > 0 and num_iters_hyper > 0
    global hyper_cur
    hyper_cur = -3.5  # Initialize the hyperparameter (float).

    # Define information about hyper loss and how hyper parameters are sampled.
    hyper_sample_var = 0  # 10e-4  # The variance to use when sampling hyperparameters from a Gaussian distribution.

    def sample_hypers(hyper, rs):
        """Sample a hyperparameter.

        :param hyper: The current hyperparameter ([float]).
        :param rs: A numpy randomstate.
        :return: A sampled hyperparameter (float).
        """
        return np.array([rs.randn() * hyper_sample_var + hyper]).reshape(1, -1)

    def hyper_loss(weights, hyper):
        """Find the loss for neural network that is dependant on the hyperparameter.

        :param weights: The weights ([[float]]) of the neural network.
        :param hyper: The hyperparameter (float) input to the hypernetwork.
        :return: The loss (float) of network dependant on the hyperparameter.
        """
        return -log_gaussian(weights, np.exp(hyper))

    example_hyper = sample_hypers(hyper_cur, npr.RandomState(global_seed))  # Test the sample function.
    assert example_hyper is not None

    train_inputs, train_targets = train_data
    valid_inputs, valid_targets = valid_data
    test_inputs, test_targets = test_data
    batch_ind, feature_ind = 0, 1
    elementary_input_size = np.shape(train_inputs)[feature_ind]
    elementary_output_size = np.shape(train_targets)[feature_ind]
    elementary_layer_sizes = [elementary_input_size, elementary_output_size]
    num_hypers = example_hyper.shape[feature_ind]  # The dimensionality of the hyperparameter space (integer).

    # Define neural network and function to turn a vector into its weight structure.
    example_elementary_params = init_random_params(init_scale, elementary_layer_sizes, npr.RandomState(global_seed))
    flat_elementary_params, unflatten_vector_to_network_weights = flatten(example_elementary_params)
    assert hyper_loss(example_elementary_params, example_hyper) is not None
    num_elementary_params = len(flat_elementary_params)

    # Define a hypernetwork parametrized by some hyperparameters.
    hypernet_layer_sizes = [num_hypers, num_elementary_params]  # Note that there are no hidden units.

    objective_functions = get_loss_functions(unflatten_vector_to_network_weights, sample_hypers, hyper_loss, batch_size,
                                             train_inputs, train_targets, test_inputs, test_targets, valid_inputs,
                                             valid_targets, global_seed)
    hypernet, train_objective, valid_objective, test_objective = objective_functions[:4]
    hyper_train_objective, hyper_valid_objective, hyper_test_objective = objective_functions[4:-1]
    hyper_train_stochastic_objective = objective_functions[-1]

    # Next, train a neural network from scratch with different hyperparameter values.
    real_step_size = 0.0001  # The step size to use to find the real loss (float).
    real_num_iters = 1000  # The number of iterations to use to find the real loss (integer).
    range_min = -2.0  # The min log variance for the hyper parameter of the variance of weight distribution to graph.
    range_max = 4.0  # The max log variance for the hyper parameter of the variance of weight distribution to graph.
    num_visual_points = 10  # The number of points to test the real loss of - expensive (integer).
    real_hyper_range = np.linspace(range_min + 1.0, range_max - 1.0, num_visual_points)
    real_train_loss = np.zeros(real_hyper_range.shape)
    real_train_performance = np.zeros(real_hyper_range.shape)
    real_valid_loss = np.zeros(real_hyper_range.shape)
    real_test_loss = np.zeros(real_hyper_range.shape)
    min_real_valid_loss, min_real_hyper = 10e32, 10e32
    for i, hypers in enumerate(real_hyper_range):
        print("Optimizing network parameters: ", i)
        init_params = init_random_params(init_scale, elementary_layer_sizes, npr.RandomState(global_seed))

        def cur_obj(w, seed):
            """The current objective function of the neural network.

            :param w: The weights ([float]) of the neural network.
            :param seed: The seed (integer) for sampling a hyperparameter.
            :return: The current objective value (float).
            """
            return train_objective(w, hypers, seed)

        optimized_params, _, _, _ = adam(grad(cur_obj), init_params, step_size=real_step_size, num_iters=real_num_iters)
        real_train_loss[i] = train_objective(optimized_params, hypers, global_seed)
        real_train_performance[i] = real_train_loss[i] - hyper_loss(optimized_params, hypers)
        real_valid_loss[i] = valid_objective(optimized_params, hypers, global_seed)
        if real_valid_loss[i] < min_real_valid_loss:
            min_real_valid_loss = real_valid_loss[i]
            print("Best hyperparameter found = ", hypers)
        real_test_loss[i] = test_objective(optimized_params, hypers, global_seed)

    fig, axs = create_figure_and_axs()

    # Set up the arrays to store information for plotting.
    num_hyper_test_points = 200  # Test a large number of hyperparameters with the learned function - cheap (integer)!
    learned_hyper_range = np.linspace(range_min, range_max, num_hyper_test_points) # Hyperparameters to test.
    hyper_train_loss = np.zeros(learned_hyper_range.shape)  # Hypernetwork training loss per hyperparameter.
    hyper_train_performance = np.zeros(learned_hyper_range.shape)  # Hypernetwork training performance per
    # hyperparameter.  Note that performance is loss - regularization loss.
    hyper_valid_loss, hyper_test_loss = np.zeros(learned_hyper_range.shape), np.zeros(learned_hyper_range.shape)

    def callback(hyper_weights, opt_iteration, g):
        """Do whatever work is desired on each optimization iteration.
        Draws graphs, prints information, and stores information.

        :param hyper_weights: The weights ([[float]]) of the hypernetwork.
        :param opt_iteration: The current iteration of optimization.
        :param g: The gradient ([[float]]) of the optimizer.
        :return: None.
        """
        global log_likelihoods, valid_loss, test_loss, grad_norms_hyper, grad_norms_hypernet, global_opt_iteration
        global hyper_cur
        log_likelihood = hyper_train_objective(hyper_weights, hyper_cur)
        log_likelihoods[global_opt_iteration] = log_likelihood  # Store the training loss.
        weights_cur = hypernet(hyper_weights, hyper_cur)
        train_performance[global_opt_iteration] = log_likelihood - hyper_loss(weights_cur, hyper_cur)
        valid_loss[global_opt_iteration] = hyper_valid_objective(hyper_weights, hyper_cur)
        test_loss[global_opt_iteration] = hyper_test_objective(hyper_weights, hyper_cur)
        grad_norm = np.sum([np.sum([np.sum(np.abs(weight_or_bias)) for weight_or_bias in layer]) for layer in g])
        grad_norms_hypernet[global_opt_iteration] = grad_norm
        grad_norms_hyper[global_opt_iteration] = grad_norms_hyper[global_opt_iteration-1]
        global_opt_iteration += 1
        print("Iteration {} Loss {} Grad L1 Norm {}".format(opt_iteration, log_likelihood, grad_norm))

        if global_opt_iteration % graph_mod == 0:  # Only print on every iteration that is a multiple of graph_mod.
            [ax.cla() for ax in axs]  # Clear all of the axes.
            axs[0].set_xlabel('Hyperparameter $\lambda$'), axs[0].set_ylabel('Loss $\mathcal{L}$')

            for cur, hyper in enumerate(learned_hyper_range):
                hyper_train_loss[cur] = hyper_train_objective(hyper_weights, hyper)
                weights = hypernet(hyper_weights, hyper)
                hyper_train_performance[cur] = hyper_train_loss[cur] - hyper_loss(weights, hyper)
                hyper_valid_loss[cur] = hyper_valid_objective(hyper_weights, hyper)
                hyper_test_loss[cur] = hyper_test_objective(hyper_weights, hyper)

            axs[0].plot(real_hyper_range, real_train_loss, 'bx', ms=28, label='Train loss of optimized weights')
            axs[0].plot(learned_hyper_range, hyper_train_loss, 'b-', label='Train loss of hypernetwork weights')
            axs[0].set_ylim([-1.5, 3.8])

            axs[0].plot(real_hyper_range, real_valid_loss, 'rx', ms=28, label='Valid. loss of optimized weights')
            axs[0].plot(learned_hyper_range, hyper_valid_loss, 'r-', label='Valid. loss of hypernetwork weights')
            min_hyper_found = 1.836  # Known minimum from doing a search with 1000 points over this range.
            axs[0].axvline(x=min_hyper_found, c='k', linestyle='dashed', label='Optimal hyperparameter $\lambda$')

            pdf_range = np.linspace(hyper_cur - 0.5, hyper_cur + 0.5, 100)
            axs[0].plot(pdf_range, norm.pdf(pdf_range, loc=hyper_cur, scale=0.06) / 4.0 + axs[0].get_ylim()[0], c='g',
                        label='$p (\lambda | \hat{\lambda})$')

            [ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45),
                       borderaxespad=0.0, fancybox=True, framealpha=0.0, fontsize=28)
             for ax in axs]  # Create a legend for all the axes.
            setup_ax_and_save(axs, fig, 'hypernets_local_small')

    def callback_outer(hyper, opt_iteration, g):
        """Do whatever work is desired on each outer optimization iteration.
        Stores information.

        :param hyper: The hyperparameter (float) input to the hypernetwork.
        :param opt_iteration: The current iteration of optimization.
        :param g: The gradient ([[float]]) of the optimizer.
        :return: None.
        """
        global grad_norms_hyper, train_hypers, global_hyperopt_iteration
        grad_norms_hyper[global_opt_iteration - 1] = np.abs(g)
        train_hypers[global_hyperopt_iteration] = hyper
        global_hyperopt_iteration += 1
        print("Outer Iteration {} Hyper {} Grad L1 Norm {}".format(global_hyperopt_iteration, hyper,
                                                                   grad_norms_hyper[global_opt_iteration]))

    init_hypernet_params = init_random_params(init_scale, hypernet_layer_sizes, npr.RandomState(global_seed))
    m_hyper = None  # A record of the current m for re-starting the Adam optimizer.
    v_hyper = None  # A record of the current v for re-starting the Adam optimizer
    cur_iter_hyper = None  # A record of the current iteration for re-starting the Adam optimizer.
    for _ in range(num_iters):
        def hyper_train_stochastic_objective_current(hyper_weights, seed):
            """The objective for the hypernetwork, with a fixed hyperparameter.

            :param hyper_weights: The weights ([[float]]) of the hypernetwork.
            :param seed: The seed (integer) for sampling a hyperparameter.
            :return: The hypernetwork's loss (float).
            """
            return hyper_train_stochastic_objective(hyper_cur, hyper_weights, seed)

        init_hypernet_params = sgd(grad(hyper_train_stochastic_objective_current), init_hypernet_params,
                                   step_size=step_size_hypernet, num_iters=num_iters_hypernet, callback=callback,
                                   mass=0)

        def valid_objective_current(hyper, seed):
            """The objective for the hyperparameter, with a fixed hypernetwork.

            :param hyper: The hyperparameter (float) input to the hypernetwork.
            :param seed: The seed (integer) for sampling a hyperparameter.
            :return: The validation loss (float).
            """
            return valid_objective(hypernet(init_hypernet_params, hyper), hyper, seed)

        hyper_cur, m_hyper, v_hyper, cur_iter_hyper = adam(grad(valid_objective_current), hyper_cur,
                                                           step_size=step_size_hyper, num_iters=num_iters_hyper,
                                                           callback=callback_outer, m=m_hyper, v=v_hyper,
                                                           offset=cur_iter_hyper)
        print("The current hyperparameter is:", hyper_cur)

if __name__ == '__main__':
    params = opt_params(graph_iters)

    _, train_images, train_labels, test_images, test_labels = load_mnist()
    n_data, n_data_val, n_data_test = params['n_data'], params['n_data_val'], params['n_data_test']
    train_data = (train_images[:n_data], train_labels[:n_data])
    valid_data = (train_images[n_data:n_data + n_data_val], train_labels[n_data:n_data + n_data_val])
    test_data = (test_images[:n_data_test], test_labels[:n_data_test])

    experiment(train_data, valid_data, test_data, params['init_scale'], params['batch_size'],
               params['num_iters_hypernet'], params['step_size_hypernet'], params['num_iters_hyper'],
               params['step_size_hyper'], params['num_iters'], params['graph_mod'], params['global_seed'])
