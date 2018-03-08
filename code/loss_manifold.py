# !/usr/bin/python
# -*- coding: utf-8 -*-
"""Run the main body of this code to generate the loss manifold."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
from mpl_toolkits.mplot3d import proj3d


def orthogonal_proj(zfront, zback):
    """Find an orthogonal projection.

    :param zfront:
    :param zback:
    :return:
    """
    a = (zfront + zback) / (zfront - zback)
    b = -2 * (zfront * zback) / (zfront - zback)
    # -0.0001 added for numerical stability as suggested in: http://stackoverflow.com/questions/23840756
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, a, b], [0, 0, -0.0001, zback]])


def train_loss_fun(args):
    """The loss function of two variable to do a 3d plot of.

    :param args: The arguments for the loss is an array of [x, y] which are the arguments to the loss.
    :return: The loss of the arguments.
    """
    x, y = args[0], args[1]
    return -np.logaddexp(-1, -(x - np.sin(y + np.pi)) ** 2) - (y + np.sin(y * 2.0)) / 10.0


def best_response(y):
    """The best-response of the weights to some hyperparameter.

    :param y: The hyperparameter to evaluate the best-response at.
    :return: The weights that are best-responding to the hyperparameter y.
    """
    return np.sin(y + np.pi)


def linearized_best_response(y):
    """A linearization of the best-response of the weights to some hyperparameter at some point.

    :param y: The hyperparameter to evaluate the linearization at.
    :return: The linearized best-response.
    """
    return -1.0*y + 0.0


def setup_fig_and_ax(ms_size):
    """Setup the figure.

    :param ms_size: The size of the font.
    :return:
    """
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    plt.show(block=False)
    font = {'family': 'Times New Roman', 'size': ms_size}
    matplotlib.rc('font', **font)
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax


def plot_train_loss_manifold(ms_size):
    """Plot the training loss manifold.

    :param ms_size:  The text size.
    :return:  None, but an image is saved.
    """
    fig, ax = setup_fig_and_ax(ms_size)

    x_cur, y_cur = 0.0, 0.0  # The current (W*(lambda), lambda) tuple.
    z_cur = train_loss_fun([x_cur, y_cur])  # The loss of the current (W*(lambda), lambda)
    x_min, x_max, x_granularity = -2.0, 2.0, 0.04  # Graphing range and number of points for W.
    y_min, y_max, y_granularity = -np.pi, np.pi, 0.04 * (2 * np.pi) / (x_max - x_min)  # Graphing range and number of
    # points for lambda.
    z_min, z_max, z_granularity = -0.8, 1.2, 0.04

    # Show the loss surface.
    xs, ys = np.arange(x_min, x_max, x_granularity), np.arange(y_min, y_max, y_granularity)
    X, Y = np.meshgrid(xs, ys)
    zs = np.array([train_loss_fun(np.concatenate(([x], [y]))) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, color='Black')
    proj3d.persp_transformation = orthogonal_proj

    # Show the best-response function.
    xs_best_response = [best_response(y) for y in ys]
    args_best_response = [[xs_best_response[i], ys[i]] for i in range(len(xs_best_response))]
    zs_best_response = [train_loss_fun(arg) for arg in args_best_response]
    ax.plot(xs_best_response, ys, zs_best_response, '-', color='Blue', linewidth=2,
            label='$\mathcal{L} (\mathrm{w}^{*} ( \lambda ), \lambda)$')

    # Show the projected linear loss.
    xs_best_response_projected = np.zeros(len(ys)) + x_min
    ax.plot(xs_best_response_projected, ys, zs_best_response, '-', color='Blue', linewidth=2)

    # Show the linearized best-response function.
    ys_linear = np.arange(y_min / 1.61, y_max / 1.65, y_granularity)
    xs_linear = [linearized_best_response(y) for y in ys_linear]
    args_linear = [[xs_linear[i], ys_linear[i]] for i in range(len(xs_linear))]
    zs_linear = [train_loss_fun(arg) for arg in args_linear]
    ax.plot(xs_linear, ys_linear, zs_linear, '-', color='Red', linewidth=2,
            label='$\mathcal{L} ( \mathrm{w}_{\phi^{*}} ( \lambda ), \lambda )$')

    # Show the projected linear loss.
    xs_linear_projected = np.zeros(len(ys_linear)) + x_min
    ax.plot(xs_linear_projected, ys_linear, zs_linear, '-', color='Red', linewidth=2)

    # Show the current hyperparameter in optimization.
    y_curs, z_curs = np.zeros(len(xs)) + y_cur, np.zeros(len(xs)) + z_cur
    ax.plot(xs, y_curs, z_curs, '--', color='Green', linewidth=2, label='$\hat{\lambda}$')
    z_curs = np.arange(train_loss_fun([x_cur, y_cur]), z_max, z_granularity)
    x_curs, y_curs = np.zeros(len(z_curs)) + x_min, np.zeros(len(z_curs)) + y_cur
    ax.plot(x_curs, y_curs, z_curs, '--', color='Green', linewidth=2)
    ax.plot([x_cur], [y_cur], [z_cur], '*', color='Green', markersize=15,
            label="$\mathrm{w}^{*} ( \hat{\lambda}), \mathrm{w}_{\phi^{*}} ( \hat{\lambda})$")

    # Set up the axes and legend.
    ax.set_xlabel('Parameter $\mathrm{w}$'), ax.set_ylabel('Hyperparameter $\lambda$')
    ax.set_zlabel('Loss $\mathcal{L}_{\mathrm{Train}} (\mathrm{w}, \lambda)$', rotation=90)
    ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
    ax.set_xlim([x_min, x_max]), ax.set_ylim([y_min, y_max])
    ax.view_init(elev=46, azim=120)
    pane_color = (1.0, 1.0, 1.0, 1.0)
    ax.w_xaxis.set_pane_color(pane_color), ax.w_yaxis.set_pane_color(pane_color), ax.w_zaxis.set_pane_color(pane_color)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.grid(False)
    fig.tight_layout()
    ax.legend(numpoints=1, fancybox=True, bbox_to_anchor=(0.1, 0.75), borderaxespad=0.0, framealpha=0.0)

    # Save and display the figure.
    fig.savefig('figures/train_loss_manifold.pdf', bbox_inches='tight')
    fig.savefig('figures/train_loss_manifold.png', bbox_inches='tight')
    plt.show()


def valid_loss_fun(args):
    """The loss function of two variable to do a 3d plot of.

    :param args: The arguments for the loss is an array of [x, y] which are the arguments to the loss.
    :return: The loss of the arguments.
    """
    x, y = args[0], args[1]
    return np.sin(x)


def plot_valid_loss_manifold(ms_size):
    """Plot the validation loss manifold.

    :param ms_size:  The text size.
    :return:  None, but an image is saved.
    """
    fig, ax = setup_fig_and_ax(ms_size)

    x_min, x_max, x_granularity = -2.0, 2.0, 0.04  # Graphing range and number of points for W.
    y_min, y_max, y_granularity = -np.pi, np.pi, 0.04 * (2 * np.pi) / (x_max - x_min)  # Graphing range and number of
    # points for lambda.
    z_min, z_max, z_granularity = -1.0, 1.0, 0.02

    # Show the loss surface.
    xs, ys = np.arange(x_min, x_max, x_granularity), np.arange(y_min, y_max, y_granularity)
    X, Y = np.meshgrid(xs, ys)
    zs = np.array([valid_loss_fun(np.concatenate(([x], [y]))) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, color='Black')
    proj3d.persp_transformation = orthogonal_proj

    # Show the best-response function.
    xs_best_response = [best_response(y) for y in ys]
    args_best_response = [[xs_best_response[i], ys[i]] for i in range(len(xs_best_response))]
    zs_best_response = [valid_loss_fun(arg) for arg in args_best_response]
    min_hyper_best_response = ys[np.argmin(zs_best_response)]
    min_weight_best_response= xs_best_response[np.argmin(zs_best_response)]
    min_loss_best_response = zs_best_response[np.argmin(zs_best_response)]
    ax.plot(xs_best_response, ys, zs_best_response, '-', color='Blue', linewidth=2)

    # Show the linearized best-response function.
    ys_linear = np.arange(y_min / 1.6, y_max / 1.55, y_granularity)
    xs_linear = [linearized_best_response(y) for y in ys_linear]
    args_linear = [[xs_linear[i], ys_linear[i]] for i in range(len(xs_linear))]
    zs_linear = [valid_loss_fun(arg) for arg in args_linear]
    min_hyper_linear = ys_linear[np.argmin(zs_linear)]
    min_weight_linear = xs_linear[np.argmin(zs_linear)]
    min_loss_linear= zs_linear[np.argmin(zs_linear)]
    ax.plot(xs_linear, ys_linear, zs_linear, '-', color='Red', linewidth=2)

    # Show the projected loss.
    xs_best_response_projected = np.zeros(len(ys)) + x_min
    x_shift = 1.0
    ax.plot(xs_best_response_projected - x_shift, ys, zs_best_response, '-', color='Blue', linewidth=2)

    # Show the minimum of the best-response.
    ys_best_response_hyper = np.zeros(len(xs)) + min_hyper_best_response
    zs_best_response_hyper = np.zeros(len(xs)) + min_loss_best_response
    ax.plot(xs - x_shift, ys_best_response_hyper, zs_best_response_hyper, '--', color='Blue', linewidth=2,
            label='$\lambda^{*}$')
    z_curs = np.arange(z_min, z_max, z_granularity) + 0.1
    x_curs = np.zeros(len(z_curs)) + x_min - x_shift
    y_curs = np.zeros(len(z_curs)) + min_hyper_best_response
    ax.plot(x_curs, y_curs, z_curs, '--', color='Blue', linewidth=2)
    ax.plot([min_weight_best_response], [min_hyper_best_response], [min_loss_best_response], '*', color='Blue',
            label="$\mathrm{w}^{*} ( \lambda^{*})$", markersize=15)

    # Show the projected linear loss.
    xs_linear_projected = np.zeros(len(ys_linear)) + x_min
    ax.plot(xs_linear_projected - x_shift, ys_linear, zs_linear, '-', color='Red', linewidth=2)

    # Show the minimum of the linearized best-response.
    ys_linear_hyper = np.zeros(len(xs)) + min_hyper_linear
    zs_linear_hyper = np.zeros(len(xs)) + min_loss_linear
    ax.plot(xs - x_shift, ys_linear_hyper, zs_linear_hyper, '--', color='Red', linewidth=2,
            label='$\lambda_{\phi^{*}}$')
    z_curs = np.arange(z_min, z_max, z_granularity)
    x_curs = np.zeros(len(z_curs)) + x_min - x_shift
    y_curs = np.zeros(len(z_curs)) + min_hyper_linear
    ax.plot(x_curs, y_curs, z_curs, '--', color='Red', linewidth=2)
    ax.plot([min_weight_linear], [min_hyper_linear], [min_loss_linear], '*', color='Red', markersize=15,
            label="$\mathrm{w}_{\phi^{*}} ( \lambda_{\phi^{*}} )$")

    # Set up the axes and legend.
    ax.set_xlabel('Parameter $\mathrm{w}$'), ax.set_ylabel('Hyperparameter $\lambda$')
    ax.set_zlabel('Loss $\mathcal{L}_{\mathrm{Valid.}} (\mathrm{w})$', rotation=90)
    ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
    ax.set_xlim([x_min - x_shift, x_max]), ax.set_ylim([y_min, y_max])
    ax.view_init(elev=46, azim=120)
    pane_color = (1.0, 1.0, 1.0, 1.0)
    ax.w_xaxis.set_pane_color(pane_color), ax.w_yaxis.set_pane_color(pane_color), ax.w_zaxis.set_pane_color(pane_color)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.grid(False)
    fig.tight_layout()
    ax.legend(numpoints=1, fancybox=True, bbox_to_anchor=(1.45, 0.75), borderaxespad=0.0, framealpha=0.0)

    # Save and display the figure.
    fig.savefig('figures/valid_loss_manifold.pdf', bbox_inches='tight')
    fig.savefig('figures/valid_loss_manifold.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    cur_ms_size = 32
    print("Plotting train loss manifold...")
    plot_train_loss_manifold(cur_ms_size)
    print("Plotting validation loss manifold...")
    plot_valid_loss_manifold(cur_ms_size)
