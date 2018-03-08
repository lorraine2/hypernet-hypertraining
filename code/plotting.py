# !/usr/bin/python
# -*- coding: utf-8 -*-
"""Plotting utility functions."""
import matplotlib
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt


def create_figure_and_axs(fig_width=10, fig_height=10, num_rows=1, num_cols=1, ms_size=28):
    """Create the figure and axes for experiments

    :param fig_width: The figure width (int > 0).
    :param fig_height: The figure height (int > 0).
    :param num_rows: The number of rows in the figure (int > 0).
    :param num_cols: The number of columns in the figure (int > 0).
    :param ms_size: The text size (int > 0).
    :return: The figure and axes.
    """
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')

    plt.show(block=False)

    font = {'family': 'Times New Roman', 'size': ms_size}
    matplotlib.rc('font', **font)
    axs = [fig.add_subplot(num_rows, num_cols, ind + 1, frameon=False) for ind in range(num_rows * num_cols)]
    return fig, axs


def setup_ax_and_save(axs, fig, title, do_xticks=True, do_yticks=True, y_mod=1500.0, dpi=None):
    """Setup axes for the paper and save a figure.

    :param fig: A matplotlib figure.
    :param axs: An array of matplotlib axes.
    :param title: The image title in saving (str).
    :param do_xticks: True if we should remove the xticks (bool).
    :param do_yticks: True if we should remove the yticks (bool).
    :param y_mod: Inverse of scale factor for shifting axes.
    :param dpi: The dpi of the final saved image.
    :return: None, but saves a figure.
    """
    for ax in axs:
        if do_xticks:
            ax.set_xticklabels([])
        if do_yticks:
            ax.set_yticklabels([])
        ax.tick_params(axis='x', which='both', bottom='off', top='off')
        ax.tick_params(axis='y', which='both', left='off', right='off')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.axvline(xlim[0] + (xlim[1] - xlim[0]) / 2000.0, color='Black')
        ax.axhline(ylim[0] + (ylim[1] - ylim[0]) / y_mod, color='Black')

        plt.draw()
        if dpi is not None:
            fig.savefig('figures/' + title + '.pdf', bbox_inches='tight', dpi=dpi)
            fig.savefig('figures/' + title + '.png', bbox_inches='tight', dpi=dpi)
        else:
            fig.savefig('figures/' + title + '.pdf', bbox_inches='tight')
            fig.savefig('figures/' + title + '.png', bbox_inches='tight')
        plt.pause(1.0 / 60.0)


def full_extent(ax, pad=0.0, do_xticks=True, do_yticks=True):
    """Get the full extent of an axes, including axes labels, tick labels, and titles.

    :param ax: A matplotlib axis
    :param pad: The amount of padding to apply.
    :param do_xticks: True if we should include the xticks.
    :param do_yticks: True if we should include the yticks.
    :return: The full extent of ax
    """
    # ax.figure.canvas.draw()
    items = [ax]
    if do_xticks:
        items += ax.get_xticklabels()
    if do_yticks:
        items += ax.get_yticklabels()
    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)
