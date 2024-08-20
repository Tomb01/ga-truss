import matplotlib.pyplot as plt
from src.structure import Structure
import numpy as np
from math import atan2, degrees


def plot_node(plot, node: np.array, size=5, index=0) -> None:
    plot.scatter(node[0], node[1], size, c="r")
    plot.text(node[0], node[1], index, fontsize=8, horizontalalignment="right")


def plot_structure(
    structure: Structure, figure: plt.figure = None, axes: plt.axes = None, color="blue"
) -> None:
    trusses = structure._trusses
    constrain = structure._n_constrain
    nodes = structure._nodes

    if axes == None:
        axes = plt.gca()

    axes.grid()
    adj = np.triu(trusses[0])
    n = len(nodes)

    space = structure._parameters.corner
    axes.set_xlim(space.min_x, space.max_x)
    axes.set_ylim(space.min_y, space.max_y)

    for i in range(0, n):
        plot_node(axes, nodes[i], 20 if i < constrain else 5, index=i)
        conn = adj[i]
        for j in range(0, n):
            if conn[j] == 1:
                s_x = nodes[i, 0]
                e_x = nodes[j, 0]
                s_y = nodes[i, 1]
                e_y = nodes[j, 1]
                axes.plot([s_x, e_x], [s_y, e_y], color=color)
                if e_x == s_x:
                    angle = 90
                else:
                    angle = degrees(atan2((e_y - s_y), (e_x - s_x)))
                axes.text(
                    (s_x + e_x) / 2,
                    (s_y + e_y) / 2,
                    "{a:.3f}".format(a=1-trusses[5, i, j], b=trusses[2, i, j], c=trusses[1, i, j]),
                    fontsize=6,
                    rotation=angle,
                    horizontalalignment="left" if angle > 90 and angle < 270 else "right",
                    verticalalignment="bottom",
                    rotation_mode="anchor",
                    transform_rotates_text=True
                )


def show():
    plt.show()
