import matplotlib.pyplot as plt
from src.structure import Structure
import numpy as np
from math import atan2, degrees

def plot_load_arrow(axes, x, y, P, x_dir = True, color = "red") -> None:
    scale = 0.2 
    d = (P and P / abs(P) or 0)*scale
    dx = d*x_dir
    dy = d*(not x_dir)
    axes.arrow(x, y, dx, dy, length_includes_head=True, head_width=0.3*scale, head_length=0.2*scale, color=color)
    axes.annotate("{:.3f} N".format(P), xy=(x+dx, y+dy), fontsize=8, xytext=((not dx)*1,(not dy)*1), textcoords='offset fontsize', color=color)
    
def plot_hinge(axes, x, y, Rx, Ry, color = "blue") -> None:
    size = 100
    axes.scatter(x, y, size, color = "black", marker=6)
    plot_load_arrow(axes, x, y, Rx, True, color=color)
    plot_load_arrow(axes, x, y, Ry, False, color=color)
    

def plot_node(axes, node: np.array, size=5, index=0) -> None:
    axes.scatter(node[0], node[1], size, color = "grey")
    axes.annotate(index, xy=(node[0], node[1]), fontsize=10, xytext=(1,-1.2), textcoords='offset fontsize')
    if node[8] != 0:
        plot_load_arrow(axes, node[0], node[1], node[8], True)
    if node[9] != 0:
        plot_load_arrow(axes, node[0], node[1], node[9], False)
    if node[4] and node[5]:
        # hinge
        plot_hinge(axes, node[0], node[1], node[6], node[7])

def plot_structure(
    structure: Structure, figure: plt.figure = None, axes: plt.axes = None, color="grey"
) -> None:
    trusses = structure._trusses
    constrain = structure._n_constrain
    nodes = structure._nodes

    if axes == None:
        axes = plt.gca()

    #axes.grid()
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
                    "{a:.3f} N".format(a=1-trusses[2, i, j], b=trusses[2, i, j], c=trusses[1, i, j]),
                    fontsize=6,
                    rotation=angle,
                    horizontalalignment="left" if angle > 90 and angle < 270 else "right",
                    verticalalignment="bottom",
                    rotation_mode="anchor",
                    transform_rotates_text=True
                )


def show():
    plt.show()
