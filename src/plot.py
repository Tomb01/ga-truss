import matplotlib.pyplot as plt
from src.structure import Structure
import numpy as np
from math import atan2, degrees
from src.utils.misc import newline
from src.operations import upper_tri_masking

def plot_load_arrow(axes, x, y, P, x_dir = True, color = "red") -> None:
    scale = 0.2 
    d = (P and P / abs(P) or 0)*scale
    dx = d*x_dir
    dy = d*(not x_dir)
    axes.arrow(x, y, dx, dy, length_includes_head=True, head_width=0.1*scale, head_length=0.2*scale, color=color)
    axes.annotate("{:.3f} N".format(P), xy=(x+dx, y+dy), fontsize=8, xytext=((not dx)*1,(not dy)*1), textcoords='offset fontsize', color=color)
    
def plot_hinge(axes, x, y, Rx, Ry, color = "blue") -> None:
    size = 100
    axes.scatter(x, y, size, color = "black", marker=6)
    plot_load_arrow(axes, x, y, Rx, True, color=color)
    plot_load_arrow(axes, x, y, Ry, False, color=color)
    

def plot_node(axes, node: np.array, size=50, index=0, annotation = True) -> None:
    axes.scatter(node[0], node[1], size, color = "grey")
    axes.text(node[0], node[1], index, fontsize=7, va="center", ha="center", color = "white", weight='bold')
    if annotation:
        if node[8] != 0:
            plot_load_arrow(axes, node[0], node[1], node[8], True)
        if node[9] != 0:
            plot_load_arrow(axes, node[0], node[1], node[9], False)
        if node[4] and node[5]:
            # hinge
            plot_hinge(axes, node[0], node[1], node[6], node[7])

def plot_structure(
    structure: Structure, figure: plt.figure = None, axes: plt.axes = None, color=None, annotation = True, area_range = None) -> None:
    trusses = structure._trusses
    constrain = structure._n_constrain
    nodes = structure._nodes

    if axes == None:
        axes = plt.gca()

    areas = trusses[1]
    if area_range == None:
        area_range = np.max(areas) - np.min(areas[np.nonzero(areas)])
    else:
        area_range = area_range[1] # - area_range[0]
    adj = np.triu(trusses[0])
    area_ratio = np.triu(np.divide(areas, area_range, out=np.zeros_like(trusses[1]), where=(trusses[1]!=0)))
    n = len(nodes)

    space = structure._parameters.corner
    #axes.set_xlim(space.min_x, space.max_x)
    #axes.set_ylim(space.min_y, space.max_y)
    #axes.set_aspect('equal', adjustable='box')
    node_size = 80

    for i in range(0, n):
        plot_node(axes, nodes[i], node_size*1.5 if i < constrain else node_size, index=i, annotation = annotation)
        conn = adj[i]
        for j in range(0, n):
            if conn[j] == 1:
                s_x = nodes[i, 0]
                e_x = nodes[j, 0]
                s_y = nodes[i, 1]
                e_y = nodes[j, 1]
                line = (5 - 1) * area_ratio[i,j] + 1
                axes.plot([s_x, e_x], [s_y, e_y], color=color, linewidth = line, zorder=0)
                if e_x == s_x:
                    angle = 90
                else:
                    angle = degrees(atan2((e_y - s_y), (e_x - s_x)))
                if annotation:
                    text = "{a:.3f} N".format(a=1-trusses[2, i, j], b=trusses[2, i, j], c=trusses[1, i, j])
                else:
                    text = ""
                axes.text(
                    (s_x + e_x) / 2,
                    (s_y + e_y) / 2,
                    text,
                    fontsize=6,
                    rotation=angle,
                    horizontalalignment="left" if angle > 90 and angle < 270 else "right",
                    verticalalignment="bottom",
                    rotation_mode="anchor",
                    transform_rotates_text=True
                )


def savetxt(structure: Structure, file: str) -> None:
    with open(file, "w") as f:
        mass, total_mass, _ = structure.get_mass()
        v_max = np.max(structure._nodes[3])
        # General data
        newline(f, structure.get_fitness())
        newline(f, mass)
        newline(f, total_mass)
        newline(f, v_max)
        newline(f, structure.get_DOF())
        newline(f, structure.check())
        # Nodes
        newline(f, "")
        for n in structure._nodes[:, 0:-1]:
            newline(f, ";".join(str(c) for c in n))
        # Trusses (normal form)
        newline(f, "")
        newline(f, "")
        _, triu_mask = upper_tri_masking(structure.get_connections())
        for t in structure._trusses:
            truss_data = t[triu_mask]
            newline(f, ";".join(str(c) for c in truss_data))
        # diameters
        newline(f, ";".join(str(c) for c in structure.get_diameters()[triu_mask]))
        
