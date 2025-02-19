import matplotlib.pyplot as plt
from src.structure import Structure
import numpy as np
from math import atan2, degrees
from src.misc import newline
from src.operations import upper_tri_masking

## Set of varius output and graphics function

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

def plot(
    structure: Structure, figure: plt.figure = None, axes: plt.axes = None, color=None, annotation = True, area_range = None) -> None:
    """
    Plot a structure with matplotlib

    Args:
        structure (Structure): structure to plot
        figure (plt.figure, optional): matplotlib Figure object. Defaults to None.
        axes (plt.axes, optional): matplotlib Axes objec. Defaults to None.
        color (_type_, optional): structure color. Defaults to None.
        annotation (bool, optional): add annotation to bar and node. Defaults to True.
        area_range (_type_, optional): scale line width on specific area range. Defaults to None.
    """
    trusses = structure._trusses
    constrain = structure._n_constrain
    nodes = structure._nodes

    if axes == None:
        axes = plt.gca()

    areas = trusses[1]
    if area_range == None:
        area_range = np.max(areas) 
    else:
        area_range = area_range[1] 
    adj = np.triu(trusses[0])
    area_ratio = np.triu(np.divide(areas, area_range, out=np.zeros_like(trusses[1]), where=(trusses[1]!=0)))
    n = len(nodes)
    
    max_x = np.max(nodes[:,0])
    max_y = np.max(nodes[:,1])
    min_x = np.min(nodes[:,0])
    min_y = np.min(nodes[:,1])

    axes.set_xlim(min_x-1, max_x+1)
    axes.set_ylim(min_y-1, max_y+1)
    axes.set_aspect('equal', adjustable='box')
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
                line = (3 - 1) * area_ratio[i,j] + 1
                axes.plot([s_x, e_x], [s_y, e_y], color=color, linewidth = line)
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


def save_report(structure: Structure, file: str) -> None:
    """
    Save structure parameter in a .txt report

    Args:
        structure (Structure): 
        file (str): output file path
    """
    with open(file, "w") as f:
        mass, total_mass, _ = structure.get_mass()
        d_max = structure.get_max_dispacement()
        # General data
        newline(f, "------ Units of measure are the same used in the problem definition ------")
        newline(f, "fitness = " + str(structure.get_fitness()))
        newline(f, "mass (no nodes) = " + str(mass))
        newline(f, "total mass (with nodes) = " + str(total_mass))
        newline(f, "max displacement = " + str(d_max))
        newline(f, "Can carry imposed loads? " + str(structure.is_broken()))
        newline(f, "Is valid structure? " + str(structure.is_valid()))
        # Nodes
        newline(f, "")
        newline(f, "------ Structure node values ------")
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
        
