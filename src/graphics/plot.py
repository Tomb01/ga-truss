import matplotlib.pyplot as plt
from src.classes.structure import Structure, Truss, TrussNode

def plot_node(node: TrussNode) -> None:
    x,y = node.get_coordinate()
    plt.scatter(x, y, s=5, c='r')
    
def plot_truss(truss: Truss) -> None:
    start, end = truss.get_nodes()
    plot_node(start)
    plot_node(end)
    s_x, s_y = start.get_coordinate()
    e_x, e_y = end.get_coordinate()
    plt.plot([s_x, e_x], [s_y, e_y])
    
def plot_structure(structure: Structure) -> None:
    for truss in structure._trusses:
        plot_truss(truss)
    
def show():
    plt.show()