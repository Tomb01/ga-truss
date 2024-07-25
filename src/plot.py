import matplotlib.pyplot as plt
import numpy as np

def plot_node(plot, node: np.array, size=5, index=0) -> None:
    plot.scatter(node[0], node[1], size, c='r')
    plot.text(node[0], node[1], index, fontsize=8)
    
def plot_structure(nodes: np.array, trusses: np.array, constrain: int, axis, row, col, color="blue") -> None:
    adj = np.triu(trusses[0])
    if isinstance(axis[0], list):
        plot = axis[row, col]
    else:
        plot = axis[col]
        
    plot.grid()
    
    n = len(nodes)
    for i in range(0, n):
        plot_node(plot, nodes[i], 20 if i<constrain else 5, index=i)
        conn = adj[i]
        for j in range(0, n):
            if conn[j] == 1:
                s_x = nodes[i,0]
                e_x = nodes[j,0]
                s_y = nodes[i,1]
                e_y = nodes[j,1]
                plot.plot([s_x, e_x], [s_y, e_y], color=color)
                plot.text((s_x+e_x)/2, (s_y+e_y)/2, "{f:.3f}".format(f=trusses[5,i,j]), fontsize=8)
    
def show():
    plt.grid()
    plt.show()