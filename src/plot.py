import matplotlib.pyplot as plt
import numpy as np

def plot_node(node: np.array) -> None:
    plt.scatter(node[0], node[1], s=5, c='r')
    
def plot_structure(nodes: np.array, trusses: np.array) -> None:
    adj = np.triu(trusses[0])
    
    n = len(nodes)
    for i in range(0, n):
        plot_node(nodes[i])
        conn = adj[i]
        for j in range(0, n):
            if conn[j]:
                s_x = nodes[i,0]
                e_x = nodes[j,0]
                s_y = nodes[i,1]
                e_y = nodes[j,1]
                plt.plot([s_x, e_x], [s_y, e_y])
                plt.text((s_x+e_x)/2, (s_y+e_y)/2, i, fontsize=8)
    
def show():
    plt.show()