import matplotlib.pyplot as plt
import numpy as np

def plot_node(node: np.array, size=5, index=0) -> None:
    plt.scatter(node[0], node[1], size, c='r')
    plt.text(node[0], node[1], index, fontsize=8)
    
def plot_structure(nodes: np.array, trusses: np.array, constrain: int) -> None:
    adj = np.triu(trusses[0])
    
    n = len(nodes)
    for i in range(0, n):
        plot_node(nodes[i], 20 if i<constrain else 5, index=i)
        conn = adj[i]
        for j in range(0, n):
            if conn[j]:
                s_x = nodes[i,0]
                e_x = nodes[j,0]
                s_y = nodes[i,1]
                e_y = nodes[j,1]
                plt.plot([s_x, e_x], [s_y, e_y])
                #plt.text((s_x+e_x)/2, (s_y+e_y)/2, i, fontsize=8)
    
def show():
    plt.grid()
    plt.show()