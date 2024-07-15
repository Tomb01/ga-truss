from src.operations import trussK, solve
from src.structure import Node
import numpy as np

np.set_printoptions(precision=3, suppress=True)

nodes = np.array([
    Node(0, 0, True, True, 0, 0),
    Node(1, 1, False, False, 1, 0),
    Node(1, 0, True, True, 0, 0),
])
adj = np.array([[0,1,0], [1,0,1], [0,1,0]])

n = len(nodes)
k = len(adj)

trusses = np.zeros((4, n, n))
trusses[0] = adj
trusses[1] = adj #areas

solve(nodes, trusses, 1)
print(trusses[3])


