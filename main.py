from src.operations import solve
from src.structure import Structure, Node
import numpy as np
from src.plot import show

np.set_printoptions(precision=3, suppress=True)

s = Structure([
    Node(0,0,True,True,0,0),
    Node(1,1,False,False,1000,0),
    Node(1,0,True,True,0,0)
])

s.init_random(max_nodes=3)
s.solve()
s.plot()
show()