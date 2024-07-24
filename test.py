from src.structure import Structure, Node
import matplotlib.pyplot as plt
from src.plot import show

problem = [
    Node(0,0,True,True,0,0),
    Node(1,1,False,False,1000,0),
    Node(1,0,True,True,0,0)
]

figure, axis = plt.subplots(1,2)

s = Structure(problem, elastic_modulus=1)
s.init_random(max_nodes=[0,6])
print(s.check())
s.solve()
s.plot(axis)

show()