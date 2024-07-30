from src.structure import Structure, Node
import matplotlib.pyplot as plt
from src.plot import show
from src.genetis import crossover, area_mutation, get_distance

problem = [
    Node(0,0,True,True,0,0),
    Node(1,1,False,False,1000,0),
    #Node(0,1,False,False,0,0),
    Node(1,0,True,True,0,0)
]

figure, axis = plt.subplots(1,3)

s = Structure(problem, 1, 1, 1, 1, [0,0,2,2])
s.init_random(max_nodes=[0,3], area=[0.001, 10])

s.add_node(2,2)
s.remove_node(3)
s.plot(axis)

show()