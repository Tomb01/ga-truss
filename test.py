from src.structure import Structure, Node
import matplotlib.pyplot as plt
from src.plot import show
from src.genetis import crossover, area_mutation, get_compatibility

problem = [
    Node(0,0,True,True,0,0),
    Node(1,1,False,False,1000,0),
    Node(0,1,False,False,0,0),
    Node(1,0,True,True,0,0)
]

figure, axis = plt.subplots(1,3)

s = Structure(problem, elastic_modulus=1)
s1 = Structure(problem, elastic_modulus=1)
s.init_random(max_nodes=[0,2])
s1.init_random(max_nodes=[0,2])

area_mutation(s)

#show()