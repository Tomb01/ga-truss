from src.operations import solve
from src.structure import Structure, Node
import numpy as np
from src.plot import show
from src.genetis import crossover
import matplotlib.pyplot as plt

np.set_printoptions(precision=8, suppress=True)

problem = [
    Node(0,0,True,True,0,0),
    Node(1,1,False,False,1000,0),
    Node(1,0,True,True,0,0)
]

s1 = Structure(problem)
s2 = Structure(problem)

k = 0

while True:
    try:
        s1.init_random(max_nodes=3)
        s2.init_random(max_nodes=1)
        s1.solve()
        s2.solve()
        k = k+1
        
        if len(s1._nodes) == len(s2._nodes):
            pass
        else:
            print("solved", len(s1._nodes), len(s2._nodes))

        c = crossover(s1, s2, len(problem))
        
        figure, axis = plt.subplots(1,3)
        
        s1.plot(axis, 0, 0, color="green")
        s2.plot(axis, 0, 1, color="blue")
        c.plot(axis, 0, 2, color="red")

        show()
        break
    
    except Exception as e:
        print(e)
        if e.args[0] == "Invalid structure" or e.args[0] == "Singular matrix":
            if k > 100:
                break
            else:
                pass
        else:
            print(k)
            raise e
