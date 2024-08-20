from src.structure import Node, Structure
import numpy as np
from src.plot import show
import matplotlib.pyplot as plt
from src.database import Database

np.set_printoptions(precision=8, suppress=True)

problem = [
    Node(0,0,True,True,0,0),
    Node(1,1,False,False,1000,0),
    Node(1,0,True,True,0,0)
]

s = Structure(problem, 1, 1, 1, 1, 1, [0, 0, 3, 3], 2, 2)
s.init_random([0,1], [0.1,10])
f = s.compute()

db = Database("test.db")
db.append_generation(1, f)
db.save_structure(1, s)

s1 = db.read_structure(1, 0)

figure, axis = plt.subplots(1,2)
s.plot(axis)

show()
