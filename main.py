from src.structure import Node
import numpy as np
from src.evolution import epoch
from src.plot import show
import matplotlib.pyplot as plt

np.set_printoptions(precision=8, suppress=True)

problem = [
    Node(0,0,True,True,0,0),
    Node(1,1,False,False,1000,0),
    Node(1,0,True,True,0,0)
]

figure, axis = plt.subplots(1,2)

best, mid_best = epoch(problem)
best.plot(axis)
mid_best.plot(axis, 0, 1)

show()
