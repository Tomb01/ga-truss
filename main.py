from src.structure import Node, Structure
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

max_epoch = 10
plot_interval = int(max_epoch/4)
figure, axis = plt.subplots(1,max_epoch//plot_interval+1)
#s.plot(axis)
best, fitness_values = epoch(problem, n_population=100, max_epoch=max_epoch, to_keep=0.1, area=[100,1000], elastic_modulus=69000, yield_stress=280, max_node=[1,20])
for i in range(0, len(best)):
    if i%plot_interval==0:
        best[i//plot_interval].plot(axis, 0, i//plot_interval+1)

axis[0].plot(np.linspace(0, max_epoch, num=max_epoch), fitness_values)

show()
