from src.structure import Structure, Node, StructureParameters, Material, SpaceArea, MATERIAL_ALLUMINIUM
from src.plot import show, plot_structure
from src.operations import make_sym
from src.genetis import connection_mutation, crossover
import numpy as np
import matplotlib.pyplot as plt
import copy
from src.utils.misc import node2table

problem = [
    Node(0,0,True,True,0,0),
    Node(1,1,False,False,1000,0),
    Node(1,0,True,True,0,0),
    Node(0,1,False,False,-1000,0),
    #Node(0.5,1.2,False,False,0,0)
]

n = len(problem)
trusses = np.zeros((7,n,n))
#trusses[0] = make_sym(np.array([[0, 1, 0, 1, 1], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0,0,0,0,0]]))
#trusses[0] = make_sym(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))
#trusses[1] = trusses[0]  

area = [0.1, 1]
nodes_range = [1,20]

param = StructureParameters()
param.corner = SpaceArea(-0.5,-0.5,3,3)
param.aggregation_radius = 0.3
param.safety_factor_yield = 1
param.safety_factor_buckling = 1.5
param.material = MATERIAL_ALLUMINIUM
param.node_mass_k = 1
param.round_digit = 2
param.max_area = 1
param.min_area = 1

p1 = Structure(problem, param)
p2 = Structure(problem, param)

p1.init_random(nodes_range, area)
p2.init_random(nodes_range, area)

fit1 = p1.compute()
fit2 = p2.compute()

c = crossover(p1, p2, len(problem), fit1, fit2)
c.compute()

figure, axes = plt.subplots(1,3, )

figure.set_figheight(5)
figure.set_figwidth(15)

plot_structure(p1, figure, axes[0], annotation=False, area=area)
plot_structure(p2, figure, axes[1], annotation=False, area=area)
plot_structure(c, figure, axes[2], annotation=False, area=area)

print(c.is_broken(), c._valid, c.get_DOF())

show()