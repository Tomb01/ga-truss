from src.structure import Structure, Node, StructureParameters, Material, SpaceArea
from src.plot import show, plot_structure
from src.operations import make_sym
import numpy as np

problem = [
    Node(0,0,True,True,0,0),
    Node(1,1,False,False,1,0),
    #Node(0,1,False,False,0,0),
    Node(1,0,True,True,0,0)
]

n = len(problem)
trusses = np.zeros((7,n,n))
trusses[0] = make_sym(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))
trusses[1] = make_sym(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))

param = StructureParameters()
param.corner = SpaceArea(-0.5,-0.5,1.5,1.5)
param.aggregation_radius = 1
param.safety_factor_yield = 1
param.material = Material(1, 1, 1)
param.node_mass_k = 1
param.round_digit = 1

s = Structure(problem, param)
s._trusses = trusses
s.compute()

print(s._trusses)

plot_structure(s)
show()