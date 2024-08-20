from src.structure import Structure, Node, StructureParameters, MATERIAL_ALLUMINIUM, SpaceArea
from src.plot import show, plot_structure

problem = [
    Node(0,0,True,True,0,0),
    Node(1,1,False,False,1000,0),
    #Node(0,1,False,False,0,0),
    Node(1,0,True,True,0,0)
]

param = StructureParameters()
param.corner = SpaceArea(0,0,2,2)
param.crossover_radius = 1
param.safety_factor_yield = 1
param.material = MATERIAL_ALLUMINIUM
param.node_mass_k = 1
param.round_digit = 1

s = Structure(problem, param)
s.init_random([0,1], [1000,1000.1])
s.compute()

plot_structure(s)
show()