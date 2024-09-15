from src.structure import Structure, Node, StructureParameters, SpaceArea, MATERIAL_ALLUMINIUM
from src.genetis import crossover
from src.plot import plot_structure
import matplotlib.pyplot as plt
from src.operations import is_cyclic

problem = [
    #Node(1, 1, False, False, 0, 1),
    Node(0, 0, True, True, 0, 0),
    Node(1, 0.2, False, False, 0, 1),
    Node(2, 0, True, True, 0, 0),
    Node(0, 3, True, True, 0, 0),
]

connections = [
    [0,1,1,1],
    [1,0,0,1],
    [1,0,0,0],
    [1,1,0,0]
]

sparam = StructureParameters()
sparam.corner = SpaceArea(0,0,3,3)
sparam.aggregation_radius = 0.1
sparam.material = MATERIAL_ALLUMINIUM
sparam.max_area = 1
sparam.min_area = 0.5
sparam.round_digit = 4
sparam.node_mass_k = 1
sparam.safety_factor_buckling = 0
sparam.safety_factor_yield = 1 
sparam.max_displacement = 1

s1 = Structure(problem, sparam)
s1.constrain_connections(connections)
s1.init_random([0,0])
print(s1.has_collinear_edge())

fig, ax = plt.subplots(1,3)
fig.set_figwidth(15)
plot_structure(s1, fig, ax[0], annotation=False)

plt.show()