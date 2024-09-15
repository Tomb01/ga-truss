from src.evolve import evolve, EvolutionParameter
from src.structure import StructureParameters, Node, SpaceArea, Material

####### 39 bar #######

ksi = 6.895e+6
in2 = 0.00064516
inch = 0.0254
lbin3 = 27679.9
lb2N =	4.44822

# Units N, m, kg
l = 3.048
P = 88964.43

# areas = 0.05 in2 (3.226×10−5 m2) and 2.25 in2 (1.452×10−3 m2)

problem = [
    Node(0,0,True,True,0,0),
    Node(l,0.01,False,False,0,-P),
    Node(2*l,0,False,False,0,-P),
    Node(3*l,0.01,False,False,0,-P),
    Node(4*l,0,False,True, 0, 0)
]

"""problem = [
    Node(0,0,True,True,0,0),
    Node(0,l,False,False,0,0),
    Node(l,0,False,False,0,-P),
    Node(2*l,0,True,False,0,-P/2),
    Node(2*l,2*l,True,False,0,0)
]"""


connections = None

sparam = StructureParameters()
sparam.corner = SpaceArea(0,0.5*l,4*l,2*l)
sparam.aggregation_radius = 0.2*l
sparam.material = Material(1.38e8, 2768, 6.89e10)
sparam.max_area = 0.00145
sparam.min_area = 0.0000322
sparam.round_digit = 8
sparam.node_mass_k = 1
sparam.safety_factor_buckling = 0
sparam.safety_factor_yield = 1
sparam.max_displacement = 0.0508
sparam.max_length = 1.5*1.41*l

eparam = EvolutionParameter()
eparam.elite_ratio = 0.01
eparam.kill_ratio = 0.1
eparam.total_mutation_ratio = 0.1
eparam.sort_adj_fitness = True
eparam.epochs = 100
eparam.population = 100
eparam.mutation_area = 3
eparam.mutation_connection = 0
eparam.mutation_node_delete = 0.1
eparam.mutation_node_insert = 0
eparam.mutation_node_position = 1
eparam.node_range = [2,3]
eparam.dynamic_mutation = False
eparam.niche_radius = 0.001
eparam.mass_target = 1