from src.evolve import evolve, EvolutionParameter
from src.structure import StructureParameters, Node, SpaceArea, Material

####### 10 bar fixed node #######

# Units N, m, kg
l = 9.14
P = 4.45e5

# 5 --- 3 --- 1
# 6 --- 4 --- 2

problem = [
    Node(2*l, l, False, False, 0, 0),
    Node(2*l, 0, False, False, 0, -P),
    Node(l, l, False, False, 0, 0),
    Node(l, 0, False, False, 0, -P),
    Node(0, l, True, True, 0, 0),
    Node(0, 0, True, True, 0, 0),
]

connections = [
    [0, 1, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0]
]

sparam = StructureParameters()
sparam.corner = SpaceArea(0,l,2*l,l)
sparam.aggregation_radius = 0
sparam.material = Material(1.72e8, 2770, 6.9e10)
sparam.max_area = 0.01
sparam.min_area = 0.0005
sparam.round_digit = 4
sparam.node_mass_k = 1
sparam.safety_factor_buckling = 0
sparam.safety_factor_yield = 1 
sparam.max_displacement = 5.08e-2

eparam = EvolutionParameter()
eparam.elite_ratio = 0.1
eparam.kill_ratio = 0.1
eparam.total_mutation_ratio = 0.2
eparam.sort_adj_fitness = True
eparam.epochs = 2000
eparam.population = 100
eparam.mutation_area = 1
eparam.mutation_connection = 0
eparam.mutation_node_delete = 0
eparam.mutation_node_insert = 0
eparam.mutation_node_position = 0
eparam.node_range = [0,0]
eparam.dynamic_mutation = False
eparam.niche_radius = 0.0001
eparam.mass_target = 2000