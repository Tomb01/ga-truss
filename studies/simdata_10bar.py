from src.evolve import evolve, EvolutionParameter
from src.structure import StructureParameters, Node, SpaceArea, Material

####### 10 bar free node #######

# Units N, m, kg
l = 9.14
P = 4.45e5

# 5 --- 3 --- 1
# 6 --- 4 --- 2

problem = [
    Node(2*l, 0, False, False, 0, -P),
    Node(l, 0, False, False, 0, -P),
    Node(0, l, True, True, 0, 0),
    Node(0, 0, True, True, 0, 0),
]

connections = None

sparam = StructureParameters()
sparam.corner = SpaceArea(0,0,2*l,l)
sparam.aggregation_radius = 0.01*l
sparam.material = Material(1.72e8, 2770, 6.9e10)
sparam.max_area = 0.03
sparam.min_area = 0.0001
sparam.round_digit = 4
sparam.node_mass_k = 1
sparam.safety_factor_buckling = 0
sparam.safety_factor_yield = 1 
sparam.max_displacement = 5.08e-2

eparam = EvolutionParameter()
eparam.elite_ratio = 0.01
eparam.kill_ratio = 0.05
eparam.total_mutation_ratio = 0.1
eparam.sort_adj_fitness = True
eparam.epochs = 2000
eparam.population = 100
eparam.mutation_area = 3
eparam.mutation_connection = 2
eparam.mutation_node_delete = 0.1
eparam.mutation_node_insert = 1
eparam.mutation_node_position = 3
eparam.node_range = [10,15]
eparam.dynamic_mutation = False
eparam.niche_radius = 0.001
eparam.mass_target = 2200