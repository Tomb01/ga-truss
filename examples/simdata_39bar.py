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

bar39_problem = [
    Node(0,0,True,True,0,0),
    Node(l,0,False,False,0,-P),
    Node(2*l,0,False,False,0,-P),
    Node(3*l,0,False,False,0,-P),
    Node(4*l,0,False,True, 0, 0)
]

connections = None

bar39_sparam = StructureParameters()
bar39_sparam.corner = SpaceArea(0,0.3*l,4*l,2*l)
bar39_sparam.aggregation_radius = 0.3*l
bar39_sparam.material = Material(1.38e8, 2768, 6.89e10)
bar39_sparam.max_area = 0.00145
bar39_sparam.min_area = 0.0000322
bar39_sparam.round_digit = 8
bar39_sparam.node_volume = 1
bar39_sparam.safety_factor_buckling = 0
bar39_sparam.safety_factor_yield = 1
bar39_sparam.max_displacement = 0.0508
bar39_sparam.max_length = 1.5*1.41*l

bar39_eparam = EvolutionParameter()
bar39_eparam.elite_ratio = 0.01
bar39_eparam.kill_ratio = 0.1
bar39_eparam.total_mutation_ratio = 0.25
bar39_eparam.sort_adj_fitness = True
bar39_eparam.epochs = 100
bar39_eparam.population = 200
bar39_eparam.mutation_area = 5
bar39_eparam.mutation_connection = 2
bar39_eparam.mutation_node_delete = 0.1
bar39_eparam.mutation_node_insert = 1
bar39_eparam.mutation_node_position = 1
bar39_eparam.node_range = [8,12]
bar39_eparam.niche_radius = 0.0015
bar39_eparam.mass_target = 1