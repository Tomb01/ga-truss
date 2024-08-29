from src.evolve import evolve, EvolutionParameter
from src.structure import StructureParameters, Node, SpaceArea, Material, Structure
from src.plot import plot_structure
import matplotlib.pyplot as plt
import numpy as np

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
sparam.aggregation_radius = 0.1
sparam.material = Material(1.72e8, 2770, 6.9e10)
sparam.max_area = 0.1
sparam.min_area = 1e-6
sparam.round_digit = 8
sparam.node_mass_k = 1
sparam.safety_factor_buckling = 0
sparam.safety_factor_yield = 1 

eparam = EvolutionParameter()
eparam.elite_ratio = 0.1
eparam.kill_ratio = 0.1
eparam.total_mutation_ratio = 0.1
eparam.sort_adj_fitness = True
eparam.epochs = 100
eparam.population = 100
eparam.mutation_area = 10
eparam.mutation_connection = 0
eparam.mutation_node_delete = 0
eparam.mutation_node_insert = 0
eparam.mutation_node_position = 0
eparam.node_range = [0,0]
eparam.dynamic_area = False
 
# sample, best, fitness
SAMPLE = 4
figure, axes = plt.subplots(1,SAMPLE+2)
figure.set_figheight(5)
figure.set_figwidth(15)

# Evolve
fitness, best, sample, area_range = evolve(problem, eparam, sparam, sample_point=SAMPLE, constrained_connections=connections)

axes[-1].plot(range(0, eparam.epochs), fitness)
s = 0
for s in range(0, SAMPLE):
    plot_structure(sample[s], figure, axes[s], annotation=False)

plot_structure(best, figure, axes[-2], annotation=False)
#print(best._trusses[1], best._trusses[2])
np.set_printoptions(precision=3, suppress=True)
print(best._trusses[1]*1e5)
print(best._trusses[5])
print(fitness[-1], best.is_broken(), best.check(), best.get_mass(), best.get_max_dispacement())
plt.show()