from src.evolve import evolve, EvolutionParameter
from src.structure import StructureParameters, Node, SpaceArea, Material, Structure
from src.plot import plot_structure
import matplotlib.pyplot as plt

# Units N, mm, kg
l = 9*1e3

problem = [
    Node(0, 0, True, True, 0, 0),
    Node(0, l, True, True, 0, 0),
    Node(l, 0, False, False, 0, 450e3),
    Node(l, l, False, False, 0, 0),
    Node(2*l, 0, False, False, 0, 450e3),
    Node(2*l, l, False, False, 0, 0)
]

connections = [
    []
]

sparam = StructureParameters()
sparam.corner = SpaceArea(0,l,2*l,l)
sparam.aggregation_radius = 10
sparam.material = Material(130, 0.261, 70000)
sparam.max_area = 1000
sparam.min_area = 1
sparam.round_digit = 0
sparam.node_mass_k = 1
sparam.safety_factor_buckling = 1
sparam.safety_factor_yield = 1 

eparam = EvolutionParameter()
eparam.elite_ratio = 0.1
eparam.kill_ratio = 0.05
eparam.total_mutation_ratio = 0.1
eparam.enable_adj_fitness = True
eparam.epochs = 100
eparam.population = 100
eparam.mutation_area = 10
eparam.mutation_connection = 0
eparam.mutation_node_delete = 0
eparam.mutation_node_insert = 0
eparam.mutation_node_position = 0
eparam.node_range = [0,0]

# sample, best, fitness
SAMPLE = 4
figure, axes = plt.subplots(1,SAMPLE+2)
figure.set_figheight(5)
figure.set_figwidth(15)

s = Structure(problem, sparam)
s.init_random([0,0], [1, 1000])
plot_structure(s, figure, axes[-1])
plt.show(block=True)

# Evolve
fitness, best, sample = evolve(problem, eparam, sparam, sample_point=SAMPLE)

axes[-1].plot(range(0, eparam.epochs), fitness)
s = 0
for s in range(0, SAMPLE):
    plot_structure(sample[s], figure, axes[s], annotation=True)

plot_structure(best, figure, axes[-2], annotation=True)
print(fitness[-1], best.is_broken(), best.check())
plt.show()