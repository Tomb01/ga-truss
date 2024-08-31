from src.evolve import evolve, EvolutionParameter
from src.structure import StructureParameters, Node, SpaceArea, Material, Structure
from src.plot import plot_structure, savetxt
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sim_name = sys.argv[1]
sim_kill_ration = float(sys.argv[2])
sim_mutation_ratio = float(sys.argv[3])
sim_elite_ratio = float(sys.argv[4])

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
sparam.max_area = 0.05
sparam.min_area = 0.001
sparam.round_digit = 3
sparam.node_mass_k = 1
sparam.safety_factor_buckling = 0
sparam.safety_factor_yield = 1 
sparam.max_displacement = 5.08e-2

eparam = EvolutionParameter()
eparam.elite_ratio = sim_elite_ratio
eparam.kill_ratio = sim_kill_ration
eparam.total_mutation_ratio = sim_mutation_ratio
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
eparam.niche_radius = 0.0005
eparam.mass_target = 2400

# sample, best, fitness
SAMPLE = 4
figure = plt.gcf()

SIMULATIONS = 5
mass = np.zeros(SIMULATIONS)
sim_folder = "{}/k{:.2f}m{:.2f}e{:.2f}".format(sim_name, sim_kill_ration, sim_mutation_ratio, sim_elite_ratio)
if not os.path.exists(sim_folder):
    os.makedirs(sim_folder)
print("Start {}".format(sim_folder))

for s in range(0, SIMULATIONS):
    # Evolve
    fitness, best, sample, area_range = evolve(problem, eparam, sparam, sample_point=SAMPLE, constrained_connections=connections)
    mass[s] = best.get_mass()[0]
    plot_structure(best, annotation=False)
    figure.savefig("{}/{}_s.png".format(sim_folder, s))
    figure.clear()
    x_e = range(0, len(fitness))
    plt.plot(x_e, fitness)
    np.savetxt("{}/{}_f.csv".format(sim_folder, s), np.array([x_e, fitness]).T, delimiter=";",)
    figure.savefig("{}/{}_f.png".format(sim_folder, s))
    figure.clear()
    
    # Print and save
    print(s, len(fitness), fitness[-1], best.is_broken(), best.check(), best.get_mass()[0], best.get_max_dispacement(), area_range)
    savetxt(best, "{}/{}_s.csv".format(sim_folder, s))

# append to master file
fm = open("{}/../master.csv".format(sim_folder), "a")
fm.write("{};{};{};{};{}\n".format(eparam.kill_ratio, eparam.total_mutation_ratio, eparam.elite_ratio, len(fitness), ";".join(str(m) for m in mass)))
fm.close()

