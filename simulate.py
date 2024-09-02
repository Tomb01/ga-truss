from studies.simdata_10bar import problem, connections, eparam, sparam
from src.plot import plot_structure, savetxt
from src.evolve import evolve
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sim_name = sys.argv[1]
sim_count = int(sys.argv[2])

if len(sys.argv) > 3:
    sim_kill_ration = float(sys.argv[3])
    sim_mutation_ratio = float(sys.argv[4])
    sim_elite_ratio = float(sys.argv[5])

    # overwrite evolve param
    eparam.elite_ratio = sim_elite_ratio
    eparam.kill_ratio = sim_kill_ration
    eparam.total_mutation_ratio = sim_mutation_ratio
else:
    sim_elite_ratio = eparam.elite_ratio
    sim_kill_ration = eparam.kill_ratio
    sim_mutation_ratio = eparam.total_mutation_ratio

# sample, best, fitness
SAMPLE = 4
figure = plt.gcf()


SIMULATIONS = sim_count
mass = np.zeros(SIMULATIONS)
sim_folder = "{}/k{:.2f}m{:.2f}e{:.2f}".format(sim_name, sim_kill_ration, sim_mutation_ratio, sim_elite_ratio)
if not os.path.exists(sim_folder):
    os.makedirs(sim_folder)
print("Start {}".format(sim_folder))

for s in range(0, SIMULATIONS):
    fitness, best, sample, area_range = evolve(problem, eparam, sparam, sample_point=SAMPLE, constrained_connections=connections)
    mass[s] = best.get_mass()[0]
    to_plot = np.append(sample, [best])
    for f in range(0, SAMPLE+1):
        if to_plot[f] != None:
            plot_structure(to_plot[f], annotation=False, area_range=[sparam.min_area, sparam.max_area])
            figure.savefig("{}/{}_s{}.png".format(sim_folder, s, f))
            figure.clear()
    x_e = range(0, len(fitness))
    axis = plt.gca()
    axis.plot(x_e, fitness)
    axis.set_xlabel("generazioni")
    axis.set_ylabel("fitness")
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

