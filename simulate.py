from examples.simdata_39bar import bar39_problem, bar39_sparam, bar39_eparam
from src.plot import plot, save_report
from src.evolve import evolve
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

#### SIMULATIONS SETUP ####

# Problem definition (see examples) and parameters
problem = bar39_problem
eparam = bar39_eparam
sparam = bar39_sparam
connections = [] # not used in this case

#### END SIMULATIONS SETUP ####

# Name of the simulation
name = sys.argv[1]
# Number of simulation
count = int(sys.argv[2])

# If passed by argument the evolution parameter are overwritten (used in case of multiple simulations)
if len(sys.argv) > 3:
    kill_ratio = float(sys.argv[3])
    mutation_ratio = float(sys.argv[4])
    elite_ratio = float(sys.argv[5])
    # overwrite evolve param
    eparam.elite_ratio = elite_ratio
    eparam.kill_ratio = kill_ratio
    eparam.total_mutation_ratio = mutation_ratio
else:
    elite_ratio = eparam.elite_ratio
    kill_ratio = eparam.kill_ratio
    mutation_ratio = eparam.total_mutation_ratio

# Number of sample
samples = 10
figure = plt.gcf()

#### INIT THE SIMULATIONS ####
simulations = count
mass = np.zeros(simulations)
# All the results will be saved in the folder passed as "name" argument
# For each simulation a subfolder is created with the name format "k[kill_ratio]m[mutant_ratio]e[elite_ratio]"
output_folder = "{}/k{:.2f}m{:.2f}e{:.2f}".format(name, kill_ratio, mutation_ratio, elite_ratio)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print("Start run {}".format(output_folder))

#### EXECUTE ALL THE SIMULATIONS ####
for s in range(0, simulations):
    # Evolve the algorithm
    fitness, best, sample = evolve(problem, eparam, sparam, sample_point=samples)
    # Print and save
    print(s, len(fitness), fitness[-1], best.is_broken(), best.is_valid(), best.get_mass()[0], best.get_max_dispacement(), np.max(best._trusses[6])/sparam.max_length, np.max(best._trusses[5]))
    mass[s] = best.get_mass()[0]
    # Plot the results
    to_plot = np.append(sample, [best])
    for f in range(0, samples+1):
        if to_plot[f] != None:
            plot(to_plot[f], annotation=False, area_range=[sparam.min_area, sparam.max_area])
            figure.savefig("{}/run{}_sample{}.png".format(output_folder, s, f))
            figure.clear()
    x_e = range(0, len(fitness))
    axis = plt.gca()
    axis.plot(x_e, fitness)
    axis.set_xlabel("generazioni")
    axis.set_ylabel("fitness")
    np.savetxt("{}/run{}_fitness.csv".format(output_folder, s), np.array([x_e, fitness]).T, delimiter=";",)
    figure.savefig("{}/run{}_fitness.png".format(output_folder, s))
    figure.clear()
    save_report(best, "{}/run{}_sample.csv".format(output_folder, s))

# append to master file
fm = open("{}/../master.csv".format(output_folder), "a")
fm.write("{};{};{};{};{}\n".format(eparam.kill_ratio, eparam.total_mutation_ratio, eparam.elite_ratio, len(fitness), ";".join(str(m) for m in mass)))
fm.close()

