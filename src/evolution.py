from src.structure import Structure
from src.genetis import crossover, fit
import numpy as np
from typing import Tuple
import random

def epoch(problem: np.array, n_population=100, yield_stress = 1, to_keep = 0.5,elastic_modulus = 1, area = [0,1], max_epoch=100, max_node = [0,10]) -> Tuple[Structure, Structure]:
    mid_best: Structure
    population = np.empty(n_population, dtype=np.object_)
    best = np.empty(max_epoch, dtype=np.object_)
    fitness = np.zeros(n_population)
    fitness_values = np.zeros(max_epoch)
    
    for e in range(0, max_epoch):
        for p in range(0, n_population):
            if e == 0: # or p > n_population//4:
                # init all random
                s = Structure(problem, elastic_modulus)
                s.init_random(area_range=area, nodes_range=max_node)
                population[p] = s
                
            population[p].solve()
            fitness[p] = fit(population[p].get_fitness(yield_stess=yield_stress))
        
        # Order population by fitness
        current_fit = np.max(fitness)
        print(e, current_fit)
        fit_index = np.argsort(-fitness)
        fit_population = population[fit_index]
        fitness = fitness[fit_index]
        first_kill = np.where(fitness == 0)[0]
        if len(first_kill) == 0:
            first_kill = n_population
        else:
            first_kill = first_kill[0]
        filter_population = min(first_kill, int(n_population*to_keep))
          
        best[e] = fit_population[0]
        fitness_values[e] = current_fit
        
        best_fit = fit_population[0:filter_population]
        print(filter_population)
        
        # make new populations
        for k in range(0, n_population):
            p1 = random.randrange(0, len(best_fit))
            p2 = random.randrange(0, len(best_fit))
            population[k] = crossover(best_fit[p1], best_fit[p2], len(problem), fitness_values[p1], fitness_values[p2])
            #population[:-k] = crossover(parent1[k], parent2[k], len(problem))
        
    return best, fitness_values