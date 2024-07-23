from src.structure import Structure
from src.genetis import crossover, fit
import numpy as np
from typing import Tuple

def epoch(problem: np.array,  n_population=100, yield_stress = 1, elastic_modulus = 1, area = [0,1], max_epoch=100, max_node = [0,10]) -> Tuple[Structure, Structure]:
    mid_best: Structure
    population = np.empty(n_population, dtype=np.object_)
    best = np.empty(max_epoch, dtype=np.object_)
    fitness = np.zeros(n_population)
    
    for e in range(0, max_epoch):
        for p in range(0, n_population):
            if e == 0: # or p > n_population//4:
                # init all random
                s = Structure(problem, elastic_modulus)
                s.init_random(area=area, max_nodes=max_node)
                population[p] = s
                
            population[p].solve()
            fitness[p] = fit(population[p].get_fitness(yield_stess=yield_stress))
        
        # Order population by fitness
        current_fit = np.max(fitness)
        print(e, current_fit)
        fit_index = np.argsort(-fitness)
        fit_population = population[fit_index]
        
        best[e] = fit_population[0]
        
        parent1 = fit_population[0:n_population//2]
        parent2 = fit_population[n_population//2:]
        
        population[0:n_population//2] = parent1
        
        # make new populations
        for k in range(0, len(parent1)):
            population[k+len(parent1)] = crossover(parent1[k], parent2[k], len(problem))
            #population[:-k] = crossover(parent1[k], parent2[k], len(problem))
        
    return best