from src.structure import Structure
from src.genetis import crossover
import numpy as np
from typing import Tuple

def epoch(problem: np.array, elastic_modulus = 1, max_epoch=100, n_population=100) -> Tuple[Structure, Structure]:
    best_structure: Structure
    mid_best: Structure
    
    for e in range(0, max_epoch):
        population = np.empty(n_population, dtype=np.object_)
        fitness = np.zeros(n_population)
        for p in range(0, n_population):
            s = Structure(problem, elastic_modulus)
            if e == 0:
                # init all random
                s.init_random()
            else:
                # test crossover with random
                s.init_random()
                #s = crossover(best_structure, s, len(problem))
                
            s.solve()
            population[p] = s
            fitness[p] = s.get_fitness()
        
        # best fitness
        best_idx = np.argmax(fitness)
        best_structure = population[best_idx]
        
        if e == max_epoch//2:
            print("midbest")
            mid_best = best_structure
        
    return best_structure, mid_best