import numpy as np
from src.operations import FLOAT_MAX, vector_distance

def paretoRanking(fitness_1: np.array, fitness_2: np.array) -> np.array:
    n = len(fitness_1)
    population_idx = np.arange(n, dtype=int)
    sorted_population = np.zeros(n, dtype=int)
    unassigned_rank = np.arange(n)
    rank = np.zeros(n)
    rank_count = 1
    
    while len(unassigned_rank) > 0:
        dominated = np.empty_like(unassigned_rank, dtype=bool)
        for i in range(0, len(unassigned_rank)):
            if fitness_1[i] < FLOAT_MAX and fitness_2[i] < FLOAT_MAX:
                rank_idx_fit1 = fitness_1[i] > fitness_1
                rank_idx_fit2 = fitness_2[i] > fitness_2
                dominated[i] = np.any(rank_idx_fit1*rank_idx_fit2)
            else:
                dominated[i] = True
        
        if np.all(dominated == True) or len(unassigned_rank) == 1:
            rank[rank == 0] = rank_count
            break
        else:
            dominant_idx = unassigned_rank[np.invert(dominated)]
            rank[dominant_idx] = rank_count
            unassigned_rank = unassigned_rank[dominated]
            rank_count = rank_count + 1
            
    # Crowding and sort population
    rank_sorting_offset = 0
    for r in range(1, rank_count+1):
        rank_idx = rank == r
        rank_population = population_idx[rank_idx]
        fos_fitness_rank = fitness_1[rank_idx]
        mass_fitness_rank = fitness_2[rank_idx]
        rank_population_count = len(rank_population)
        
        # Sort in fos direction 
        sorted_fos_idx = np.argsort(fos_fitness_rank)
        fos_fitness_rank = fos_fitness_rank[sorted_fos_idx]
        crowding_distance = np.zeros_like(fos_fitness_rank)
        
        # Calculate distance in mass direction
        for j in range(0, len(rank_population)):
            if j == 0 or j == len(rank_population)-1:
                crowding_distance[j] = 0
            else:
                distance_left = vector_distance(mass_fitness_rank[j-1], mass_fitness_rank[j], fos_fitness_rank[j-1], fos_fitness_rank[j])
                distance_right =  vector_distance(mass_fitness_rank[j+1], mass_fitness_rank[j], fos_fitness_rank[j+1], fos_fitness_rank[j])
                #print(distance_left, distance_right)
                crowding_distance[j] = distance_left + distance_right
                
        sorted_distance_idx = np.argsort(-crowding_distance)
        rank_population = rank_population[sorted_distance_idx]
        sorted_population[rank_sorting_offset:rank_sorting_offset+rank_population_count] = rank_population
        rank_sorting_offset = rank_sorting_offset + rank_population_count
        
    return sorted_population