from src.structure import Structure, Node, FLOAT_MAX
from src.genetis import get_compatibility, is_shared, crossover, mutate
import numpy as np
import random
import matplotlib.pyplot as plt
from src.plot import show
from src.operations import vector_distance, binary_turnament

# Problem parameter
problem = [
    Node(0,0,True,True,0,0),
    Node(5,1,False,False,1000,0),
    Node(10,0,True,True,0,0),
    #Node(0,1,False,False,0,0)
]

elastic_modulus = 72000
yield_strenght = 261
area = [0.000001,100]
node_mass = 1
Fos_target = 2
corner = [0,0,10,10]

# evolution parameter
EPOCH = 10
POPULATION = 100
START_NODE_RANGE = [0,3]
ELITE_RATIO = 0.1
KILL_RATIO = 0.5
MUTANT_RATIO = 0.1

# Mutation
MUTATION_NODE_POSITION = 0.9
MUTATION_AREA = 0.9
MUTATION_CONNECTION = 0.9
MUTATION_NODE_DELETE = 0.1
MUTATION_NODE_INSERT = 0.1

# Init variables
current_population = np.empty(POPULATION, dtype=np.object_)
new_population = np.empty(POPULATION, dtype=np.object_)
sorted_population = np.empty(POPULATION, dtype=np.object_)
fos_fitness = np.zeros((POPULATION), dtype=float)
mass_fitness = np.zeros((POPULATION), dtype=float)
norm_fos_fitness = np.zeros((POPULATION), dtype=float)
norm_mass_fitness = np.zeros((POPULATION), dtype=float)
rank = np.zeros(POPULATION)

fitness_curve = np.zeros(EPOCH)
figure, axis = plt.subplots(1,2)

# New population distribution
elite_count = round(ELITE_RATIO*POPULATION)
mutant_count = round(MUTANT_RATIO*POPULATION)
crossover_count = POPULATION-mutant_count-elite_count

# Initial population -> random
for i in range(0, POPULATION):
    s = Structure(problem, elastic_modulus, Fos_target, node_mass, yield_strenght, corner=corner)
    s.init_random(max_nodes=START_NODE_RANGE, area=area)
    new_population[i] = s
    
# Evolution
for e in range(0, EPOCH):
    current_population = new_population 
    new_population = np.empty(POPULATION, dtype=np.object_)
    
    # Calculate fitness
    for i in range(0, POPULATION):
        fitness_i = current_population[i].compute()
        mass_fitness[i] = fitness_i[0]
        fos_fitness[i] = fitness_i[1]    
        
    # Normalize fitness
    mass_fitness_sum = np.sum(mass_fitness)
    fos_fitness_sum = np.sum(fos_fitness)
    norm_fos_fitness = fos_fitness/fos_fitness_sum
    norm_mass_fitness = mass_fitness/mass_fitness_sum
    # Normalize for multi-objective optimization
    norm_fos_fitness = norm_fos_fitness/np.min(norm_fos_fitness)
    norm_mass_fitness = norm_mass_fitness/np.min(norm_mass_fitness)
    
    #print(norm_fos_fitness, norm_mass_fitness)
    
    # Pareto ranking
    unassigned_rank = np.arange(POPULATION)
    rank_count = 1
    while len(unassigned_rank) > 0:
        all_dominated = True
        dominated = np.empty_like(unassigned_rank, dtype=bool)
        for i in range(0, len(unassigned_rank)):
            rank_idx_fos = norm_fos_fitness[i] > norm_fos_fitness
            rank_idx_mass = norm_mass_fitness[i] > norm_mass_fitness
            dominated[i] = np.any(rank_idx_fos*rank_idx_mass)
        
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
    for r in reversed(range(1, rank_count+1)):
        rank_idx = rank == r
        rank_population = current_population[rank_idx]
        fos_fitness_rank = norm_fos_fitness[rank_idx]
        mass_fitness_rank = norm_mass_fitness[rank_idx]
        rank_population_count = len(rank_population)
        
        # Sort in fos direction 
        sorted_fos_idx = np.argsort(fos_fitness_rank)
        fos_fitness_rank = fos_fitness_rank[sorted_fos_idx]
        crowding_distance = np.zeros_like(fos_fitness_rank)
        
        # Calculate distance in mass direction
        for j in range(0, len(rank_population)):
            if j == 0 or j == len(rank_population)-1:
                crowding_distance[j] = FLOAT_MAX
            else:
                distance_left = vector_distance(mass_fitness_rank[j-1], mass_fitness_rank[j], fos_fitness_rank[j-1], fos_fitness_rank[j])
                distance_right =  vector_distance(mass_fitness_rank[j+1], mass_fitness_rank[j], fos_fitness_rank[j+1], fos_fitness_rank[j])
                #print(distance_left, distance_right)
                crowding_distance[j] = distance_left + distance_right
                
        sorted_distance_idx = np.argsort(-crowding_distance)
        rank_population = rank_population[sorted_fos_idx]
        sorted_population[rank_sorting_offset:rank_sorting_offset+rank_population_count] = rank_population
        rank_sorting_offset = rank_sorting_offset + rank_population_count
        
    # Selection
    idx = np.arange(POPULATION)
    for i in range(0, POPULATION):
        p1 = binary_turnament(idx)
        p2 = binary_turnament(idx)
        c = crossover(sorted_population[p1], sorted_population[p2], len(problem), p1, p2)
        c = mutate(c, MUTATION_NODE_POSITION, MUTATION_AREA, MUTATION_CONNECTION, MUTATION_NODE_INSERT, MUTATION_NODE_DELETE, corner, area)
        new_population[i] = c
        
    print(e, "---", sorted_population[0].compute())
        
    # Check new population
    #if np.all(new_population == None):
    #    raise Exception("Population off")

best = current_population[0]
best.plot(axis, 0, 0)
print(best.get_DOF(), best.compute(), fitness_curve[e])

# Print fitness
axis[-1].scatter(norm_fos_fitness, norm_mass_fitness)
for j in range(0, len(norm_fos_fitness)):
    axis[-1].annotate(str(rank[j]), (norm_fos_fitness[j], norm_mass_fitness[j]))

show()
        
        
        
        
        
        
        