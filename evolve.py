from src.structure import Structure, Node, FLOAT_MAX
from src.genetis import get_distance, sharing, crossover, mutate
import numpy as np
import random
import matplotlib.pyplot as plt
from src.plot import show
from src.operations import vector_distance, binary_turnament
from src.raking import paretoRanking

# Problem parameter
problem = [
    Node(0,0,True,True,0,0),
    Node(25,1,False,False,0,1e4),
    #Node(0,1,False,False,0,0)
]

elastic_modulus = 72000
yield_strenght = 240
area = [0.000001,1000]
node_mass = 1
Fos_target = 1
density = 0.00000271
corner = [0,0,25,5]

# evolution parameter
EPOCH = 10
POPULATION = 100
START_NODE_RANGE = [0,2*len(problem)]
ELITE_RATIO = 0.1
KILL_RATIO = 0.5
MUTANT_RATIO = 0.1
NICHE_RADIUS = 0.1

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
fitness = np.zeros((POPULATION), dtype=float)
adj_fitness = np.zeros((POPULATION), dtype=float)

fitness_curve = np.zeros(EPOCH)
figure, axis = plt.subplots(1,5)
best = np.empty(4, dtype=np.object_)
b_count = 0

# New population distribution
elite_count = round(ELITE_RATIO*POPULATION)
mutant_count = round(MUTANT_RATIO*POPULATION)
crossover_count = POPULATION-mutant_count-elite_count

# Initial population -> random
for i in range(0, POPULATION):
    s = Structure(problem, elastic_modulus, Fos_target, node_mass, yield_strenght, density=density, corner=corner)
    s.init_random(max_nodes=START_NODE_RANGE, area=area)
    new_population[i] = s
    
# Evolution
for e in range(0, EPOCH):
    current_population = new_population 
    new_population = np.empty(POPULATION, dtype=np.object_)
    
    # Calculate fitness and adj fitness
    for i in range(0, POPULATION):
        fitness[i] = current_population[i].compute()    
        niche_count = sharing(current_population, i, NICHE_RADIUS)
        adj_fitness[i] = fitness[i]/niche_count
        #print(fitness[i], adj_fitness[i])
            
    # Crossover
    i = 0
    idx = np.arange(POPULATION)
    sorted_idx = np.argsort(adj_fitness)
    #print(sorted_idx)
    sorted_population = current_population[sorted_idx]
    fitness = fitness[sorted_idx]
    adj_fitness = adj_fitness[sorted_idx]
    
    while i < POPULATION and len(idx) > 2:
        idx_p1 = binary_turnament(idx)
        idx_p2 = binary_turnament(idx)
        p1 = idx[idx_p1]
        p2 = idx[idx_p2]
        if p1 != p2:
            parent1 = sorted_population[p1]
            parent2 = sorted_population[p2]
            c = crossover(parent1, parent2, len(problem), fitness[p1], fitness[p2])
            c = mutate(c, MUTATION_NODE_POSITION, MUTATION_AREA, MUTATION_CONNECTION, MUTATION_NODE_INSERT, MUTATION_NODE_DELETE, corner, area)
            child_fitness = c.compute()
            family_fitness = np.array([fitness[p1], fitness[p2], child_fitness])
            family = np.array([parent1, parent2, c])
            family_idx = np.argsort(family_fitness)
            new_population[i] = family[family_idx[0]]
            new_population[i+1] = family[family_idx[1]]
            i = i+2
            idx = np.delete(idx, [idx_p1, idx_p2])
            
    new_population[-2] = sorted_population[0]
    new_population[-1] = sorted_population[1]
    
    fitness_curve[e] = fitness[0]
    if e > 0:
        if e%(EPOCH//4) == 0:
            best[b_count] = sorted_population[0]
            b_count = b_count+1
    print(e, "---", sorted_population[0].compute())

best[-1] = sorted_population[0]
for j in range(0, len(best)):
    best[j].plot(axis, 0, j) 
print(best[-1].get_DOF(), best[-1].compute(), fitness_curve[e])
axis[-1].plot(range(0, EPOCH), fitness_curve)
show()
        
        
        
        
        
        
        