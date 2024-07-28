from src.structure import Structure, Node, FLOAT_MAX
from src.genetis import get_compatibility, is_shared, crossover, mutate
import numpy as np
import random
import matplotlib.pyplot as plt
from src.plot import show
from src.operations import vector_distance, binary_turnament
from src.raking import paretoRanking

# Problem parameter
problem = [
    Node(0,0,True,True,0,0),
    Node(50,1,False,False,1e5,0),
    Node(100,0,True,True,0,0),
    #Node(0,1,False,False,0,0)
]

elastic_modulus = 72000
yield_strenght = 261
area = [0.000001,1000]
node_mass = 1
Fos_target = 1
corner = [0,0,100,100]

# evolution parameter
EPOCH = 10
POPULATION = 100
START_NODE_RANGE = [0,10]
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
fitness_Fos = np.zeros((POPULATION), dtype=float)
norm_fitness = np.zeros((POPULATION), dtype=float)
fitness_mass = np.zeros((POPULATION), dtype=float)

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
        fitness_Fos[i] = fitness_i[1]
        fitness_mass[i] = fitness_i[0]
        
    # Crossover
    i = 0
    idx = np.arange(POPULATION)
    sorted_idx = np.argsort(fitness_Fos)
    #print(sorted_idx)
    sorted_population = current_population[sorted_idx]
    fitness_Fos = fitness_Fos[sorted_idx]
    fitness_mass =fitness_mass[sorted_idx]
    while i < POPULATION and len(idx) > 2:
        idx_p1 = binary_turnament(idx)
        idx_p2 = binary_turnament(idx)
        p1 = idx[idx_p1]
        p2 = idx[idx_p2]
        if p1 != p2:
            parent1 = sorted_population[p1]
            parent2 = sorted_population[p2]
            c = crossover(parent1, parent2, len(problem), fitness_Fos[p1], fitness_Fos[p2])
            c = mutate(c, MUTATION_NODE_POSITION, MUTATION_AREA, MUTATION_CONNECTION, MUTATION_NODE_INSERT, MUTATION_NODE_DELETE, corner, area)
            child_fitness = c.compute()
            family_fitness = np.array([fitness_Fos[p1], fitness_Fos[p2], child_fitness[1]])
            family = np.array([parent1, parent2, c])
            family_idx = np.argsort(family_fitness)
            new_population[i] = family[family_idx[0]]
            new_population[i+1] = family[family_idx[1]]
            i = i+2
            idx = np.delete(idx, [idx_p1, idx_p2])
    
    new_population[-2] = sorted_population[0]
    new_population[-1] = sorted_population[1]
    fitness_curve[e] = fitness_Fos[0]
    print(e, "---", sorted_population[0].compute()[1])

best = sorted_population[0]
best.plot(axis, 0, 0)
print(best.get_DOF(), best.compute()[1], fitness_curve[e])
axis[-1].plot(range(0, EPOCH), fitness_curve)
show()
        
        
        
        
        
        
        