from src.structure import Structure, Node
from src.genetis import get_compatibility, is_shared, crossover, mutate
import numpy as np
import random
import matplotlib.pyplot as plt
from src.plot import show

# Problem parameter
problem = [
    Node(0,0,True,True,0,0),
    Node(5,1,False,False,1000,0),
    Node(10,0,True,True,0,0),
    #Node(0,1,False,False,0,0)
]

elastic_modulus = 72000
yield_strenght = 261
area = [0.0000001,100]
node_mass = 1
Fos_target = 2
corner = [0,0,10,10]

# evolution parameter
EPOCH = 100
POPULATION = 1000
START_NODE_RANGE = [0,10]
ELITE_RATIO = 0.05
KILL_RATIO = 0.8
MUTANT_RATIO = 0.5

# Mutation
MUTATION_NODE_POSITION = 0.9
MUTATION_AREA = 0.9
MUTATION_CONNECTION = 0.9
MUTATION_NODE_DELETE = 0.1
MUTATION_NODE_INSERT = 0.1

# Init variables
current_population = np.empty(POPULATION, dtype=np.object_)
new_population = np.empty(POPULATION, dtype=np.object_)
fitness = np.zeros(POPULATION, dtype=float)
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
    
fx_computation = np.vectorize(Structure.compute)
    
# Evolution
for e in range(0, EPOCH):
    current_population = new_population 
    new_population = np.empty(POPULATION, dtype=np.object_)
    
    # Calculate fitness value
    fitness = fx_computation(current_population)
    fitness_max = np.max(fitness)
    fitness_curve[e] = fitness_max
    fitness = np.divide(fitness, fitness_max)
    
    if np.max(fitness)<=1e-4:
        print(fitness)
        break
    
    # Sort
    sorted_idx = np.argsort(-fitness)
    current_population = current_population[sorted_idx]
    #print(current_population[0].compute())
    fitness = fitness[sorted_idx]
    
    # Elite -> pass without modifications
    new_population[0:elite_count] = current_population[0:elite_count]
    
    to_edit_range = range(elite_count, round(POPULATION*KILL_RATIO))
    # Crossover
    crossover_range = range(elite_count, elite_count+crossover_count)
    for k in crossover_range:
        parents = random.sample(to_edit_range, 2)
        p1 = parents[0]
        p2 = parents[1]
        new_population[k] = crossover(current_population[p1], current_population[p2], len(problem), fitness[p1], fitness[p2])
        
    # Mutant
    for m in range(elite_count+crossover_count, POPULATION):
        i = random.choice(to_edit_range)
        new_population[m] = mutate(current_population[i], MUTATION_NODE_POSITION, MUTATION_AREA, MUTATION_CONNECTION, MUTATION_NODE_INSERT, MUTATION_NODE_DELETE, corner[0], corner[1], corner[2], corner[3], area[0], area[1])
    
    #Check new population
    if np.all(new_population == None):
        raise Exception("Population off")
        
    print(e,"-----------", fitness_curve[e])
    #print(fitness)
    #ex = input()
    #if ex == "y":
    #    break

best = current_population[0]
best.plot(axis, 0, 0)
print(best.get_DOF(), best.compute(), fitness_curve[e])
axis[-1].plot(range(0, EPOCH), fitness_curve)

show()
        
        
        
        
        
        
        