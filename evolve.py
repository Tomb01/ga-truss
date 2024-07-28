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
POPULATION = 10
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
fos_fitness = np.zeros((POPULATION), dtype=float)
mass_fitness = np.zeros((POPULATION), dtype=float)
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
        
    # Selection
    # Delete invalid trusses
    filter_fos = fos_fitness == FLOAT_MAX
    filter_mass = mass_fitness == FLOAT_MAX
    to_kill = filter_fos + filter_mass
    to_kill_count = np.sum(to_kill)
    sorted_population = np.empty(to_kill_count, dtype=np.object_)
    sorted_population = current_population[to_kill == False]
    valid = len(sorted_population)
        
    # Normalize fitness
    norm_fos_fitness = np.zeros((valid), dtype=float)
    norm_mass_fitness = np.zeros((valid), dtype=float)
    mass_fitness_sum = np.sum(mass_fitness)
    fos_fitness_sum = np.sum(fos_fitness)
    norm_fos_fitness = fos_fitness/fos_fitness_sum
    norm_mass_fitness = mass_fitness/mass_fitness_sum
    
    # Normalize for multi-objective optimization
    norm_fos_fitness = norm_fos_fitness/np.min(norm_fos_fitness)
    norm_mass_fitness = norm_mass_fitness/np.min(norm_mass_fitness)
    
    # Pareto ranking
    sorted_idx = paretoRanking(norm_fos_fitness, norm_mass_fitness)
    sorted_population = current_population[sorted_idx]
    norm_fos_fitness =  norm_fos_fitness[sorted_idx]
    norm_mass_fitness = norm_mass_fitness[sorted_idx]
    
    print(norm_fos_fitness, norm_mass_fitness)
    input("here")
    
    # Crossover
    i = 0
    idx = np.arange(valid)
    while i < POPULATION and len(idx) >= 2:
        p1 = binary_turnament(idx)
        p2 = binary_turnament(idx)
        c = crossover(sorted_population[p1], sorted_population[p2], len(problem), p1, p2)
        c = mutate(c, MUTATION_NODE_POSITION, MUTATION_AREA, MUTATION_CONNECTION, MUTATION_NODE_INSERT, MUTATION_NODE_DELETE, corner, area)
        child_fitness = c.compute()
        fitness_1 = np.array([norm_fos_fitness[p1], norm_fos_fitness[p2], child_fitness[1]])
        fitness_2 = np.array([norm_mass_fitness[p1], norm_mass_fitness[p2], child_fitness[0]])
        family = np.array([current_population[p1], current_population[p2], c])
        family_sorted = paretoRanking(fitness_1, fitness_2)
        new_population[i] = family[family_sorted[0]]
        new_population[i+1] = family[family_sorted[1]]
        idx = np.delete(idx, [p1, p2])
        print(family_sorted)
        i = i+2
        
    while i < POPULATION:
        s = Structure(problem, elastic_modulus, Fos_target, node_mass, yield_strenght, corner=corner)
        s.init_random(max_nodes=START_NODE_RANGE, area=area)
        new_population[i] = s
        i = i+1
    
    #new_population[0] = sorted_population[0]    
    if np.all(fos_fitness == FLOAT_MAX):
        print("collapse")
        #print(fos_fitness, mass_fitness)
        break
    
    print(e, "---", sorted_population[0].compute(), valid)
        
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
        
        
        
        
        
        
        