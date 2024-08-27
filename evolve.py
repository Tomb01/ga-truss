import random
from src.structure import Structure, Node, StructureParameters, MATERIAL_ALLUMINIUM, SpaceArea
from src.genetis import sharing, crossover, mutate
import numpy as np
import matplotlib.pyplot as plt
from src.plot import plot_structure, show
from src.operations import binary_turnament
from src.database import Database
import datetime

# Problem parameter
problem = [
    Node(0,0,True,True,0,0),
    Node(1,2,False,False,1e3,0),
    Node(1,0,True,True,0,0)
]

param = StructureParameters()
param.corner = SpaceArea(-2,0,4,4)
param.crossover_radius = 0.2
param.safety_factor_yield = 1
param.safety_factor_buckling = 1.5
param.material = MATERIAL_ALLUMINIUM
param.node_mass_k = 1
param.round_digit = 2

area_range = [1.5,2]

# evolution parameter
EPOCH = 20
POPULATION = 10
START_NODE_RANGE = [0,20]
ELITE_RATIO = 0.1
MUTATION_RATIO = 0.2
KILL_RATIO = 0.1
NICHE_RADIUS = 0.01
CROSSOVER_RADIUS = 0.1

# Mutation
MUTATION_NODE_POSITION = 2
MUTATION_AREA = 1
MUTATION_CONNECTION = 1
MUTATION_NODE_DELETE = 2
MUTATION_NODE_INSERT = 1
# Fix mutation
mutation_k = np.array([MUTATION_NODE_POSITION,MUTATION_AREA,MUTATION_CONNECTION,MUTATION_NODE_DELETE,MUTATION_NODE_INSERT], dtype=np.float64)
mutation_k_tot = np.sum(mutation_k)
mutation_k = np.divide(mutation_k, mutation_k_tot, out=np.zeros_like(mutation_k), where=(mutation_k!=0))

# Init variables
current_population = np.empty(POPULATION, dtype=np.object_)
new_population = np.empty(POPULATION, dtype=np.object_)
sorted_population = np.empty(POPULATION, dtype=np.object_)
fitness = np.zeros((POPULATION), dtype=float)
adj_fitness = np.zeros((POPULATION), dtype=float)
db = Database(".trash/"+datetime.date.today().strftime('%Y%m%d%H%M%S') +".db")

fitness_curve = np.zeros(EPOCH)
"""figure, axis = plt.subplots(1,5)
figure.set_figheight(5)
figure.set_figwidth(15)"""
best = np.empty(4, dtype=np.object_)
b_count = 0

# New population distribution
elite_count = round(ELITE_RATIO*POPULATION)
kill_count = round(KILL_RATIO*POPULATION)
crossover_count = POPULATION-kill_count-elite_count

# Initial population -> random
for i in range(0, POPULATION):
    s = Structure(problem, param)
    #n = i%START_NODE_RANGE[1]
    s.init_random(nodes_range=START_NODE_RANGE, area_range=area_range)
    new_population[i] = s
    
# Evolution
for e in range(0, EPOCH):
    current_population = new_population 
    new_population = np.empty(POPULATION, dtype=np.object_)
    fitness = np.zeros((POPULATION), dtype=float)
    adj_fitness = np.zeros((POPULATION), dtype=float)
    #db.append_generation(POPULATION,0)
    
    # Calculate fitness and adj fitness
    for i in range(0, POPULATION):
        current_fitness = current_population[i].compute()    
        niche_population = len(np.where(fitness==current_fitness)[0])
        fitness[i] = current_fitness
        adj_fitness[i] = current_fitness*2**(niche_population)
        #db.save_structure(e+1, current_population[i])
            
    # Crossover
    i = 0
    idx = np.arange(POPULATION)
    sorted_idx = np.argsort(adj_fitness)
    #print(sorted_idx)
    sorted_population = current_population[sorted_idx]
    fitness = fitness[sorted_idx]
    adj_fitness = adj_fitness[sorted_idx]
    
    #print(fitness)
    
    while i < POPULATION-elite_count-kill_count:
        p1 = binary_turnament(-adj_fitness)
        p2 = binary_turnament(-adj_fitness)
        parent1 = sorted_population[p1]
        parent2 = sorted_population[p2]
        
        c = crossover(parent1, parent2, len(problem), -fitness[p1], -fitness[p2])
        if np.random.choice([0,1], p=[1-MUTATION_RATIO, MUTATION_RATIO]) == 1 and fitness[p1]==fitness[p2]:
            #print("mutate")
            c = mutate(c, mutation_k, area_range)
        child_fitness = c.compute()
        family_fitness = np.array([fitness[p1], fitness[p2], child_fitness])
        family = np.array([parent1, parent2, c])
        family_idx = np.argsort(family_fitness)
        new_population[i] = family[family_idx[0]]
        new_population[i+1] = family[family_idx[1]]
        i = i+2
            
    #new_population[-2] = sorted_population[0]
    while i < POPULATION-elite_count:
        s = Structure(problem, param)
        s.init_random(nodes_range=START_NODE_RANGE, area_range=area_range)
        new_population[i] = s
        i = i+1
        
    new_population[-elite_count:] = sorted_population[:elite_count]
    
    fitness_curve[e] = fitness[0]
    """if e > 0:
        if e%(EPOCH//4) == 0:
            best[b_count] = sorted_population[0]
            b_count = b_count+1"""
    
    print(e, "---", sorted_population[0].compute())
    """if e==EPOCH-1: 
        figure1, axis1 = plt.subplots(1,POPULATION)
        for j in range(0, POPULATION):
            plot_structure(sorted_population[j], figure1, axis1[j], annotation=False, area=area_range)
        
        figure1.set_figheight(5)
        figure1.set_figwidth(15)
        plt.show(block=True)"""

best[-1] = sorted_population[0]

figure, axis = plt.subplots(1,2)
figure.set_figheight(5)
figure.set_figwidth(15)
axis[-1].plot(range(0, EPOCH), fitness_curve)
plot_structure(sorted_population[0], figure, axis[0], annotation=False, area=area_range)
print(sorted_population[0].is_broken(), sorted_population[0].check())
show()
        
        
        
        
        
        
        