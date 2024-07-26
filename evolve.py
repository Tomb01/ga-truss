from src.structure import Structure, Node
from src.genetis import get_compatibility, is_shared, crossover, mutate
import numpy as np
import random
import matplotlib.pyplot as plt
from src.plot import show

# Problem parameter
problem = [
    Node(0,0,True,True,0,0),
    Node(1,1,False,False,1000,0),
    Node(1,0,True,True,0,0),
    #Node(1,0,True,True,0,0)
]

elastic_modulus = 72000
yield_strenght = 261
area = [0.0000001,50]
node_mass = 1
Fos_target = 2

# evolution parameter
EPOCH = 100
POPULATION = 100
START_NODE_RANGE = [0,4]
C1 = 1
C3 = 1
COMPATIBILITY_THRESHOLD = 10

# Mutation
MUTATION_NODE_POSITION = 0.1
MUTATION_AREA = 0.9
MUTATION_CONNECTION = 0.9
MUTATION_NODE_DELETE = 0
MUTATION_NODE_INSERT = 0
MUTATION_RATE = 0.9

# Init variables
current_population = np.empty(POPULATION, dtype=np.object_)
new_population = np.empty(POPULATION, dtype=np.object_)
adj_fitness = np.zeros(POPULATION, dtype=float)
fitness = np.zeros(POPULATION, dtype=float)
compatibility = np.zeros_like(adj_fitness)
fitness_curve = np.zeros(EPOCH)
figure, axis = plt.subplots(1,2)

# Initial population -> random
for i in range(0, POPULATION):
    s = Structure(problem, elastic_modulus, Fos_target, node_mass, yield_strenght)
    s.init_random(max_nodes=START_NODE_RANGE, area=area)
    new_population[i] = s
    
fx_computation = np.vectorize(Structure.compute)
fx_compatibility = np.vectorize(get_compatibility)
fx_sharing = np.vectorize(is_shared)

spiecies_compatibility = np.zeros(1)
spieces_ids = np.zeros(1, dtype=int)
spices_count = 0
spiece = np.zeros(POPULATION, dtype=int)
    
# Evolution
for e in range(0, EPOCH):
    current_population = new_population 
    
    # Calculate fitness value
    fitness = fx_computation(current_population)
    compatibility = fx_compatibility(current_population, C1, C3)
    fitness_curve[e] = np.max(fitness)
    
    if spices_count == 0:
        spiecies_compatibility[0] = compatibility[0]
        spieces_ids[0] = 0
        spices_count = 0
    
    # Adjusted fitness and speciation
    for i in range(0, POPULATION):
        ci = fx_sharing(compatibility[i], compatibility, COMPATIBILITY_THRESHOLD)
        adj_fitness[i] = fitness[i]/np.sum(ci)
        spiece_idx = np.argwhere(fx_sharing(spiecies_compatibility, compatibility[i], COMPATIBILITY_THRESHOLD)==1)
        if len(spiece_idx) == 0:
            # new species
            spices_count = spices_count+1
            spieces_ids = np.append(spieces_ids, [spices_count])
            spiece[i] = spices_count
            spiecies_compatibility = np.append(spiecies_compatibility, [compatibility[i]])
        elif len(spiece_idx[0]) > 1:
            raise Exception("Speciation failed")
        else:
            #print(spieces_ids, spiece_idx)
            spiece[i] = spieces_ids[spiece_idx[0,0]]
        
    valid_idx = np.where(np.in1d(spieces_ids, spiece)==True)
    spieces_ids= spieces_ids[valid_idx]
    spiecies_compatibility = spiecies_compatibility[valid_idx]
    adj_fitness_total = np.sum(adj_fitness)
    remaining_fitness = adj_fitness_total
    remaining_offspring = POPULATION
    
    # Selection -> filter by species
    for s in range(0, len(spieces_ids)):
        ind_index = np.where(spiece == spieces_ids[s])
        #print(ind_index, spieces_ids[s])
        spiece_population = current_population[ind_index]
        spiece_fitness = adj_fitness[ind_index]
        
        # Calculate new offspring number -> based on adjusted fitness ratio
        spiece_fitness_total = np.sum(spiece_fitness)
        offspring_start_index = POPULATION-remaining_offspring
        if remaining_fitness == 0:
            continue
        new_offsping_count = round(spiece_fitness_total/remaining_fitness*remaining_offspring)
        remaining_fitness = remaining_fitness - spiece_fitness_total
        remaining_offspring = remaining_offspring - new_offsping_count
        #print(offspring_start_index, new_offsping_count)
        
        if len(spiece_population) == 0 or new_offsping_count == 0:
            print("extint", spieces_ids[s])
            #spieces_ids = np.delete(spieces_ids, [spieces_ids[s]])
            continue

        # Order by fitness
        fit_index = np.argsort(-spiece_fitness)
        fit_mean = np.mean(spiece_fitness, dtype=np.float64)
        spiece_fitness = spiece_fitness[fit_index]
        spiece_population = spiece_population[fit_index]
        
        survivor_index = np.argmax(spiece_fitness <= fit_mean)
        if survivor_index > 0:
            spiece_population = spiece_population[0:survivor_index]
        #print(survivor_index, fit_mean, spiece_fitness)
        
        if len(spiece_population) > 1:
            # Crossover better individuals
            spice_range = range(0, len(spiece_population))
            for k in range(0, new_offsping_count):
                parents = random.sample(spice_range, 2)
                p1 = parents[0]
                p2 = parents[1]
                new_population[offspring_start_index+k] = crossover(spiece_population[p1], spiece_population[p2], len(problem), spiece_fitness[p1], spiece_fitness[p2])
        elif len(spiece_population) == 1:
            # Only clone
            print(spieces_ids[s])
            clones = np.full(new_offsping_count, spiece_population[0], dtype=object)
            #print(niche_population, (s+1)*niche_population-s*niche_population)
            #new_population[s*niche_population:(s+1)*niche_population] = np.vectorize(mutate)(clones, MUTATION_RATE, NODE_MUTATION, AREA_MUTATION, NODE_INSERTION, NODE_DELETE)
            #Mutate
            #new_population[offspring_start_index:(offspring_start_index+new_offsping_count)] = np.vectorize(mutate)(clones, 0.01, NODE_MUTATION, AREA_MUTATION, CONNECTION_MUTATION, NODE_DELETE)
        else:
            #extint
            print("extint")
            pass
        
        new_population[offspring_start_index] = spiece_population[0]
        
    #Mutate
    new_population = np.vectorize(mutate)(new_population, MUTATION_RATE, MUTATION_NODE_POSITION, MUTATION_AREA, MUTATION_CONNECTION, MUTATION_NODE_DELETE)
         
    print(fitness_curve[e], len(spieces_ids), spiece[np.argmax(adj_fitness)])
    #if len(spieces_ids) <= 2:
        #COMPATIBILITY_THRESHOLD = COMPATIBILITY_THRESHOLD/2

amax = np.argmax(fitness)

best = current_population[amax]
#for i in range(0, POPULATION):
    #best.plot(axis, 0, i)

best.plot(axis, 0, 0)
print(amax, best.get_DOF(), best.compute(), fitness_curve[-1])
axis[-1].plot(range(0, EPOCH), fitness_curve)

show()
        
        
        
        
        
        
        