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
area = [0.001,100]

# evolution parameter
EPOCH = 100
POPULATION = 100
START_NODE_RANGE = [0,4]
C1 = 1
C3 = 1
COMPATIBILITY_THRESHOLD = 100

# Mutation
NODE_MUTATION = 0.9
AREA_MUTATION = 0.9
CONNECTION_MUTATION = 0.9
NODE_DELETE = 0
MUTATION_RATE = 0.9

# Init variables
current_population = np.empty(POPULATION, dtype=np.object_)
new_population = np.empty(POPULATION, dtype=np.object_)
fitness = np.zeros(POPULATION, dtype=float)
compatibility = np.zeros_like(fitness)
fitness_curve = np.zeros(EPOCH)
figure, axis = plt.subplots(1,3)

# Initial population -> random
for i in range(0, POPULATION):
    s = Structure(problem, elastic_modulus)
    s.init_random(max_nodes=START_NODE_RANGE, area=area)
    new_population[i] = s
    
new_population[i].plot(axis)
    
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
    fitness = fx_computation(current_population, yield_strenght)
    compatibility = fx_compatibility(current_population, C1, C3)
    
    if spices_count == 0:
        spiecies_compatibility[0] = compatibility[0]
        spieces_ids[0] = 0
        spices_count = 0
    
    # Adjusted fitness and speciation
    for i in range(0, POPULATION):
        ci = fx_sharing(compatibility[i], compatibility, COMPATIBILITY_THRESHOLD)
        fitness[i] = fitness[i]/np.sum(ci)
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
    niche_population = POPULATION//len(spieces_ids)
    new_random = POPULATION%len(spieces_ids)
    
    #print(spiece, spices_count, spieces_ids)
    
    # Selection -> filter by species
    for s in range(0, len(spieces_ids)):
        ind_index = np.where(spiece == spieces_ids[s])
        #print(ind_index, spieces_ids[s])
        spiece_population = current_population[ind_index]
        spiece_fitness = fitness[ind_index]
        
        if len(spiece_population) == 0:
            spieces_ids = np.delete(spieces_ids, [s])
            raise Exception("Extint")

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
            new_population[s*niche_population] = spiece_population[0]
            for k in range(1, niche_population):
                parents = random.sample(spice_range, 2)
                p1 = parents[0]
                p2 = parents[1]
                new_population[s*niche_population+k] = crossover(spiece_population[p1], spiece_population[p2], len(problem), spiece_fitness[p1], spiece_fitness[p2])
            #Mutate
            new_population[s*niche_population:(s+1)*niche_population] = np.vectorize(mutate)(new_population[s*niche_population:(s+1)*niche_population], MUTATION_RATE, NODE_MUTATION, AREA_MUTATION, CONNECTION_MUTATION, NODE_DELETE)
        elif len(spiece_population) == 1:
            # Only clone
            print(spieces_ids[s])
            clones = np.full(niche_population, spiece_population[0], dtype=object)
            #print(niche_population, (s+1)*niche_population-s*niche_population)
            #new_population[s*niche_population:(s+1)*niche_population] = np.vectorize(mutate)(clones, MUTATION_RATE, NODE_MUTATION, AREA_MUTATION, NODE_INSERTION, NODE_DELETE)
            #Mutate
            new_population[s*niche_population:(s+1)*niche_population] = np.vectorize(mutate)(clones, 0.01, NODE_MUTATION, AREA_MUTATION, CONNECTION_MUTATION, NODE_DELETE)
        else:
            #extint
            print("extint")
            pass
        
        new_population[s*niche_population] = spiece_population[0]
    # Fill with random (to test)
    ###print(new_population)
    for r in range(POPULATION-new_random, POPULATION):
        s = Structure(problem, elastic_modulus)
        s.init_random(max_nodes=START_NODE_RANGE, area=area)
        new_population[r] = s
         
    print(np.max(fitness), len(spieces_ids), spiece[np.argmax(fitness)])
    fitness_curve[e] = np.max(fitness)
    if fitness_curve[e] > fitness_curve[e-1]*10 and fitness_curve[e-1]!=0:
        break
    if len(spieces_ids) <= 2:
       COMPATIBILITY_THRESHOLD = COMPATIBILITY_THRESHOLD/2

amax = np.argmax(fitness)

best = current_population[amax]
best.plot(axis, 0, 1)
print(np.max(fitness), best.get_DOF())
axis[-1].plot(range(0, EPOCH), fitness_curve)

show()
        
        
        
        
        
        
        