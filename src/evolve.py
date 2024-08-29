from src.structure import Structure, StructureParameters
from src.genetis import crossover, mutate
import numpy as np
from src.operations import binary_turnament
from src.database import Database
from typing import Tuple
import warnings

class EvolutionParameter:
    epochs: int
    population: int
    node_range: list
    elite_ratio: float
    kill_ratio: float
    total_mutation_ratio: float
    mutation_node_position: float
    mutation_area: float
    mutation_node_delete: float
    mutation_node_insert: float
    mutation_connection: float
    enable_adj_fitness: bool

    
def evolve(problem: np.array, eparam: EvolutionParameter, sparam: StructureParameters, sample_point: int = 4, database_file = None, constrained_connections: np.array = None) -> Tuple[np.array, Structure, np.array]:

    # Override evolu parameters
    area_range = [sparam.min_area, sparam.max_area]

    # Check mutation parameter
    mutation_k = np.array([eparam.mutation_node_position, eparam.mutation_area, eparam.mutation_connection, eparam.mutation_node_delete, eparam.mutation_node_insert], dtype=np.float64)
    mutation_k_tot = np.sum(mutation_k)
    if mutation_k_tot>1:
        warnings.warn("Mutation percentage sum exceed 1. Automatic fixing will be executed")
        mutation_k = np.divide(mutation_k, mutation_k_tot, out=np.zeros_like(mutation_k), where=(mutation_k!=0))

    # Init flag and constants
    FLAG_ADJ_FITNESS = False if eparam.enable_adj_fitness == None else eparam.enable_adj_fitness
    FLAG_DATABASE = False if database_file == None else database_file
    POPULATION = eparam.population
    EPOCH = eparam.epochs
    FLAG_FIXED_CONNECTIONS = False
    if constrained_connections != None:
        warnings.warn("Fix connections activated. Only area optimization will be performed")
        FLAG_FIXED_CONNECTIONS = True

    # Init population array
    current_population = np.empty(POPULATION, dtype=np.object_)
    new_population = np.empty(POPULATION, dtype=np.object_)
    current_population = np.empty(POPULATION, dtype=np.object_)
    fitness = np.zeros((POPULATION), dtype=float)
    adj_fitness = np.zeros((POPULATION), dtype=float)
    
    # Init database
    if FLAG_DATABASE:
        db = Database(eparam.database_file)

    # Init variable for fitness saving and sample
    out_max_fitness = np.zeros(EPOCH)
    out_sample = np.empty(sample_point, dtype=np.object_)
    sample_index = 0

    # Population distribution
    elite_count = round(eparam.elite_ratio*POPULATION)
    kill_count = round(eparam.kill_ratio*POPULATION)
    crossover_count = POPULATION-kill_count-elite_count

    # Initial population -> random
    for i in range(0, POPULATION):
        s = Structure(problem, sparam)
        if FLAG_FIXED_CONNECTIONS:
            s.constrain_connections(constrained_connections)
        s.init_random(nodes_range=eparam.node_range, area_range=area_range)
        new_population[i] = s
    
    # Evolution
    for e in range(0, EPOCH):
        current_population = new_population 
        new_population = np.empty(POPULATION, dtype=np.object_)
        fitness = np.zeros((POPULATION), dtype=float)
        adj_fitness = np.zeros((POPULATION), dtype=float)
        if FLAG_DATABASE:
            db.append_generation(POPULATION,0)
        
        # Calculate fitness and adj fitness
        for i in range(0, POPULATION):
            current_fitness = current_population[i].compute()    
            fitness[i] = current_fitness
            if FLAG_ADJ_FITNESS:
                niche_population = len(np.where(fitness==current_fitness)[0])
                adj_fitness[i] = current_fitness*2**(niche_population)
            if FLAG_DATABASE:
                db.save_structure(e+1, current_population[i])
            
        # Sort by fitness variables
        if FLAG_ADJ_FITNESS:
            sorted_idx = np.argsort(adj_fitness)
        else:
            sorted_idx = np.argsort(fitness)
        current_population = current_population[sorted_idx]
        fitness = fitness[sorted_idx]
        adj_fitness = adj_fitness[sorted_idx]
    
        # Crossover
        i = 0
        while i < crossover_count:
            p1 = binary_turnament(-fitness)
            p2 = binary_turnament(-fitness)
            parent1 = current_population[p1]
            parent2 = current_population[p2]
            
            c = crossover(parent1, parent2, len(problem), -fitness[p1], -fitness[p2])
            if np.random.choice([0,1], p=[1-eparam.total_mutation_ratio, eparam.total_mutation_ratio]) == 1 or fitness[p1] == fitness[p2]:
                c = mutate(c, mutation_k, area_range)
    
            child_fitness = c.compute()
            family_fitness = np.array([fitness[p1], fitness[p2], child_fitness])
            family = np.array([parent1, parent2, c])
            family_idx = np.argsort(family_fitness)
            new_population[i] = family[family_idx[0]]
            new_population[i+1] = family[family_idx[1]]
            i = i+2
            
        # Kill and replace with fresh structures
        while i < POPULATION-elite_count:
            s = Structure(problem, sparam)
            s.init_random(nodes_range=eparam.node_range, area_range=area_range)
            new_population[i] = s
            i = i+1
            
        # Elitism
        new_population[-elite_count:] = current_population[:elite_count]
        i = i+elite_count
            
        if POPULATION > i:
            raise IndexError("Failed to generate new population {c}/{r}".format(c=i+1, r=POPULATION))
        
        # save data for return
        out_max_fitness[e] = fitness[0]
        if e > 0:
            if e%(EPOCH//(sample_point+1)) == 0:
                out_sample[sample_index] = current_population[0]
                sample_index+=1
        
    return out_max_fitness, current_population [0], out_sample, 
        
        
        
        
        
        
        