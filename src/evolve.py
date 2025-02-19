from src.structure import Structure, StructureParameters
from src.genetis import crossover, mutate
import numpy as np
from src.operations import binary_turnament
from typing import Tuple
import warnings

class EvolutionParameter:
    """
    Parameters for an evolution run.

    **Note 1**: All attributes with the `_ratio_` suffix must be set as decimal values. For example, 0.1 corresponds to 10% of the population. 

    **Note 2**: The `total_mutation_ratio` controls the probability of **any** mutation occurring, while attributes with the `_mutation_` suffix control the probability of a specific mutation occurring once the individual is selected for mutation. 
    For instance, if `total_mutation_ratio` is set to 0.5 and `mutation_node_position` is set to 0.1, the individual has a 50% chance of mutation and, if mutated, a 10% chance that the mutation will be a random change in node position.

    Attributes:
        epochs (int): Maximum number of generations (epochs) for the evolution run.
        populations (int): The number of individuals in each generation.
        node_range (List[float]): Range `[min, max]` for the initial population and for reborn individuals' node positions.
        elite_ratio (float): Proportion of the population that is transferred directly from one generation to the next without mutation.
        kill_ratio (float): Proportion of the population that is killed off and replaced with random individuals.
        total_mutation_ratio (float): Probability of **any** mutation occurring (1 = 100% chance).
        mutation_node_position (float): Probability that a mutation will involve changing the node position.
        sort_adj_fitness (bool): If `True`, use adjusted fitness values to sort the population during selection.
        niche_radius (float): Range within which two individuals will be considered equal, and their fitness will be adjusted accordingly.
        mass_target (float): Target mass at which the algorithm will stop evolving and return results.
    """
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
    sort_adj_fitness: bool
    niche_radius: float
    mass_target: float

    
def evolve(problem: np.array, eparam: EvolutionParameter, sparam: StructureParameters, sample_point: int) -> Tuple[np.array, Structure, np.array]:
    """
    Main function to perform the evolution of the structure.

    Args:
        problem (np.array): Node array containing the constraints and loads for the problem.
        eparam (EvolutionParameter): Evolution parameters (e.g., mutation rate, selection method, etc.).
        sparam (StructureParameters): Structural parameters (e.g., material properties, initial structure, etc.).
        sample_point (int): The number of best structures to save during the evolution. 
                             For example, if set to 10 and `eparam.epochs` is 100, the function will save an image of the current best structure every 10 generations.

    Raises:
        IndexError: Raised if there is an issue with array indexing, such as exceeding array bounds during evolution steps.

    Returns:
        Tuple[np.array, Structure, np.array, np.array]:
            - An array of best fitness values for each generation.
            - The best structure found throughout the evolution.
            - An array of sampled structures (based on the `sample_point` value).
    """
    
    # Init flag and constants
    area_range = [sparam.min_area, sparam.max_area]
    FLAG_ADJ_FITNESS = False if eparam.sort_adj_fitness == None else eparam.sort_adj_fitness
    POPULATION = eparam.population
    EPOCH = eparam.epochs
        
    # Check mutation parameter
    mutation_k = np.array([eparam.mutation_node_position, eparam.mutation_area, eparam.mutation_connection, eparam.mutation_node_delete, eparam.mutation_node_insert], dtype=np.float64)
    mutation_k_tot = np.sum(mutation_k)
    if mutation_k_tot>1:
        warnings.warn("Mutation percentage sum exceed 1. Automatic fixing will be executed")
        mutation_k = np.divide(mutation_k, mutation_k_tot, out=np.zeros_like(mutation_k), where=(mutation_k!=0))
        print(mutation_k)

    # Init population array
    current_population = np.empty(POPULATION, dtype=np.object_)
    new_population = np.empty(POPULATION, dtype=np.object_)
    current_population = np.empty(POPULATION, dtype=np.object_)
    fitness = np.zeros((POPULATION), dtype=float)
    adj_fitness = np.zeros((POPULATION), dtype=float)
    niche_count = np.zeros((POPULATION), dtype=float)

    # Init variable for fitness saving and sample
    out_max_fitness = np.zeros(EPOCH)
    out_sample = np.empty(sample_point, dtype=np.object_)
    sample_index = 0

    # Population distribution
    kill_count = round(eparam.kill_ratio*POPULATION)
    elite_count = round(eparam.elite_ratio*POPULATION)
    elite_count = 1 if elite_count <= 0 else elite_count
    crossover_count = POPULATION-kill_count-elite_count

    # Initial population -> random
    for i in range(0, POPULATION):
        s = Structure(problem, sparam)
        s.init_random(nodes_range=eparam.node_range, area_range=area_range)
        new_population[i] = s
    
    # Evolution
    e = 0
    mass = eparam.mass_target*2 if eparam.mass_target!=0 else 1
    
    while e < EPOCH and mass > eparam.mass_target:
        # Init current generation array
        current_population = new_population 
        new_population = np.empty(POPULATION, dtype=np.object_)
        fitness = np.zeros((POPULATION), dtype=float)
        adj_fitness = np.zeros((POPULATION), dtype=float)
        # Calculate fitness and adj fitness
        for i in range(0, POPULATION):
            current_fitness = round(current_population[i].compute(),6) 
            fitness[i] = current_fitness
            if FLAG_ADJ_FITNESS:
                niche_population = len(np.nonzero((fitness[0:i]<=(current_fitness+eparam.niche_radius))*(fitness[0:i]>=(current_fitness-eparam.niche_radius)))[0])
                niche_count[i] = niche_population+1
                adj_fitness[i] = current_fitness*(niche_count[i])
            
        # Sort by fitness variables
        if FLAG_ADJ_FITNESS:
            sorted_idx = np.argsort(adj_fitness)
        else:
            sorted_idx = np.argsort(fitness)
            adj_fitness = fitness
        current_population = current_population[sorted_idx]
        fitness = fitness[sorted_idx]
        adj_fitness = adj_fitness[sorted_idx]
    
        # Crossover
        i = 0
        while i < crossover_count:
            p1 = binary_turnament(-adj_fitness)
            p2 = binary_turnament(-adj_fitness)
            parent1 = current_population[p1]
            parent2 = current_population[p2]
            
            c = crossover(parent1, parent2, -adj_fitness[p1], -adj_fitness[p2])
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
            
        # population check
        if POPULATION > i:
            raise IndexError("Failed to generate new population {c}/{r}".format(c=i+1, r=POPULATION))
        
        mass = current_population[0].get_mass()[0]
        mass = eparam.mass_target*2 if mass==0 else mass
        
        # save data for return
        out_max_fitness[e] = fitness[0]
        if e > 0:
            if e%(EPOCH//(sample_point)) == 0:
                out_sample[sample_index] = current_population[0]
                sample_index+=1
        e += 1
        
    # Reshape fitness for output
    out_max_fitness = out_max_fitness[0:e]
    
    return out_max_fitness, current_population[0], out_sample
        
        
        
        
        
        
        