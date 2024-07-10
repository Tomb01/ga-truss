from src.classes.genome import Phenotype, FixedNodeData
from src.classes.material import Material
from typing import List
import random
import numpy as np
from itertools import compress

class PlayGround:

    def __init__(self, max_epochs=10000, population=10):
        self.epochs = max_epochs
        self.population = population

    def play(
        self,
        geometric_constrains: List[FixedNodeData],
        material: Material,
        min_area: float,
        max_area: float,
        truss_n_min=2,
        truss_n_max=10,
    ):

        to_generate = self.population
        better_population = []
        
        for epoch in range(self.epochs):
            
            i = 0
            random_phenotipe = []
            while i < to_generate:
                try:
                    element = Phenotype.random(
                        geometric_constrains,
                        material,
                        max_area,
                        random.randint(truss_n_min, truss_n_max),
                    ) 
                    i = i+1
                    random_phenotipe.append(element)
                except ValueError as e:
                    if str(e) == "Structure not correct. Free nodes":
                        continue
                    else:
                        raise e
            
            generation = better_population + random_phenotipe
            
            score = np.array(list(map(lambda x: x.get_score(), generation)))
            mean = np.mean(score)
            better_index = np.array(score >= mean)
            better_population = list(compress(generation, better_index))
                
            extint = self.population - len(better_population)
            to_generate = extint
            print(score, extint)
            
            print("epoch {epoch}: mean score = {score}, trashed = {extint}".format(epoch=epoch, score=mean, extint = extint))
        
        return better_population

    