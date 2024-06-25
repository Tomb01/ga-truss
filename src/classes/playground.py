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
        truss_n_min=1,
        truss_n_max=10,
    ):

        to_generate = self.population
        better_population = []
        
        for epoch in range(self.epochs):
            
            random_phenotipe = [
                Phenotype.random(
                    geometric_constrains,
                    material,
                    random.uniform(min_area, max_area),
                    random.uniform(truss_n_min, truss_n_max),
                )
                for _ in range(to_generate)
            ]
            
            generation = better_population + random_phenotipe
            
            score = np.array(list(map(lambda x: x.get_score(), generation)))
            mean = np.mean(score)
            better_index = np.array(score >= mean)
            better_population = list(compress(generation, better_index))
            print(score)
            
            extint = self.population - len(better_population)
            to_generate = extint
            
            print("epoch {epoch}: mean score = {score}".format(epoch=epoch, score=mean))

