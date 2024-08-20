from src.structure import Node, Structure, MATERIAL_ALLUMINIUM, StructureParameters, SpaceArea
import numpy as np
from src.plot import show
import matplotlib.pyplot as plt
from src.database import Database

np.set_printoptions(precision=8, suppress=True)

problem = [
    Node(0,0,True,True,0,0),
    Node(1,1,False,False,1000,0),
    Node(1,0,True,True,0,0)
]

param = StructureParameters()
param.corner = SpaceArea(0,0,3,3)
param.crossover_radius = 1
param.Fos_target = 1
param.material = MATERIAL_ALLUMINIUM
param.node_mass_k = 1
param.round_digit = 3

s = Structure(problem, param)
s.init_random([1,5], [0.1,10])
f = s.compute()

db = Database("test.db")
db.append_generation(1, f)
db.save_structure(1, s)

s1 = db.read_structure(1, 0, param)

figure, axis = plt.subplots(1,2)
s.plot(axis)
s1.plot(axis, 0, 1)

show()
