from src.operations import get_distance, get_lenght, get_inclination
import numpy as np

nodes = np.array([[0,0], [1,0], [1,1]])
trusses = np.array([[0,0,1], [0,0,1], [1,1,0]])

#print(repeat(nodes))
distances = get_distance(nodes, trusses)
print(get_inclination(distances))