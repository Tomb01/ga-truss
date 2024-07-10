import numpy as np
from src.types import Node

def connections(m: np.array, adj: np.array) -> np.array:
    #print(m)
    return np.multiply(m, np.array([adj]*(len(m.shape)-1)))

def repeat(m: np.array) -> np.array:
    return np.swapaxes([m]*(m.shape[0]), 0, 2)

def get_distance(nodes: np.array, trusses: np.array):
    all_nodes = repeat(nodes)
    return connections(np.swapaxes(all_nodes, 2, 1) - all_nodes , trusses)

def get_lenght(distances: np.array) -> np.array:
    return (distances[0]**2+distances[1]**2)**(1/2)

def get_inclination(distances: np.array) -> np.array:
    return np.arctan2(distances[1], distances[0])