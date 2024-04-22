import numpy as np
from math import sqrt

def make_brige_coordinate(truss_length: float, bottom_truss: int) -> np.array:
    coordinates = np.zeros((bottom_truss*2+1, 2))
    y_top = sqrt((0.5*truss_length)**2 + truss_length**2)
    for i in range(0, bottom_truss*2+1):
        coordinates[i, 0] = i*0.5*truss_length
        coordinates[i, 1] = (i%2)*y_top
        
    return coordinates