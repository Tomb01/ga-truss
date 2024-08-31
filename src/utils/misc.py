from io import TextIOWrapper
import numpy as np
from math import sqrt
from typing import Union

def make_brige_coordinate(truss_length: float, bottom_truss: int) -> np.array:
    coordinates = np.zeros((bottom_truss*2+1, 2))
    y_top = sqrt((0.5*truss_length)**2 + truss_length**2)
    for i in range(0, bottom_truss*2+1):
        coordinates[i, 0] = i*0.5*truss_length
        coordinates[i, 1] = (i%2)*y_top
        
    return coordinates

def node2table(node: np.array) -> str:
    node = node.tolist()
    str = ""
    for j in range(0, len(node)):
        node[j] = ['{:.3f}'.format(x) for x in node[j]]
        str += ";".join(node[j]) + "\n"
    return str

def newline(f: TextIOWrapper, line: Union[str, float]) -> None:
    if isinstance(line, float):
        line = "{:.9f}".format(line)
    f.write(f"{line}\n")