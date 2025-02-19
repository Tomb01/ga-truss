from io import TextIOWrapper
import numpy as np
from math import sqrt
from typing import Union
from typing import Callable, Tuple

def make_brige_coordinate(truss_length: float, bottom_truss: int) -> np.array:
    """
    Generate the coordinates of a truss bridge's nodes based on its bottom truss length and number of trusses.
    
    Args:
        truss_length (float): The length of one truss unit (distance between two nodes).
        bottom_truss (int): The number of trusses in the bottom row.
    
    Returns:
        np.array: A 2D array of coordinates where each row represents a node's (x, y) position.
    """
    coordinates = np.zeros((bottom_truss*2+1, 2))
    y_top = sqrt((0.5*truss_length)**2 + truss_length**2)
    for i in range(0, bottom_truss*2+1):
        coordinates[i, 0] = i*0.5*truss_length
        coordinates[i, 1] = (i%2)*y_top
        
    return coordinates

def node2table(node: np.array) -> str:
    """
    Converts a node array into a semicolon-separated string format.
    
    Args:
        node (np.array): A 2D numpy array where each row represents a node's (x, y) coordinates.
    
    Returns:
        str: A string where each row contains the x and y coordinates of a node, formatted to three decimal places.
    """
    node = node.tolist()
    str = ""
    for j in range(0, len(node)):
        node[j] = ['{:.3f}'.format(x) for x in node[j]]
        str += ";".join(node[j]) + "\n"
    return str

def newline(f: TextIOWrapper, line: Union[str, float]) -> None:
    """
    Writes a line to a text file with the appropriate formatting.
    
    Args:
        f (TextIOWrapper): A file object to write the line to.
        line (Union[str, float]): The line to be written. It can be a string or a float value.
    """
    if isinstance(line, float):
        line = "{:.9f}".format(line)
    f.write(f"{line}\n")
    
# Define types for callable and fixed node data
NodeCallable = Callable[[], Tuple[float, float]]
FixedNodeData = Tuple[float, float, bool, bool, float, float]