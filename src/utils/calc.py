from src.classes.node import TrussNode
import math
import numpy as np

def calculate_inclination(node1: TrussNode, node2: TrussNode) -> float:
    delta_x = node2.x - node1.x
    delta_y = node2.y - node1.y

    return math.atan2(delta_y, delta_x)

def get_truss_x_direction(internal_node: TrussNode, external_node: TrussNode) -> int:
    """
    In node equation truss load is positive when it goes towards the inside of the node.
    This function return 1 if direction is corret, -1 if must be inverted.
    Multiply the results and the load to get the correct value.
    """
    if internal_node.x >= external_node.x:
        return 1
    elif internal_node.x < external_node.x:
        return -1
    
def get_truss_y_direction(internal_node: TrussNode, external_node: TrussNode) -> int:
    """
    In node equation truss load is positive when it goes towards the inside of the node.
    This function return 1 if direction is corret, -1 if must be inverted.
    Multiply the results and the load to get the correct value.
    """
    if internal_node.y >= external_node.y:
        return 1
    elif internal_node.y < external_node.y:
        return -1

def add_to_matrix(arr: np.array, row: int, col:int, value: float) -> None:
    current = arr[row, col]
    arr[row, col] = current + value