from src.classes.node import TrussNode
import math

def calculate_inclination(node1: TrussNode, node2: TrussNode) -> float:
    delta_x = node2.x - node1.x
    delta_y = node2.y - node1.y

    return math.atan2(delta_y, delta_x)