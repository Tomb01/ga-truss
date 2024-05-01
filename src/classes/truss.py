from typing import Tuple, Callable
from src.utils import calc
from src.utils.types import NodeCallable
from src.classes.node import TrussNode
from math import cos, sin, sqrt

class Truss:
    """
    A truss il linear element connected to two node (start node, end node).
    """

    _start_node: TrussNode
    _end_node: TrussNode
    area: float
    E: float
    _N: float
    _stress: float
    _length: float
    _inclination: float
    _absinclination: float

    def __init__(self, start_node: TrussNode, end_node: TrussNode, section_area:float, young_modulus: float) -> None:
        self._start_node = start_node
        self._end_node = end_node
        self.area = section_area
        self.E = young_modulus

        self.calculate()

    def calculate(self) -> None:
        #print("calc")
        self._inclination = calc.calculate_inclination(self._start_node, self._end_node) 
        self._length = sqrt((self._end_node.x-self._start_node.x)**2 + (self._end_node.y-self._start_node.y)**2)
        u2, v2 = self._end_node.get_displacement()
        u1, v1 = self._start_node.get_displacement()
        self._N = self.E * self.area / self._length * ((u2-u1)*cos(self._inclination)+(v2-v1)*sin(self._inclination))
        self._stress = self._N/self.area

    def get_nodes(self) -> Tuple[TrussNode, TrussNode]:
        return (self._start_node, self._end_node)

    def get_inclination(self) -> float:
        return self._inclination
    
    def get_length(self) -> float:
        return self._length
    
    def get_load(self) -> float:
        return self._N
    
    def get_stress(self) -> float:
        return self._stress