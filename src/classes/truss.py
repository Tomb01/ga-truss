from src.classes.node import TrussNode
from src.utils import calc
from math import cos, sin, sqrt

class Truss:
    """
    A truss il linear element connected to two node (start node, end node).
    """

    _start_node: TrussNode
    _end_node: TrussNode
    _A: float
    _E: float
    _N: float
    _length: float
    _inclination: float
    _absinclination: float

    def __init__(self, start_node: TrussNode, end_node: TrussNode, section_area:float, young_modulus: float) -> None:
        self._start_node = start_node
        self._end_node = end_node
        self._A = section_area
        self._E = young_modulus

        self._inclination = calc.calculate_inclination(self._start_node, self._end_node)                

        self._start_node.set_displacement(cos(self._inclination), sin(self._inclination))
        self._end_node.set_displacement(cos(self._inclination), sin(self._inclination))
        self._length = sqrt((self._end_node.x-self._start_node.x)^2 + (self._end_node.y-self._start_node.y)^2)

    def get_inclination(self) -> float:
        return self._inclination
    
    def getlength(self) -> float:
        return self._length