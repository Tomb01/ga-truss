from src.utils.types import ConstrainCallable
from typing import Tuple 
import uuid

class TrussNode:
    """
    A node is a point where two or more truss are bonded togheter.
    It is defined by his coordinate in 2D plane (x,y)
    """

    _id: str
    x: float = 0
    y: float = 0
    _u: float = 0
    _v: float = 0
    _index: int
    _constrain: ConstrainCallable
    _constrain_y: bool = False
    _constrain_x: bool = False

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self._u = 0
        self._v = 0
        self._id = "n_" + str(uuid.uuid4())

    def set_displacement(self, u: float, v: float) -> None:
        self._u = u
        self._v = v
        if hasattr(self, "_constrain"):
            self._u, self._v = self._constrain(u,v)

    def get_displacement(self) -> Tuple[float, float]:
        return self._u, self._v
    
    def compute_displacement(self, u: float, v:float) -> Tuple[float, float]:
        self.set_displacement(u,v)
        return self.get_displacement()

    def set_constrain(self, constrain: ConstrainCallable) -> None:
        self._constrain = constrain
        self.set_displacement(self._u, self._v)
        if self._u == 0:
            self._constrain_x = True
        if self._v == 0:
            self._constrain_y = True

    def get_index(self) -> int:
        if hasattr(self, "_index"):
            return self._index
        else:
            raise ValueError("Node index must be set")

    def set_index(self, index) -> None:
        self._index = index

    def get_id(self) -> str:
        return self._id
    
    def is_constrained(self) -> bool:
        return self.is_constrained_x() and self.is_constrained_y()
        
    def is_constrained_x(self) -> bool:
        return self._constrain_x
        
    def is_constrained_y(self) -> bool:
        return self._constrain_y