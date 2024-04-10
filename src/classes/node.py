from src.utils.types import NodeCallable
from typing import Tuple, Optional
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
    _constrain: NodeCallable
    _load: NodeCallable

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self._u = 0
        self._v = 0
        self._id = "n_" + str(uuid.uuid4())

    def set_displacement(self, u: float, v: float) -> None:
        self._u = u
        self._v = v

    def get_displacement(self) -> Tuple[float, float]:
        return self._u, self._v
    
    def set_load(self, load: NodeCallable) -> None:
        self._load = load
        
    def get_load(self) -> Tuple[float, float]:
        if hasattr(self, "_load"):
            return self._load(self.get_displacement())
        else:
            return (0,0)
    
    def compute_displacement(self, u: float, v:float) -> Tuple[float, float]:
        self.set_displacement(u,v)
        return self.get_displacement()

    def set_constrain(self, constrain: NodeCallable) -> None:
        self._constrain = constrain
        
    def get_constrain(self) -> Tuple[float, float]:
        if self.is_constrained():
            return self._constrain()
        else:
            return 0,0

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
        return hasattr(self, "_constrain")