class TrussNode:
    """
    A node is a point where two or more truss are bonded togheter.
    It is defined by his coordinate in 2D plane (x,y)
    """

    x: float = 0
    y: float = 0
    _u: float = 0
    _v: float = 0

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self._u = 0
        self._v = 0

    def set_displacement(self, u: float, v: float) -> None:
        self._u = u
        self._v = v