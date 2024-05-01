from typing import Callable, Tuple

NodeCallable = Callable[[], Tuple[float, float]]
FixedNodeData = Tuple[float, float, bool, bool, float, float]