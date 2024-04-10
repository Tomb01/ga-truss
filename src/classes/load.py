
from utils.types import NodeCallable
from src.classes.node import TrussNode

def Load(load_func: NodeCallable, node: TrussNode) -> None:
    node.set_load(load_func)
    
def PointVerticalLoad(node: TrussNode, load: float) -> None:
    Load(node, lambda u,v : (0.0, load))
    
def PointOrizontalLoad(node: TrussNode, load: float) -> None:
    Load(node, lambda u,v : (load, 0.0))