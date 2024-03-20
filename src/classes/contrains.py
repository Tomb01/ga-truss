from src.utils.types import ConstrainCallable
from src.classes.node import TrussNode

# Constrain function (u,v) => (u,v)
def Constrain(constrain_func: ConstrainCallable, node: TrussNode) -> None:
    node.set_constrain(constrain_func)

def HingeConstrain(node: TrussNode) -> None:
    Constrain(lambda u,v : (0.0,0.0), node)
        
def CartConstrainVertical(node: TrussNode) -> None:
    Constrain(lambda u,v : (u,0.0), node)

def CartConstrainOrizontal(node: TrussNode) -> None:
    Constrain(lambda u,v : (0.0,v), node)

def RemoveConstrain(node: TrussNode) -> None:
    delattr(node, "_constrain")