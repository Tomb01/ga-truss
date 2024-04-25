from src.utils.types import NodeCallable
from src.classes.node import TrussNode

# Constrain function () => (ak1,ak2)
def Constrain(constrain_func: NodeCallable, node: TrussNode) -> None:
    """
    Constrain function return the reaction coefficient to use in node equation
    If no constrain return 0,0
    """
    node.set_constrain(constrain_func)

def HingeConstrain(node: TrussNode) -> None:
    """
    Constrain both in x and y, return 1,1
    """
    Constrain(lambda : (1, 1), node)
        
def CartConstrainVertical(node: TrussNode) -> None:
    Constrain(lambda : (0, 1), node)

def CartConstrainOrizontal(node: TrussNode) -> None:
    Constrain(lambda : (1, 0), node)

def RemoveConstrain(node: TrussNode) -> None:
    delattr(node, "_constrain")
    
##### Load

def Load(node: TrussNode, Px: float, Py: float) -> None:
    node.set_load(Px, Py)
    