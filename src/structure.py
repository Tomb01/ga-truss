import numpy as np  
from random import randrange, uniform
    
def Node(x: float, y: float, vx: bool = False, vz: bool = False, Px: float = 0, Pz: float = 0):
    return np.array([x,y,0,0,int(vx),int(vz),0,0,Px,Pz], np.float64)

class Structure:
    
    _nodes: np.array
    _trusses: np.array
    
    def __init__(self, contrain_nodes: np.array):
        self._nodes = contrain_nodes
    
    def random(self, constrain_nodes: np.array, max_rep = 2, max_nodes = 10):
        max_x = np.amax(constrain_nodes[:,0])*max_rep
        max_y = np.amax(constrain_nodes[:,1])*max_rep
        
        n = randrange(0, max_nodes)
        self._nodes.resize((n, self._nodes.shape[1]))
        
        self._trusses = np.zeros((5,n,n))
        for i in range(0,n):
            x = uniform(-max_x, max_x)
            y = uniform(-max_y, max_y)
            self._nodes[i] = Node(x, y)
            
        