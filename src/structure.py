import numpy as np  
from random import randrange, uniform
from src.plot import plot_structure
from src.operations import solve
    
def Node(x: float, y: float, vx: bool = False, vz: bool = False, Px: float = 0, Pz: float = 0):
    return np.array([x,y,0,0,int(vx),int(vz),0,0,Px,Pz], np.float64)

class Structure:
    
    elastic_modulus: float
    _n_constrain: int
    _nodes: np.array
    _trusses: np.array
    
    def __init__(self, contrain_nodes: np.array, elastic_modulus = 1):
        self._nodes = np.array(contrain_nodes)
        self._n_constrain = len(self._nodes)
        self.elastic_modulus = elastic_modulus
    
    def init_random(self, max_rep = 2, max_nodes = 10, area = [0,1]):
        max_x = np.amax(self._nodes[:,0])*max_rep
        max_y = np.amax(self._nodes[:,1])*max_rep
        
        n = randrange(0, max_nodes) + self._n_constrain
        self._nodes.resize((n, self._nodes.shape[1]))
        
        self._trusses = np.zeros((5,n,n))
        for i in range(self._n_constrain,n):
            x = uniform(-max_x, max_x)
            y = uniform(-max_y, max_y)
            self._nodes[i] = Node(x, y)
            
        rdn_adj = np.zeros((2,n,n))
        rdn_adj[0] = np.triu(np.random.choice([0,1], size=(n,n)))
        rdn_adj[1] = np.random.uniform(area[0], area[1], size=(n,n))

        self._trusses[0] = rdn_adj[0] + rdn_adj[0].T - np.diag(np.diag(rdn_adj[0]))
        np.fill_diagonal(self._trusses[0], 0) # Delete recursive connections
        self._trusses[1] = rdn_adj[1]*self._trusses[0]
        
    def check(self) -> bool:
        # Check invalid structure -> DOF > 0
        n = len(self._nodes)
        node_edge_count = np.sum(self._trusses[0], axis=1)
        min_edge = 3 if self._n_constrain == n else np.min(node_edge_count[self._n_constrain:])
        total_edge = np.sum(self._trusses[0])/2
        print(self._trusses[0], total_edge)
        if min_edge < 2 or total_edge <= n:
            return False
        else:
            return True
        
    def solve(self):
        if self.check():
            solve(self._nodes, self._trusses, self.elastic_modulus)
        else:
            raise ValueError("Invalid structure")
        #solve(self._nodes, self._trusses, self.elastic_modulus)
    
    def plot(self):
        plot_structure(self._nodes, self._trusses, self.elastic_modulus)
        