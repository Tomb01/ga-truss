import numpy as np  
from random import randrange, uniform, sample
from src.plot import plot_structure
import sys
from src.operations import solve, make_sym, distance, lenght

FLOAT_MIN = sys.float_info.min
   
def Node(x: float, y: float, vx: bool = False, vz: bool = False, Px: float = 0, Pz: float = 0):
    return np.array([x,y,0,0,int(vx),int(vz),0,0,Px,Pz,0], np.float64)

def encode_innovation(x1, x2, y1, y2) -> int:
    return "{x1:.6f}-{y1:.6f}-{x2:.6f}-{y2:.6f}".format(x1=x1, x2=x2, y1=y1, y2=y2)
    #bytestring = str.encode('utf-8')
    #return int.from_bytes(bytestring, 'big')

def decode_innovations(innovation: str):
    #bytestring = innovation.to_bytes((innovation.bit_length() + 7) // 8, 'big')
    #str = bytestring.decode('utf-8')
    comp = innovation.split("-")
    return [float(comp[0]), float(comp[1]), float(comp[2]), float(comp[3]), float(comp[4])]

class Structure:
    
    _reactions: int
    elastic_modulus: float
    _n_constrain: int
    _nodes: np.array
    _trusses: np.array
    _valid: bool
    _innovations: np.array
    _node_mass_k: float
    _Fos_target: float
    _yield_strenght: float
    
    def __init__(self, contrain_nodes: np.array, elastic_modulus, Fos_target, node_mass, yield_strenght, corner):
        self._nodes = np.array(contrain_nodes)
        self._n_constrain = len(self._nodes)
        self.elastic_modulus = elastic_modulus
        self._Fos_target = Fos_target
        self._node_mass_k = node_mass
        self._yield_strenght = yield_strenght
        self._reactions = np.sum(self._nodes[:,4:6])
        self._valid = False
        self._corner = corner
        
    def init(self, nodes: np.array):
        self._nodes = np.vstack((self._nodes, nodes))
        n = len(self._nodes)
        self._trusses = np.zeros((6, n, n))
    
    def init_random(self, max_nodes, area):
        
        n = randrange(max_nodes[0], max_nodes[1]) + self._n_constrain
        self._nodes.resize((n, self._nodes.shape[1]))
        
        self._trusses = np.zeros((6,n,n))
        for i in range(self._n_constrain,n):
            x = uniform(self._corner[0], self._corner[2])
            y = uniform(self._corner[1], self._corner[3])
            self._nodes[i] = Node(x, y)
        
        # struttura al più stabile m = 2n
        m = 2*n
        d = np.sum(np.triu(np.ones(n)))-n
        if m < d:
            triu = np.concatenate([np.ones(m), np.zeros(int(d-m))])
            np.random.shuffle(triu)
        else:
            triu = np.ones(int(d))
            
        #print(n, m, d, len(triu))
        
        adj = np.zeros((n,n))
        adj[np.triu_indices(n, 1)] = triu
        self._trusses[0] = make_sym(adj)
        self._trusses[1] = np.multiply(make_sym(np.triu(np.random.uniform(area[0], area[1], size=(n,n)))),self._trusses[0])

        self.set_innovations()
        
    def check(self) -> bool:
        # Statically indeterminate if 2n < m
        if self.get_DOF()>0:
            return False
        else:
            return True
        
    def get_DOF(self) -> int:
        return 2*len(self._nodes) - self.get_edge_count() - self._reactions
    
    def get_edge_count(self) -> int:
        return np.sum(self._trusses[0])/2
    
    def add_node(self, x, y, area=[0.001,1]):
        self._nodes = np.append(self._nodes, [Node(x, y)], axis=0)
        n = len(self._nodes)
        new_adj = np.zeros((6,n,n))
        new_adj[0, :(n-1), :(n-1)] = self._trusses[0]
        new_adj[1, :(n-1), :(n-1)] = self._trusses[1]
        
        conn = np.concatenate([np.ones(2), np.zeros(int(n-2-1))])
        areas = np.random.uniform(area[0], area[1], size=n-1)
        np.random.shuffle(conn)
        areas = np.multiply(areas, conn)
        
        new_adj[0,:,n-1] = np.concatenate([conn, [0]])
        new_adj[1,:,n-1] = np.concatenate([areas, [0]])
        
        new_adj[0] = make_sym(np.triu(new_adj[0]))
        new_adj[1] = make_sym(np.triu(new_adj[1]))
        
        #print(new_adj[0], new_adj[1])
        self._trusses = new_adj
        
    def remove_node(self, index):
        if index < self._n_constrain:
            raise Exception("Unable to remove contrain nodes")
        
        self._nodes = np.delete(self._nodes, (index), axis=0)
        self._trusses = np.delete(self._trusses, (index), axis=1)
        self._trusses = np.delete(self._trusses, (index), axis=2)
        
    def solve(self):
        if self.check():
            try:
                solve(self._nodes, self._trusses, self.elastic_modulus)
                self._valid = True
            except Exception:
                self._valid = False
        else:
            self._valid = False
            #raise ValueError("Invalid structure")
        
    def set_innovations(self):
        n = len(self._nodes)
        self._innovations = np.empty((n,n), dtype=np.object_)
        
        for i in range(0,n):
            #self._nodes[i,10] = encode_innovation(self._nodes[i,0], 0, self._nodes[i,1], 0)
            for j in range(0,n):
                if i!=j:
                    self._innovations[i,j] = encode_innovation(self._nodes[i,0], self._nodes[j,0], self._nodes[i,1], self._nodes[j,1])
                    self._innovations[j,i] = self._innovations[i,j]

        #self._i[5] = make_sym(self._trusses[5])
        
    def get_innovations(self) -> np.array:
        return self._innovations
    
    def calculate_fitness(self) -> np.array:
        dl = distance(self._nodes, self._trusses)
        l = lenght(dl)
        mass = dl*l
        
        if self._valid and self.get_DOF() <= 0:
            
            # better when Fos is like Fos_target, Fos_target - Fos is like zeros
            Fos = np.abs(np.divide(self._yield_strenght, self._trusses[3], out=np.zeros_like(self._trusses[0]), where=self._trusses[0]!=0)) 
            self._trusses[5] = Fos
            if np.isnan(Fos).any():
                return FLOAT_MIN
            Fos = Fos.ravel()
            conn = np.argwhere(self._trusses[0].ravel() == 1)
            #print(conn)
            Fos = Fos[conn]
            min_Fos = np.min(Fos)
            if min_Fos < self._Fos_target:
                # truss collapse
                return FLOAT_MIN
            else:
                k_Fos = 1
                Fos_mean = np.mean(Fos, dtype=np.float64) - self._Fos_target

            # Better mass when is minimal
            mass_sum = np.sum(mass) + len(self._nodes)*self._node_mass_k
            # K mass -> total mass
            # K Fos -> fos mean, if fos mean 
            return 2**1/(mass_sum) * k_Fos*(1/Fos_mean)
        
        return FLOAT_MIN
    
    def compute(self) -> float:
        self.solve()
        self.set_innovations()
        return self.calculate_fitness()
    
    def plot(self, axis, row = 0, col = 0, color = "blue"):
        plot_structure(self._nodes, self._trusses, self._n_constrain, axis, row, col, color)
        