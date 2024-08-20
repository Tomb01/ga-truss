import numpy as np  
from random import randrange, uniform
from src.plot import plot_structure
from src.operations import solve, make_sym, distance, lenght, FLOAT_MAX
   
def Node(x: float, y: float, vx: bool = False, vz: bool = False, Px: float = 0, Pz: float = 0):
    return np.array([x,y,0,0,int(vx),int(vz),0,0,Px,Pz,0], np.float64)

def encode_innovation(x1, x2, y1, y2, crossover_radius, round_digit) -> int:
    return "{x1_max:.6f}-{x1_min:.6f}-{y1_max:.6f}-{y1_min:.6f}-{x2_max:.6f}-{x2_min:.6f}-{y2_max:.6f}-{y2_min:.6f}".format(
        x1_max = round(x1+crossover_radius, round_digit),
        x1_min = round(x1-crossover_radius, round_digit),
        x2_max = round(x2+crossover_radius, round_digit),
        x2_min = round(x2-crossover_radius, round_digit),
        y1_max = round(y1+crossover_radius, round_digit),
        y1_min = round(y1-crossover_radius, round_digit),
        y2_max = round(y2+crossover_radius, round_digit),
        y2_min = round(y2-crossover_radius, round_digit)
    )
    #bytestring = str.encode('utf-8')
    #return int.from_bytes(bytestring, 'big')

def decode_innovations(innovation: str):
    #bytestring = innovation.to_bytes((innovation.bit_length() + 7) // 8, 'big')
    #str = bytestring.decode('utf-8')
    comp = innovation.split("-")
    return [float(comp[0]), float(comp[1]), float(comp[2]), float(comp[3]), float(comp[4])]

TRUSS_DIMENSION = 7

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
    _crossover_radius: float    
    _round_digit: int
    
    def __init__(self, contrain_nodes: np.array, elastic_modulus, Fos_target, node_mass, yield_strenght, density, corner, crossover_radius, round_digit):
        self._nodes = np.array(contrain_nodes)
        self._n_constrain = len(self._nodes)
        self.elastic_modulus = elastic_modulus
        self._Fos_target = Fos_target
        self._node_mass_k = node_mass
        self._yield_strenght = yield_strenght
        self._reactions = np.sum(self._nodes[:,4:6])
        self._valid = False
        self._corner = corner
        self.density = density
        self._crossover_radius = crossover_radius
        self._round_digit = round_digit
        
    def init(self, nodes: np.array):
        self._nodes = np.vstack((self._nodes, nodes))
        n = len(self._nodes)
        self._trusses = np.zeros((TRUSS_DIMENSION, n, n))
    
    def init_random(self, max_nodes, area):
        
        n = randrange(max_nodes[0], max_nodes[1]) + self._n_constrain
        self._nodes.resize((n, self._nodes.shape[1]))
        
        self._trusses = np.zeros((TRUSS_DIMENSION,n,n))
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
        area_matrix = np.round(np.random.uniform(area[0], area[1], size=(n,n)), self._round_digit)
        self._trusses[1] = np.multiply(make_sym(np.triu(area_matrix)),self._trusses[0])

        self.set_innovations()
        
    def check(self) -> bool:
        # Statically indeterminate if 2n < m
        edge_node = np.sum(self._trusses[0], axis=0)[0:self._n_constrain]
        if np.all(edge_node >= 1):
            if self.get_DOF()>0:
                return False
            else:
                return True
        
    def get_DOF(self) -> int:
        #print(2*len(self._nodes), self.get_edge_count(), self._reactions)
        return 2*len(self._nodes) - self.get_edge_count() - self._reactions
    
    def get_edge_count(self) -> int:
        return np.sum(self._trusses[0])/2
    
    def add_node(self, x, y, area=[0.001,1]):
        self._nodes = np.append(self._nodes, [Node(x, y)], axis=0)
        n = len(self._nodes)
        new_adj = np.zeros((TRUSS_DIMENSION,n,n))
        new_adj[0, :(n-1), :(n-1)] = self._trusses[0]
        new_adj[1, :(n-1), :(n-1)] = self._trusses[1]
        
        conn = np.concatenate([np.ones(2), np.zeros(int(n-2-1))])
        areas = np.round(np.random.uniform(area[0], area[1], size=n-1), self._round_digit)
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
                    self._innovations[i,j] = encode_innovation(self._nodes[i,0], self._nodes[j,0], self._nodes[i,1], self._nodes[j,1], self._crossover_radius, self._round_digit)
                    self._innovations[j,i] = self._innovations[i,j]

        #self._i[5] = make_sym(self._trusses[5])
        
    def get_innovations(self) -> np.array:
        return self._innovations
    
    def calculate_fitness(self) -> float:
        
        if not self._valid:
            # invalid
            return FLOAT_MAX
        
        #print(np.min(-np.abs(self._trusses[3])))
        if np.min(-np.abs(self._trusses[3])) < -self._yield_strenght * self._Fos_target:
            # collapse
            return FLOAT_MAX
        
        # Compute structural efficency matrix
        n = len(self._nodes)
        l = self._trusses[6]
        mass = np.multiply(l, self._trusses[1])
        mass = mass * self.density
        max_load = self._trusses[2]*self._Fos_target
        e = np.abs(np.divide(max_load, mass, out = np.zeros((n,n)), where=self._trusses[2]!=0))
        self._trusses[5] = e
        #print(e)

        if np.any(e[self._trusses[0]==1] == 0):
            return FLOAT_MAX
        else:
            return np.sum(mass)**2/np.sum(np.abs(max_load))
        
    def compute(self) -> np.array:
        self.solve()
        self.set_innovations()
        return self.calculate_fitness()
    
    def plot(self, axis, row = 0, col = 0, color = "blue"):
        plot_structure(self._nodes, self._trusses, self._n_constrain, axis, row, col, color)
        