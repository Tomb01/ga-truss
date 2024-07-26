import numpy as np  
from random import randrange, uniform, sample
from src.plot import plot_structure
from src.operations import solve, make_sym, distance, lenght
   
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
    
    def __init__(self, contrain_nodes: np.array, elastic_modulus = 1):
        self._nodes = np.array(contrain_nodes)
        self._n_constrain = len(self._nodes)
        self.elastic_modulus = elastic_modulus
        self._reactions = np.sum(self._nodes[:,4:6])
        self._valid = False
        
    def init(self, nodes: np.array):
        self._nodes = np.vstack((self._nodes, nodes))
        n = len(self._nodes)
        self._trusses = np.zeros((5, n, n))
    
    def init_random(self, max_rep = 2, max_nodes = [0,10], area = [0.001,1]):
        max_x = np.amax(self._nodes[:,0])*max_rep
        max_y = np.amax(self._nodes[:,1])*max_rep
        
        n = randrange(max_nodes[0], max_nodes[1]) + self._n_constrain
        self._nodes.resize((n, self._nodes.shape[1]))
        
        self._trusses = np.zeros((6,n,n))
        for i in range(self._n_constrain,n):
            x = uniform(-max_x, max_x)
            y = uniform(-max_y, max_y)
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
        self._trusses[1] = make_sym(np.random.uniform(area[0], area[1], size=(n,n)))*self._trusses[0]
        
        #print(self._trusses[0])
        self.set_innovations()
        
    def init_valid(self, max_try = 100, max_rep = 2, max_nodes = [0,10], area = [0,1]):
        for i in range(0, max_try):
            #print(i)
            if i == max_try-1 and not self._valid:
                print(i)
                raise Exception("Invalid try")
            try:
                self.init_random(max_rep, max_nodes, area)
                self.solve()
                if not self._valid:
                    continue
                i = i+1
            except Exception as e:
                if e == "Invalid structure":
                    continue
                else:
                    raise e
                

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
    
    def calculate_fitness(self, yield_stess =1, node_mass_k = 1, Fos_target = 2) -> np.array:
        dl = distance(self._nodes, self._trusses)
        l = lenght(dl)
        mass = dl*l
        
        # K structure -> if structure is not valid Ks = 0
        k_structure = int(self._valid) * int(self.get_DOF()<=0)
        k_Fos = 0
        Fos_mean = 1
        #print(self._trusses[0])
        
        if self._valid:
            
            # better when Fos is like Fos_target, Fos_target - Fos is like zeros
            Fos = np.abs(np.divide(yield_stess, self._trusses[3], out=np.zeros_like(self._trusses[3]), where=self._trusses[0]!=0)) 
            self._trusses[5] = Fos
            Fos = Fos.ravel()
            conn = np.argwhere(self._trusses[0].ravel() == 1)
            #print(conn)
            Fos = Fos[conn]
            min_Fos = np.min(Fos)
            if min_Fos < 1:
                # truss collapse
                k_Fos = 0
                k_structure = 0
            else:
                k_Fos = 1
                Fos_mean = np.mean(Fos, dtype=np.float64) - Fos_target
        
        # Better mass when is minimal
        mass_sum = np.sum(mass) + len(self._nodes)*node_mass_k
        
        # K mass -> total mass
        # K Fos -> fos mean, if fos mean 
        return k_structure * 1/(mass_sum) * k_Fos*2**(1/Fos_mean)
    
    def compute(self, yield_stess =1, node_mass_k = 1, Fos_target = 2) -> float:
        self.solve()
        self.set_innovations()
        return self.calculate_fitness(yield_stess, node_mass_k, Fos_target)
    
    def plot(self, axis, row = 0, col = 0, color = "blue"):
        plot_structure(self._nodes, self._trusses, self._n_constrain, axis, row, col, color)
        