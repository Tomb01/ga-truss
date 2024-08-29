import numpy as np
from random import randrange, uniform
from src.operations import distance, lenght, solve, make_sym, FLOAT_MAX
from typing import Tuple

def Node(
    x: float, y: float, vx: bool = False, vy: bool = False, Px: float = 0, Pz: float = 0
):
    return np.array([x, y, 0, 0, int(vx), int(vy), 0, 0, Px, Pz, 0], np.float64)

TRUSS_DIMENSION = 7
NODE_DIMENSION = 11

class SpaceArea:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __init__(self, x1, y1, x2, y2) -> None:
        self.min_x = x1
        self.max_x = x2
        self.min_y = y1
        self.max_y = y2


class Material:
    yield_strenght: float
    density: float
    elastic_modulus: float

    def __init__(self, Re, rho, E) -> None:
        self.yield_strenght = Re
        self.density = rho
        self.elastic_modulus = E


MATERIAL_ALLUMINIUM = Material(240, 0.00000271, 72000)

class StructureParameters:
    material: Material
    node_mass_k: float
    safety_factor_yield: float
    safety_factor_buckling: float
    aggregation_radius: float
    round_digit: int
    corner: SpaceArea
    max_area: float
    min_area: float

class Structure:

    _reactions: int = 0
    _parameters: StructureParameters
    _n_constrain: int = 0
    _nodes: np.array
    _trusses: np.array
    _valid: bool = False
    _fixed_connections: bool = False

    def __init__(self, contrain_nodes: np.array, params: StructureParameters):
        self._nodes = np.array(contrain_nodes)
        self._n_constrain = len(self._nodes)
        self._parameters = params
        self._reactions = np.sum(self._nodes[:, 4:6])
        self._trusses = np.zeros((TRUSS_DIMENSION, self._n_constrain, self._n_constrain))

    def init(self, n):
        self._nodes = np.zeros((n, NODE_DIMENSION))
        self._trusses = np.zeros((TRUSS_DIMENSION, n, n))
        
    def constrain_connections(self, connections = np.array) -> None:
        if len(connections) != len(self._nodes):
            raise ValueError("connection constrained matrix wrong dimension")
        self._trusses[0] = connections
        self._trusses[1] = connections*self._trusses[1]
        self._fixed_connections = True

    def init_random(self, nodes_range, area_range):
        if self._fixed_connections and nodes_range != [0,0]:
            raise ValueError("constrained connections is only available with no nodes generation (nodes_range = [0,0])")
        
        if nodes_range[0] == nodes_range[1]:
            n = nodes_range[0]
        else:
            n = randrange(nodes_range[0], nodes_range[1]) 
        
        n = n + self._n_constrain
        self._nodes.resize((n, self._nodes.shape[1]), refcheck=False)

        allowed_space = self._parameters.corner
        
        if n!=self._n_constrain:
            # Add new random node
            for i in range(self._n_constrain, n):
                x = uniform(allowed_space.min_x, allowed_space.max_x)
                y = uniform(allowed_space.min_y, allowed_space.max_y)
                self._nodes[i] = Node(x, y)
                
        if not self._fixed_connections:
            self._trusses = np.zeros((TRUSS_DIMENSION, n, n))
            # struttura al più stabile m = 2n
            m = 2*n
            d = np.sum(np.triu(np.ones(n))) - n
            if m < d:
                triu = np.concatenate([np.ones(m), np.zeros(int(d - m))])
                np.random.shuffle(triu)
            else:
                triu = np.ones(int(d))

            adj = np.zeros((n, n))
            adj[np.triu_indices(n, 1)] = triu
            self._trusses[0] = make_sym(adj)
            
        rng = np.random.default_rng()
        random_sample = rng.exponential(scale=4, size=(n,n))
        
        b = area_range[1]
        a = area_range[0]
        area_matrix = (b - a) * random_sample + a
        
        #area_matrix = np.round(
           # np.random.uniform(area_range[0], area_range[1], size=(n, n)),
            #self._parameters.round_digit,
        #)
        self._trusses[1] = np.multiply(make_sym(np.triu(area_matrix)), self._trusses[0])
        
        self.check_connections_dimension()
        self.healing()
        self.aggregate_nodes()
        
        self._parameters.min_area = area_range[0]
        self._parameters.max_area = area_range[1]

    def check(self) -> bool:
        # Statically indeterminate if 2n < m
        free_nodes = (self._nodes[:,4]+self._nodes[:,5])==0
        edge_node = np.sum(self._trusses[0], axis=0)
        if np.all(edge_node[free_nodes] > 1): # and np.all(edge_node > 0):
            if self.get_DOF() > 0:
                self._valid = False
            else:
                self._valid = True
        else:
            self._valid = False
        
        return self._valid
        
    def healing(self) -> bool:
        if len(self._nodes) == self._n_constrain:
            return
        
        free_nodes = (self._nodes[:,4]+self._nodes[:,5])==0
        edge_node = np.sum(self._trusses[0], axis=0)
        disjoint_nodes = np.logical_and(edge_node < 2, free_nodes)
        if np.any(disjoint_nodes[0:self._n_constrain]):
            # disjoint constrained node
            return False
        else:
            # delete disjoint nodes
            self.remove_node(np.nonzero(disjoint_nodes)[0])
            self.check()
            return True
        
    def aggregate_nodes(self) -> None:
        if len(self._nodes) == self._n_constrain:
            return
        
        dl = distance(self._nodes, self._trusses, False)
        full_l = lenght(dl)
        short_trusses = full_l<self._parameters.aggregation_radius*1.41
        np.fill_diagonal(short_trusses, False)
        if np.any(short_trusses):
            row, cols = np.where(short_trusses==True)
            m = int(len(row)/2)
            if len(self._nodes) - m < self._n_constrain:
                raise ValueError("Aggregation radius is smaller than constrain min node distance")
            for j in range(0, m):
                keep = row[j]
                merge = cols[j]
                if merge < self._n_constrain:
                    merge, keep = keep, merge
                merge_connections = self._trusses[0,merge]
                current_connections = self._trusses[0,keep]
                connections = np.logical_or(merge_connections, current_connections)
                connections[keep] = 0
                self._trusses[0, keep, :] = connections
                self._trusses[0, : , keep] = connections
                
                area_merge = self._trusses[1,merge]
                area_keep = self._trusses[1,keep]
                area = np.maximum(area_merge, area_keep)
                area[keep] = 0
                self._trusses[1, keep, :] = area
                self._trusses[1, : , keep] = area
        
            to_remove = (cols > self._n_constrain)[0:m]
            remove_index = cols[0:m]
            self.remove_node(remove_index[to_remove])

    def get_node_connection(self) -> int:
        return np.sum(self._trusses[0], axis=0)
        
    def get_DOF(self) -> int:
        return 2 * len(self._nodes) - self.get_edge_count() - self._reactions

    def get_edge_count(self) -> int:
        return np.sum(self._trusses[0]) / 2

    def add_node(self, x, y, area=[0.001, 1]):
        self._nodes = np.append(self._nodes, [Node(x, y)], axis=0)
        n = len(self._nodes)
        new_adj = np.zeros((TRUSS_DIMENSION, n, n))
        new_adj[0, : (n - 1), : (n - 1)] = self._trusses[0]
        new_adj[1, : (n - 1), : (n - 1)] = self._trusses[1]

        conn = np.concatenate([np.ones(2), np.zeros(int(n - 2 - 1))])
        areas = np.round(
            np.random.uniform(area[0], area[1], size=n - 1),
            self._parameters.round_digit,
        )
        np.random.shuffle(conn)
        areas = np.multiply(areas, conn)

        new_adj[0, :, n - 1] = np.concatenate([conn, [0]])
        new_adj[1, :, n - 1] = np.concatenate([areas, [0]])

        new_adj[0] = make_sym(new_adj[0])
        new_adj[1] = make_sym(new_adj[1])

        self._trusses = new_adj

    def remove_node(self, index: np.array):
        if np.any(index < self._n_constrain):
            raise Exception("Unable to remove contrain nodes")
        index = tuple(index.tolist())

        self._nodes = np.delete(self._nodes, index, axis=0)
        self._trusses = np.delete(self._trusses, index, axis=1)
        self._trusses = np.delete(self._trusses, index, axis=2)

    def solve(self):
        self.check()
        if self._valid:
            try:
                solve(
                    self._nodes,
                    self._trusses,
                    self._parameters.material.elastic_modulus,
                )
            except Exception:
                self._valid = False
    
    def get_node_innovations(self) -> np.array:
        n = len(self._nodes)
        inn = np.empty(n, dtype=np.object_)
        crossover_radius = self._parameters.aggregation_radius
        for i in range(0, n):
            x = self._nodes[i, 0]
            y = self._nodes[i, 1]
            if crossover_radius != 0:
                x_min = x - x%crossover_radius
                x_max = x_min+crossover_radius
                y_min = y - y%crossover_radius
                y_max = y_min+crossover_radius
            else:
                x_min = x
                x_max = x
                y_min = y
                y_max = y
            inn[i] = "{:.6f}-{:.6f}-{:.6f}-{:.6f}".format(x_min, y_min, x_max, y_max)
        return inn
    
    def check_connections_dimension(self,raise_error = False):
        r = np.any(np.logical_and(np.logical_or(self._trusses[6]==0,self._trusses[1]==0), self._trusses[0]!=0))
        if r and raise_error:
            raise ValueError("Null area or lenght in some connections")
        else:
            return r
    
    def compute_stress_matrix(self) -> np.array:
        if not self._valid:
            return self._trusses[0]
        
        self.check_connections_dimension()
        # Compute allowed stress matrix
        n = len(self._nodes)
        allowed_tensile_stress = self._parameters.material.yield_strenght/ self._parameters.safety_factor_yield
        if self._parameters.safety_factor_buckling > 1:
            diameter = np.power(4*self._trusses[1]/3.14, 1/2)
            stress_cr = np.divide((3.14**2)*self._parameters.material.elastic_modulus*np.power(diameter/4,2), np.power(self._trusses[6],2)*self._parameters.safety_factor_buckling, out=np.zeros_like(self._trusses[1]), where = self._trusses[0] != 0)
            buckling = np.logical_or(stress_cr > allowed_tensile_stress, self._trusses[2]>0)
            stress_cr[buckling] = allowed_tensile_stress
        else:
            stress_cr = np.full((n, n), allowed_tensile_stress)
            stress_cr = stress_cr*self._trusses[0]

        self._trusses[5] = np.divide(
            np.abs(self._trusses[3]),
            stress_cr,
            out=np.zeros((n, n)),
            where=self._trusses[0] != 0,
        )
        
        return self._trusses[5]
           
    def calculate_fitness(self) -> float:

        if not self._valid:
            # invalid
            return 1
        
        worst_eff = np.max(self._trusses[5])
        stress_eff = 0
        if worst_eff > 1:
            # broken
            stress_eff = 10**(-worst_eff)
        else:
            eff = np.mean(np.power(self._trusses[5], 2), where=(self._trusses[0]!=0))
            stress_eff = eff
        _, mass = self.get_mass(include_density=False)
        return 1 - stress_eff/mass
        
    def is_broken(self) -> bool:
        if np.any(self._trusses[5]) > 0:
            return np.max(self._trusses[5]) > 1
        elif self._valid:
            raise ValueError("Efficency matrix not computed. Run compute method on this structure")
        
    def get_max_stess(self) -> Tuple[float, float]:
        return float(abs(np.max(self._trusses[3]))), float(abs(np.max(self._trusses[2])))
    
    def get_mass(self, include_density = True) -> Tuple[float, float]:
        truss_volume = np.sum(self._trusses[1]*self._trusses[6])
        node_volume = len(self._nodes)*self._parameters.node_mass_k
        if include_density:
            density = self._parameters.material.density
        else:
            density = 1
        return float(truss_volume*density), float((truss_volume+node_volume)*density)
    
    def get_max_dispacement(self) -> float:
        return float(np.max(self._nodes[:,2:3]))
        
    def compute(self) -> np.array:
        self.aggregate_nodes()
        self.solve()
        self.compute_stress_matrix()
        return self.calculate_fitness()
