import numpy as np
from random import randrange, uniform
from src.operations import solve, make_sym, FLOAT_MAX


def Node(
    x: float, y: float, vx: bool = False, vy: bool = False, Px: float = 0, Pz: float = 0
):
    return np.array([x, y, 0, 0, int(vx), int(vy), 0, 0, Px, Pz, 0], np.float64)


def encode_innovation(x1, x2, y1, y2, crossover_radius, round_digit) -> int:
    return "{x1_max:.6f}-{x1_min:.6f}-{y1_max:.6f}-{y1_min:.6f}-{x2_max:.6f}-{x2_min:.6f}-{y2_max:.6f}-{y2_min:.6f}".format(
        x1_max=round(x1 + crossover_radius, round_digit),
        x1_min=round(x1 - crossover_radius, round_digit),
        x2_max=round(x2 + crossover_radius, round_digit),
        x2_min=round(x2 - crossover_radius, round_digit),
        y1_max=round(y1 + crossover_radius, round_digit),
        y1_min=round(y1 - crossover_radius, round_digit),
        y2_max=round(y2 + crossover_radius, round_digit),
        y2_min=round(y2 - crossover_radius, round_digit),
    )
    # bytestring = str.encode('utf-8')
    # return int.from_bytes(bytestring, 'big')


def decode_innovations(innovation: str):
    # bytestring = innovation.to_bytes((innovation.bit_length() + 7) // 8, 'big')
    # str = bytestring.decode('utf-8')
    comp = innovation.split("-")
    return [
        float(comp[0]),
        float(comp[1]),
        float(comp[2]),
        float(comp[3]),
        float(comp[4]),
    ]


TRUSS_DIMENSION = 7


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
    crossover_radius: float
    round_digit: int
    corner: SpaceArea


class Structure:

    _reactions: int
    _parameters: StructureParameters
    _n_constrain: int
    _nodes: np.array
    _trusses: np.array
    _valid: bool
    _innovations: np.array

    def __init__(self, contrain_nodes: np.array, params: StructureParameters):
        self._nodes = np.array(contrain_nodes)
        self._n_constrain = len(self._nodes)
        self._parameters = params
        self._reactions = np.sum(self._nodes[:, 4:6])
        self._valid = False

    def init(self, nodes: np.array):
        if isinstance(nodes, int):
            nodes = np.zeros((nodes, 11))
        self._nodes = np.vstack((self._nodes, nodes))
        n = len(self._nodes)
        self._trusses = np.zeros((TRUSS_DIMENSION, n, n))

    def init_random(self, nodes_range, area_range):

        if isinstance(nodes_range, int):
            n = nodes_range
        else:
            n = randrange(nodes_range[0], nodes_range[1]) 
        n = n + self._n_constrain
        self._nodes.resize((n, self._nodes.shape[1]))

        self._trusses = np.zeros((TRUSS_DIMENSION, n, n))
        allowed_space = self._parameters.corner
        for i in range(self._n_constrain, n):
            x = uniform(allowed_space.min_x, allowed_space.max_x)
            y = uniform(allowed_space.min_y, allowed_space.max_y)
            self._nodes[i] = Node(x, y)

        # struttura al più stabile m = 2n
        m = 2 * n
        d = np.sum(np.triu(np.ones(n))) - n
        if m < d:
            triu = np.concatenate([np.ones(m), np.zeros(int(d - m))])
            np.random.shuffle(triu)
        else:
            triu = np.ones(int(d))

        # print(n, m, d, len(triu))

        adj = np.zeros((n, n))
        adj[np.triu_indices(n, 1)] = triu
        self._trusses[0] = make_sym(adj)
        area_matrix = np.round(
            np.random.uniform(area_range[0], area_range[1], size=(n, n)),
            self._parameters.round_digit,
        )
        self._trusses[1] = np.multiply(make_sym(np.triu(area_matrix)), self._trusses[0])

        self.set_innovations()

    def check(self) -> bool:
        # Statically indeterminate if 2n < m
        free_nodes = (self._nodes[:,4]+self._nodes[:,5])==0
        edge_node = np.sum(self._trusses[0], axis=0)
        if np.all(edge_node[free_nodes] > 1) and np.all(edge_node > 0):
            if self.get_DOF() > 0:
                self._valid = False
            else:
                self._valid = True
        else:
            self._valid = False
        
        return self._valid
        
    def healing(self) -> bool:
        free_nodes = (self._nodes[:,4]+self._nodes[:,5])==0
        edge_node = np.sum(self._trusses[0], axis=0)
        disjoint_nodes = np.logical_and(edge_node < 2, free_nodes)
        if np.any(disjoint_nodes[0:self._n_constrain]):
            # disjoint constrained node
            return False
        else:
            # delete disjoint nodes
            #print(np.nonzero(disjoint_nodes)[0])
            self.remove_node(np.nonzero(disjoint_nodes)[0])
            #mobile_nodes = edge_node == 1
            #print(mobile_nodes)
            self.check()
            return True

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

        new_adj[0] = make_sym(np.triu(new_adj[0]))
        new_adj[1] = make_sym(np.triu(new_adj[1]))

        # print(new_adj[0], new_adj[1])
        self._trusses = new_adj

    def remove_node(self, index):
        if isinstance(index, int):
            if index < self._n_constrain:
                raise Exception("Unable to remove contrain nodes")
            index = (index)
        
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

    def set_innovations(self):
        n = len(self._nodes)
        self._innovations = np.empty((n, n), dtype=np.object_)

        for i in range(0, n):
            # self._nodes[i,10] = encode_innovation(self._nodes[i,0], 0, self._nodes[i,1], 0)
            for j in range(0, n):
                if i != j:
                    self._innovations[i, j] = encode_innovation(
                        self._nodes[i, 0],
                        self._nodes[j, 0],
                        self._nodes[i, 1],
                        self._nodes[j, 1],
                        self._parameters.crossover_radius,
                        self._parameters.round_digit,
                    )
                    self._innovations[j, i] = self._innovations[i, j]

        # self._i[5] = make_sym(self._trusses[5])

    def get_innovations(self) -> np.array:
        return self._innovations
    
    def get_node_innovations(self) -> np.array:
        n = len(self._nodes)
        inn = np.empty(n, dtype=np.object_)
        crossover_radius = self._parameters.crossover_radius
        round_digit = self._parameters.round_digit
        for i in range(0, n):
            x = self._nodes[i, 0]
            y = self._nodes[i, 1]
            inn[i] = "{x_min:.6f}-{y_min:.6f}-{x_max:.6f}-{y_max:.6f}".format(x_max=round(x + crossover_radius, round_digit),x_min=round(x - crossover_radius, round_digit),y_max=round(y + crossover_radius, round_digit),y_min=round(y - crossover_radius, round_digit),)
        return inn
        
    def calculate_fitness(self) -> float:

        if not self._valid:
            # invalid
            return 1
        
        # Compute allowed stress matrix
        diameter = np.power(4*self._trusses[1]/3.14, 1/2)
        n = len(self._nodes)

        stress_cr = np.divide((3.14**2)*self._parameters.material.elastic_modulus*np.power(diameter/4,2), np.power(self._trusses[6],2)*self._parameters.safety_factor_buckling, out=np.zeros_like(self._trusses[1]), where = self._trusses[0] != 0)
        allowed_tensile_stress = self._parameters.material.yield_strenght/ self._parameters.safety_factor_yield
        buckling = np.logical_or(stress_cr > allowed_tensile_stress, self._trusses[2]>0)
        stress_cr[buckling] = allowed_tensile_stress

        # Compute structural efficency matrix
        self._trusses[5] = np.divide(
            np.abs(self._trusses[3]),
            stress_cr,
            out=np.zeros((n, n)),
            where=self._trusses[0] != 0,
        )
        
        max_eff = np.max(self._trusses[5])
        stress_eff = 0
        if max_eff > 1:
            # broken
            stress_eff = 10**(-max_eff)
        else:
            eff = np.mean(self._trusses[5], where=(self._trusses[0]!=0))
            stress_eff = eff
        
        mass = (np.sum(self._trusses[1]*self._trusses[6])+len(self._nodes)*self._parameters.node_mass_k)
        return 1 - stress_eff/mass
        
    def is_broken(self) -> bool:
        return np.max(self._trusses[5]) > 1
        
    def compute(self) -> np.array:
        self.solve()
        self.set_innovations()
        return self.calculate_fitness()
