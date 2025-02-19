import warnings
import numpy as np
from random import randrange
from src.operations import distance, length, solve, make_sym, get_signed_max
from typing import Tuple

def Node(
    x: float, y: float, vx: bool = False, vy: bool = False, Px: float = 0, Pz: float = 0
):
    """
    Structure node object

    Args:
        x (float): x coordinate
        y (float): y coordinate
        vx (bool, optional): movements locked in x direction. Defaults to False.
        vy (bool, optional): movements locked in y direction. Defaults to False.
        Px (float, optional): load in x direction. Defaults to 0.
        Pz (float, optional): load in y direction. Defaults to 0.
    """
    return np.array([x, y, 0, 0, int(vx), int(vy), 0, 0, Px, Pz, 0], np.float64)

TRUSS_MATRIX_DIMENSION = 7
TRUSS_ARRAY_DIMENSION = 11

class SpaceArea:
    """
    Represents the problem space for a structure, defined by a rectangular area.

    This class encapsulates the boundaries of a 2D space, which can be used to define the 
    possible positions or constraints for structural elements (e.g., nodes, trusses) within a 
    given space. The space is specified by the minimum and maximum x and y coordinates.

    Attributes:
        min_x (float): The minimum x-coordinate of the space (left boundary).
        min_y (float): The minimum y-coordinate of the space (bottom boundary).
        max_x (float): The maximum x-coordinate of the space (right boundary).
        max_y (float): The maximum y-coordinate of the space (top boundary).

    """
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __init__(self, min_x: float, min_y: float, max_x:float, max_y:float) -> None:
        """
        Initializes the space area with the given boundaries.

        Args:
            min_x (float): The minimum x-coordinate of the space (left boundary).
            min_y (float): The minimum y-coordinate of the space (bottom boundary).
            max_x (float): The maximum x-coordinate of the space (right boundary).
            max_y (float): The maximum y-coordinate of the space (top boundary).
        """
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y


class Material:
    """
    Represents a material with physical properties used in structural analysis.

    Attributes:
        yield_strength (float): The yield strength of the material, i.e., the stress at which 
                                 it begins to deform plastically.
        density (float): The density of the material, which determines its mass per unit volume.
        elastic_modulus (float): The elastic modulus (also called Young's modulus) of the material, 
                                  which measures its stiffness.

    """
    yield_strenght: float
    density: float
    elastic_modulus: float

    def __init__(self, yield_strenght: float, density: float, elastic_modulus: float) -> None:
        """
        Initializes the material with the given physical properties.

        Args:
            yield_strength (float): The yield strength of the material, i.e., the stress at which 
                                    it begins to deform plastically.
            density (float): The density of the material, which determines its mass per unit volume.
            elastic_modulus (float): The elastic modulus (also called Young's modulus) of the material, 
                                    which measures its stiffness.
        """
        self.yield_strenght = yield_strenght
        self.density = density
        self.elastic_modulus = elastic_modulus

# Example material (used in most test cases)
MATERIAL_ALLUMINIUM = Material(240, 0.00000271, 72000)

class StructureParameters:
    """
    Represents the parameters that define the properties and constraints of a structure.

    Attributes:
        material (Material): The material used in the structure.
        node_mass_k (float): The volume of a single node.
        safety_factor_yield (float): The safety factor applied to the material's yield strength.
        safety_factor_buckling (float): The safety factor applied to the buckling strength.
        aggregation_radius (float): The radius within which nodes are aggregated or considered as 
                                    part of the same group.
        round_digit (int): The number of decimal places to round values during calculations.
        corner (SpaceArea): The space area that defines the constraints for the structure's boundary 
                            in the 2D plane.
        max_area (float): The maximum allowable area for any truss cross-section.
        min_area (float): The minimum allowable area for any truss cross-section.
        max_displacement (float): The maximum allowable displacement for the structure's nodes or components.
        max_length (float): The maximum allowable length for any truss in the structure.

    """
    material: Material
    node_volume: float
    safety_factor_yield: float
    safety_factor_buckling: float
    aggregation_radius: float
    round_digit: int
    corner: SpaceArea
    max_area: float
    min_area: float
    max_displacement: float
    max_length: float

class Structure:
    """
    Represents a a structure made with nodes and trusses

    The `Structure` class encapsulates the key components and properties of a structural system,
    such as nodes, trusses, applied forces, boundary conditions (constraints), and various 
    parameters used to define the structure's behavior during analysis.

    Attributes:
        _reactions (int): The number of reaction forces in the structure, initially set to 0.
        _parameters (StructureParameters): The parameters that define the structural system.
        _n_constrain (int): The number of constraints applied to the structure, initially set to 0.
        _nodes (np.array): Array of nodes in the structure.
        _trusses (np.array): Array of trusses (elements connecting nodes).
        _valid (bool): A flag indicating whether the structure is valid.
        _fixed_connections (bool): A flag indicating whether the structure has fixed connections. In this case the evolution works only on the node position and    trusses cross-sections
    """

    _reactions: int = 0
    _parameters: StructureParameters
    _n_constrain: int = 0
    _nodes: np.array
    _trusses: np.array
    _valid: bool = False
    _fixed_connections: bool = False

    def __init__(self, problem_nodes: np.array, params: StructureParameters):
        """
        Initializes the structure with constrain and parameters

        Args:
            contrain_nodes (np.array): The nodes from which the structure must be generated.
            params (StructureParameters): The structure parameters
        """
        self._nodes = np.array(problem_nodes)
        self._n_constrain = len(self._nodes)
        self._parameters = params
        self._reactions = np.sum(self._nodes[:, 4:6])
        self._trusses = np.zeros((TRUSS_MATRIX_DIMENSION, self._n_constrain, self._n_constrain))

    def init(self, n):
        """
        Initializes an empty structure of n nodes

        Args:
            n (_type_): Number of nodes in the structure
        """
        self._nodes = np.zeros((n, TRUSS_ARRAY_DIMENSION))
        self._trusses = np.zeros((TRUSS_MATRIX_DIMENSION, n, n))
        
    def _check_dimension(self, trusses: np.array, raise_error = True) -> bool:
        """
        Check if the number of nodes and trusses is consistent.

        Args:
            trusses (np.array): trusses matrix
            raise_error (bool, optional): flag to activate error raising. Defaults to True.

        Raises:
            ValueError: Raise error in case of wrong node or trusses count

        Returns:
            bool: True if the check is passed, False if not.
        """
        if len(self._nodes) != trusses.shape[0]:
            if raise_error:
                raise ValueError("Invalid truss dimension")
            return False
        else:
            return True
        
    def set_connections(self, new_connections: np.array) -> None:
        """
        Set the connections matrix

        Args:
            new_connections (np.array): New connections matrix
        """
        if self._check_dimension(new_connections):
            self._trusses[0] = np.copy(new_connections)
            
    def get_connections(self) -> np.array:
        """
        Get a copy of the connection matrix

        Returns:
            np.array: Copy of the connection matrix
        """
        return np.copy(self._trusses[0])
    
    def set_nodes(self, new_nodes: np.array) -> None:
        """
        Set the structure nodes array

        Args:
            new_nodes (np.array): new nodes array
        """
        self._nodes = new_nodes
        
    def get_nodes(self) -> np.array:
        """
        Get a copy of the nodes array

        Returns:
            np.array: Copy of the nodes array
        """
        return np.copy(self._nodes)
    
    def get_efficency(self) -> np.array:
        """
        Get a copy of the structural trusses structural efficency matrix

        Returns:
            np.array: Trusses structural efficency matrix
        """
        return np.copy(self._trusses[5])
            
    def set_areas(self, new_area: np.array) -> None:
        """
        Set the structure areas matrix

        Args:
            new_area (np.array): new areas matrix
        """
        if self._check_dimension(new_area):
            self._trusses[1] = np.copy(new_area)*self._trusses[0]
            
    def get_areas(self) -> np.array:
        """
        Get a copy of the areas matrix

        Returns:
            np.array: Copy of the areas matrix
        """
        return np.copy(self._trusses[1])
        
    def constrain_connections(self, connections: np.array) -> None:
        """
        Set a constrained connections matrix. In this case the evolution works only on the node position and trusses cross-sections

        Args:
            connections (_type_, optional): Constrained connections matrix

        Raises:
            ValueError: If the connections matrix and node array dimension are not consistent
        """
        if len(connections) != len(self._nodes):
            raise ValueError("Connection constrained matrix wrong dimension")
        self._trusses[0] = connections
        self._trusses[1] = connections*self._trusses[1]
        self._fixed_connections = True

    def init_random(self, nodes_range: np.array, area_range = None):
        """
        Init random structure from the problem constrain

        Args:
            nodes_range (np.array): The node count range [min,max]
            area_range (np.array, optional): Cross-section area range value, if none the value are extracted from the structure parameters. Defaults to None.

        Raises:
            ValueError: _description_
        """
        if self._fixed_connections and nodes_range != [0,0]:
            raise ValueError("Constrained connections is only available with no nodes generation (nodes_range = [0,0])")
        
        if area_range == None:
            area_range = [self._parameters.min_area, self._parameters.max_area]
        
        if nodes_range[0] == nodes_range[1]:
            n = nodes_range[0]
        else:
            n = randrange(nodes_range[0], nodes_range[1]) 
        
        n = n + self._n_constrain
        self._nodes.resize((n, self._nodes.shape[1]), refcheck=False)

        allowed_space = self._parameters.corner
        
        if n!=self._n_constrain:
            # Add new random node
            self._nodes[self._n_constrain:,0] = np.random.uniform(allowed_space.min_x, allowed_space.max_x, size=(n-self._n_constrain))
            self._nodes[self._n_constrain:,1] = np.random.uniform(allowed_space.min_y, allowed_space.max_y, size=(n-self._n_constrain))
                
        if not self._fixed_connections:
            self._trusses = np.zeros((TRUSS_MATRIX_DIMENSION, n, n))
            l = length(distance(self._nodes, self._trusses[0], False))
            # Connect all nodes which distance is lower than max_length
            self._trusses[0, l < self._parameters.max_length*0.99] = 1
            conn = self.get_trusses_count()
            m = int(2*n-3-conn)
            if m > 0:
                # Fill remaining edge
                np.fill_diagonal(self._trusses[0], 1)
                adj = self.get_connections()
                rows, cols = (adj == 0).nonzero()
                # https://stackoverflow.com/a/4602224
                p = np.random.permutation(len(rows))
                rows = rows[p] 
                cols = cols[p]
                adj[rows[0:m], cols[0:m]] = 1
                self._trusses[0] = adj
                
            np.fill_diagonal(self._trusses[0], 0) 
            self._trusses[0] = make_sym(self._trusses[0])
        
        # Generate random area values
        a_min = area_range[0]
        area_matrix = np.round(
            np.random.uniform(a_min, area_range[1], size=(n,n)),
            self._parameters.round_digit,
        )
        
        # Correct area values
        area_matrix[area_matrix < self._parameters.min_area] = self._parameters.min_area
        area_matrix[area_matrix > self._parameters.max_area] = self._parameters.max_area
        self._trusses[1] = np.multiply(make_sym(np.triu(area_matrix)), self._trusses[0])
        
        # Check structure
        self.check_connections_dimension()
        self.delete_disjoint_nodes()
        self.aggregate_nodes()
        self.has_collinear_edge(True)
        
        # Overwrite area parameters
        self._parameters.min_area = area_range[0]
        self._parameters.max_area = area_range[1]
        
    def has_collinear_edge(self, delete = False) -> bool:
        """
        Detect collinear edge in the structure

        Args:
            delete (bool, optional): Delete collinear edge if founded. Defaults to False.

        Returns:
            bool: True if there are collinear edge, False if not
        """
        d = distance(self._nodes, self._trusses, conn=False)
        l = length(d)
        x = self._nodes[:,0]
        y = self._nodes[:,1]
        n = len(self._nodes)
        for i in range(0, n):
            node = self._nodes[i]
            dnode = ((node[1]-y)**2+(node[0]-x)**2)**(1/2)
            dnode_m = np.resize(dnode, (n,n))
            dsum = dnode_m+dnode_m.T
            dsum[i] = 0
            dsum[:, i] = 0
            collinear = np.nonzero(np.isclose(dsum, l, rtol=0.003)*self._trusses[0])
            if delete:
                # delete collinear edge
                self._trusses[0, collinear[0], collinear[1]] = 0
            else:
                if len(collinear[0]) > 0:
                    return True
        return False
    
    def _free_nodes(self) -> np.array:
        """
        Find non-connected (free) nodes in the structure

        Returns:
            np.array: a boolean array that represent if the node is free or not
        """
        return (self._nodes[:,4]+self._nodes[:,5]) == 0
    
    def _check(self) -> bool:
        """
        Check if the structure is valid

        Returns:
            bool: Return True if the structure is valid
        """
        # Statically indeterminate if 2n < m
        free_nodes = self._free_nodes()
        edge_node = np.sum(self._trusses[0], axis=0)
        edge_node = edge_node + ~free_nodes
        dof = self.get_dof()
        
        if np.any(edge_node < 2):
            return False
        
        if dof > 0:
            return False
        else:
            # Check if there is isostatic movable appendix
            over_conn = self._trusses[0, ~free_nodes]
            o = over_conn[:,~free_nodes].sum()/2
            if dof == 0 and o != 0:
                return False
        
        if self.has_collinear_edge():
            return False
        
        return True

    def is_valid(self) -> bool:
        """
        Check and return the structure status

        Returns:
            bool: Structure status (True if valid)
        """
        self._valid = self._check()
        return self._valid
        
    def delete_disjoint_nodes(self) -> bool:
        """
        Delete disjoint node (if are not problem node)

        Returns:
            bool: True if all non-connected nodes are deleted, False if there are problem node disconnected
        """
        if len(self._nodes) <= self._n_constrain:
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
            return True  
        
    def aggregate_nodes(self) -> None:
        """
        Aggregate nodes if their distance are too small
        """
        if len(self._nodes) <= self._n_constrain:
            return
        
        dl = distance(self._nodes, self._trusses, False)
        full_l = length(dl)  
        # Compare node distances and aggregation radius. The 1.41 parameter is to evaluate the diagonal distance
        short_trusses = full_l<self._parameters.aggregation_radius*1.41
        np.fill_diagonal(short_trusses, False)
        if np.any(short_trusses):
            row, cols = np.where(short_trusses==True)
            m = int(len(row)/2)
            if len(self._nodes) - m < self._n_constrain:
                warnings.warn("Aggregation radius is smaller than constrain min node distance")

            for j in range(0, m):
                # Resize structure matrix in order to aggregate nodes
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
        
    def get_dof(self) -> int:
        """
        Get degree of freedom count

        Returns:
            int: degree of freedom count
        """
        return 2 * len(self._nodes) - self.get_trusses_count() - self._reactions

    def get_trusses_count(self) -> int:
        """
        Get total number of trusses

        Returns:
            int: Total number of trusses
        """
        return np.sum(self._trusses[0]) / 2

    def add_node(self, x: float, y: float, area=[0.001, 1]):
        """
        Add node to the structure at the given coordinate. When a new node is created it will be connected with other two random node in order to mantain the structure validity

        Args:
            x (float): new node x-coordinate
            y (list): new node y-coordinate
            area (list, optional): Area range for the generated trusses. Defaults to [0.001, 1].
        """
        self._nodes = np.append(self._nodes, [Node(x, y)], axis=0)
        n = len(self._nodes)
        new_adj = np.zeros((TRUSS_MATRIX_DIMENSION, n, n))
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

    def remove_node(self, index: int):
        """
        Remove node from the structure and all trusses connected to it

        Args:
            index (int): the index of the npde 

        Raises:
            Exception: Raise an exception if the node to be deleted is constrained by the problem.
        """
        if np.any(index < self._n_constrain):
            raise Exception("Unable to remove contrain nodes")
        index = tuple(index.tolist())

        self._nodes = np.delete(self._nodes, index, axis=0)
        self._trusses = np.delete(self._trusses, index, axis=1)
        self._trusses = np.delete(self._trusses, index, axis=2)

    def solve(self):
        """
        Solve the structure equilibrium equation
        """
        self.is_valid()
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
        """
        Calculate nodes innovations

        Returns:
            np.array: Node innovation matrix
        """
        n = len(self._nodes)
        inn = np.empty(n, dtype=np.object_)
        crossover_radius = self._parameters.aggregation_radius
        for i in range(0, n):
            x = self._nodes[i, 0]
            y = self._nodes[i, 1]
            if crossover_radius != 0:
                x_min = x
                x_max = x 
                y_min = y 
                y_max = y 
            else:
                x_min = x
                x_max = x
                y_min = y
                y_max = y
            inn[i] = "{:.6f}-{:.6f}-{:.6f}-{:.6f}".format(x_min, y_min, x_max, y_max)
        return inn
    
    def check_connections_dimension(self) -> np.array:
        """
        Check if there are trusses with zero areas or length
        
        Raises:
            ValueError: If a zero dimension truss is found

        Returns:
            np.array: Return 
        """
        r = np.any(np.logical_and(self._trusses[1]==0, self._trusses[0]!=0))
        if r:
            raise ValueError("Null area or lenght in some connections")
    
    def compute_stress(self) -> np.array:
        """
        Compute stress matrix based on the values evaluted solving the equilibrium equations

        Returns:
            np.array: Stress matrix
        """
        if not self._valid:
            return self._trusses[0]
        
        self.check_connections_dimension()
        # Compute allowed stress matrix
        n = len(self._nodes)
        allowed_tensile_stress = self._parameters.material.yield_strenght/ self._parameters.safety_factor_yield
        if self._parameters.safety_factor_buckling > 1:
            diameter = self.get_diameters()
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
           
    def get_fitness(self) -> float:
        """
        Get the structure fitness

        Returns:
            float: Structure fitness
        """

        if not self._valid:
            # invalid
            return 1
        
        stress_efficency = 0
        max_displacement = self.get_max_dispacement()
        worst_efficency = np.max(self._trusses[5])

        total_mass, _, mass_matrix = self.get_mass(include_density=False)
        if max_displacement > 0.99*self._parameters.max_displacement or worst_efficency > 0.99:
            stress_efficency = 10**(-max(max_displacement/self._parameters.max_displacement, worst_efficency))
        else:
            mass_index = mass_matrix/total_mass
            efficency = np.mean(self._trusses[5]*mass_index, where=(self._trusses[0]!=0))
            stress_efficency = efficency + 0.1
           
        return 1 - stress_efficency
        
    def is_broken(self) -> bool:
        """
        Check if the structure can carry the imposed load

        Raises:
            ValueError: If the efficency matrix was not evaluated with the solve function

        Returns:
            bool: False if the structure cannot carry external load
        """
        if np.any(self._trusses[5]) > 0:
            return np.max(self._trusses[5]) > 1 or self.get_max_dispacement() > self._parameters.max_displacement
        elif self._valid:
            raise ValueError("Efficency matrix not computed. Run compute method on this structure")
        
    def get_max_stess(self) -> float:
        """
        Get maximum stress in the structure

        Returns:
            float: Maximum stress in the structure
        """
        return float(get_signed_max(self._trusses[3]))
    
    def get_max_load(self) -> float:
        """
        Get max load in the structure

        Returns:
            float: Max load in the structure
        """
        return float(get_signed_max(self._trusses[2]))
    
    def get_diameters(self) -> np.array:
        """
        Get the eqivalent trusses diameters if a circular cross-section is assumed

        Returns:
            np.array: Cross-sections diameters
        """
        return np.power(4*self._trusses[1]/3.14, 1/2)
    
    def get_mass(self, include_density:bool = True) -> Tuple[float, float, np.array]:
        """
        Computes the mass and volume distribution of the structure.

        This method calculates the total mass of the trusses and nodes in the structure. It can
        optionally include the material density in the calculations. It also returns the volume 
        (or mass, if density is included) of each truss element.

        Args:
            include_density (bool, optional): If True, the mass is computed using the material density. 
                                            If False, only the volume is returned (density is assumed to be 1).
                                            Defaults to True.

        Returns:
            Tuple[float, float, np.array]:
                - Total truss mass (or volume if `include_density=False`).
                - Combined truss and node mass (or volume if `include_density=False`).
                - Array containing the volume (or mass if `include_density=True`) of each truss element.
        """
        volume_matrix = self._trusses[1]*self._trusses[6]
        truss_volume = np.sum(volume_matrix)/2
        node_volume = len(self._nodes)*self._parameters.node_volume
        if include_density:
            density = self._parameters.material.density
        else:
            density = 1
        return float(truss_volume*density), float((truss_volume+node_volume)*density), volume_matrix*density
    
    def get_max_dispacement(self) -> float:
        """
        Get the maximum displacement in the structure

        Returns:
            float: Maximum displacement in the structure
        """
        return float(np.max(np.abs(self._nodes[:,2:4])))
        
    def compute(self) -> np.array:
        """
        Solve the structure and compute stress (wrapper function)

        Returns:
            np.array: Fitness matrix
        """
        self.aggregate_nodes()
        self.solve()
        self.compute_stress()
        return self.get_fitness()
