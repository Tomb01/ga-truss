import numpy as np
from typing import Tuple
import random

FLOAT_MAX = 1e100 #sys.float_info.max

def connections(m: np.array, adj: np.array) -> np.array:    
    if len(adj.shape) > 2:
        return np.multiply(m, adj[0])
    else:
        return np.multiply(m, adj)

def repeat(m: np.array) -> np.array:
    return np.swapaxes([m]*(m.shape[0]), 0, 2)
    
def distance(nodes: np.array, trusses: np.array, conn = True):
    all_nodes = repeat(nodes)
    x = np.swapaxes(all_nodes, 2, 1) - all_nodes 
    if conn:
        return connections(x, trusses)
    else:
        return x

def length(distances: np.array) -> np.array:
    return (distances[0]**2+distances[1]**2)**(1/2)

def inclination(distances: np.array) -> np.array:
    return np.arctan2(distances[1], distances[0])

def constrain(nodes: np.array) -> Tuple[np.array, np.array]:
    return nodes[:,4], nodes[:,5] 

def loads(nodes: np.array) -> Tuple[np.array, np.array]:
    return nodes[:,8], nodes[:,9] 

def disp(k_disp, trusses) -> np.array:
    return k_disp - np.matmul(k_disp, trusses[0])*np.identity(trusses.shape[1])

def make_sym(m: np.array) -> np.array:
    m = np.triu(m)
    return m + m.T - np.diag(np.diag(m))

def solve(nodes, trusses, elastic_modulus) -> np.array:
    # Calculation of trusses lenghts and inclinations
    dl = distance(nodes, trusses)
    l = length(dl)
    K = np.divide(elastic_modulus*trusses[1], l, out=np.zeros_like(l), where=trusses[0]!=0)
    a = inclination(dl)
    trusses[6] = l
    
    # Trusses constants in x (cos) and y (sin) direction
    n = len(nodes)
    Ku = K*np.cos(a)
    Kv = K*np.sin(a)
    
    # Displacement matrix in x and y direction 
    # Nij_x = (Ku[ij]*cos(a[ij]) + Kv[ij]*sin(a[ij]))*cos(a[ij])
    Ux = disp(Ku*np.cos(a), trusses)
    Uy = disp(Ku*np.sin(a), trusses)
    Vx = disp(Kv*np.cos(a), trusses)
    Vy = disp(Kv*np.sin(a), trusses)
    
    # get nodes constrain (1 = locked direction)
    Cx, Cy = constrain(nodes)

    # Reaction matrix, if R[ij] = 1 direction of that node is locked (for x and y)
    Ro = np.eye(n,n)
    Rx = np.multiply(Ro, Cx)
    Ry = np.multiply(Ro, Cy)
    
    # row = n displacement in x and y (2n) + n reaction in x and y (2n)
    # matrix must be square to solve system
    A = np.zeros((4*n, 4*n))
    
    A[0:n, 0:n] = Ux    
    A[0:n, n:2*n] = Uy
    A[0:n, 2*n:3*n] = Rx
    
    A[n:2*n, 0:n] = Vx    
    A[n:2*n, n:2*n] = Vy
    A[n:2*n, 3*n:4*n] = Ry
    
    A[2*n:3*n, 0:n] = Rx
    A[3*n:4*n, n:2*n] = Ry
    
    # Delete all zeros row and colum to make matrix non singular
    mask_row = ~np.all(A==0, axis=0)
    mask_col = ~np.all(A==0, axis=1)
    A = A[mask_row]
    A = A[:,mask_col]
    
    # Loads therm
    # N1 + N2 + ... Nn + P = 0
    Bx, By = loads(nodes)
    B = -np.concatenate([Bx,By]) # move to right, add minus(-)
    
    # Resize with zeros to match A dimension
    B.resize((A.shape[1]))
    
    # Solve the linear system
    sol = np.linalg.solve(A,B)
    
    # Map results to correct dimension based on previus deleted row 
    t_idx = np.where(mask_row)
    x = np.zeros(4*n)
    np.put(x, t_idx, sol)
    
    # Put solution in nodes matrix
    nodes[:,2] = x[0:n] # displacement u
    nodes[:,3] = x[n:2*n] # displacement v
    nodes[:,6] = x[2*n:3*n] # reaction x
    nodes[:,7] = x[3*n:4*n] # reaction y
    
    # calculate forces, N12 = Ku*(u2-u1) + Kv*(v2-v1)
    ds = distance(nodes, trusses)
    # Assign forces to trusses matrix
    trusses[2] = Ku*ds[2] + Kv*ds[3]
    # Calculate tension s = N/A, assign to trusses matrix
    #print(trusses[2])
    trusses[3] = np.divide(trusses[2], trusses[1], out=np.zeros_like(trusses[1]), where=trusses[1]!=0)
    return nodes, trusses

def pad_with_zeros(A, r, c):
    out = np.zeros((r, c))
    r_, c_ = np.shape(A)
    out[0:r_, 0:c_] = A
    return out

def upper_tri_masking(A) -> Tuple[np.array, np.array]:
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask], mask

def get_signed_max(m):
    idx = np.where(m == np.max(np.abs(m)))[0]
    return m[idx]

def vector_distance(y2, y1, x2, x1) -> float:
    return ((y2-y1)**2+(x2-x1)**2)**(1/2)

def binary_turnament(fitness: np.array) -> int:
    idx = range(0, len(fitness))
    p1, p2 = random.sample(idx, 2)
    if fitness[p1] > fitness[p2]:
        return p1
    else:
        return p2

# https://math.stackexchange.com/a/911040
def deg(adj: np.array) -> np.array:
    edge = np.sum(adj, axis=0)
    out = np.zeros_like(adj)
    diag_idx = np.diag_indices(adj.shape[0])
    out[diag_idx] = edge
    return out

def laplacian(adj: np.array) -> np.array:
    D = deg(adj)
    return D - adj

def is_cyclic(adj: np.array) -> bool:
    L = laplacian(adj)
    return np.trace(L) >= (np.linalg.matrix_rank(L)+1)
