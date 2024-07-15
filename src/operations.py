import numpy as np
from typing import Tuple

def connections(m: np.array, adj: np.array) -> np.array:    
    if len(adj.shape) > 2:
        return np.multiply(m, adj[0])
    else:
        return np.multiply(m, adj)

def repeat(m: np.array) -> np.array:
    return np.swapaxes([m]*(m.shape[0]), 0, 2)

def distance(nodes: np.array, trusses: np.array):
    all_nodes = repeat(nodes)
    return connections(np.swapaxes(all_nodes, 2, 1) - all_nodes , trusses)

def lenght(distances: np.array) -> np.array:
    return (distances[0]**2+distances[1]**2)**(1/2)

def inclination(distances: np.array) -> np.array:
    return np.arctan2(distances[1], distances[0])

def constrain(nodes: np.array) -> Tuple[np.array, np.array]:
    return nodes[:,4], nodes[:,5] 

def loads(nodes: np.array) -> Tuple[np.array, np.array]:
    return nodes[:,8], nodes[:,9] 

def disp(k_disp, trusses) -> np.array:
    return k_disp - np.matmul(k_disp, trusses[0])*np.identity(trusses.shape[1])

def solve(nodes, trusses, elastic_modulus) -> np.array:
    # Calculation of trusses lenghts and inclinations
    dl = distance(nodes, trusses)
    l = lenght(dl)
    K = np.divide(elastic_modulus*trusses[1], l, out=np.zeros_like(l), where=trusses[1]!=0)
    a = inclination(dl)
    
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
    trusses[3] = np.divide(trusses[2], trusses[1], out=np.zeros_like(trusses[1]), where=trusses[1]!=0)

    return nodes, trusses