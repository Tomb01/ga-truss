import numpy as np
from src.types import Node
from typing import Tuple
from scipy.linalg import lu

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

def trussK(nodes: np.array, trusses: np.array, elastic_modulus: float) -> np.array:
    dl = distance(nodes, trusses)
    l = lenght(dl)
    K = np.divide(elastic_modulus*trusses[1], l, out=np.zeros_like(l), where=trusses[1]!=0)
    a = inclination(dl)
    return K*np.cos(a), K*np.sin(a)

def constrain(nodes: np.array) -> Tuple[np.array, np.array]:
    return nodes[:,4], nodes[:,5] 

def loads(nodes: np.array) -> Tuple[np.array, np.array]:
    return nodes[:,8], nodes[:,9] 

def disp(k_disp, trusses) -> np.array:
    return k_disp - np.matmul(k_disp, trusses[0])*np.identity(trusses.shape[1])

def solve(nodes, trusses, elastic_modulus) -> np.array:
    dl = distance(nodes, trusses)
    l = lenght(dl)
    K = np.divide(elastic_modulus*trusses[1], l, out=np.zeros_like(l), where=trusses[1]!=0)
    a = inclination(dl)
    
    n = len(nodes)
    Ku = K*np.cos(a)
    Kv = K*np.sin(a)
    
    Ux = disp(Ku*np.cos(a), trusses)
    Uy = disp(Ku*np.sin(a), trusses)
    Vx = disp(Kv*np.cos(a), trusses)
    Vy = disp(Kv*np.sin(a), trusses)
    
    Cx, Cy = constrain(nodes)

    Ro = np.eye(n,n)
    Rx = np.multiply(Ro, Cx)
    Ry = np.multiply(Ro, Cy)
    
    A = np.zeros((4*n, 4*n))
    
    A[0:n, 0:n] = Ux    
    A[0:n, n:2*n] = Uy
    A[0:n, 2*n:3*n] = Rx
    
    A[n:2*n, 0:n] = Vx    
    A[n:2*n, n:2*n] = Vy
    A[n:2*n, 3*n:4*n] = Ry
    
    A[2*n:3*n, 0:n] = Rx
    A[3*n:4*n, n:2*n] = Ry
    
    mask_row = ~np.all(A==0, axis=0)
    mask_col = ~np.all(A==0, axis=1)
    A = A[mask_row]
    A = A[:,mask_col]
    
    Bx, By = loads(nodes)
    B = -np.concatenate([Bx,By])
    
    B.resize((A.shape[1]))

    print(A)
    x = np.linalg.solve(A,B)
    return x