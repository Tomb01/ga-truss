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

def populateA(nodes, trusses, elastic_modulus) -> np.array:
    return A

def solve(nodes, trusses, elastic_modulus) -> np.array:
    dl = distance(nodes, trusses)
    l = lenght(dl)
    K = np.divide(elastic_modulus*trusses[1], l, out=np.zeros_like(l), where=trusses[1]!=0)
    a = inclination(dl)
    
    # Make A matrxi
    Ku = K[0]
    Kv = K[1]
    Vx, Vy = constrain(nodes)
    
    n = len(nodes)
    Ro = np.eye(n,n)
    Rx = np.multiply(Ro, Vx)
    Ry = np.multiply(Ro, Vy)
    
    Ku = (Ku - np.matmul(Ku, trusses[0])*np.identity(3))
    Kv = (Kv - np.matmul(Kv, trusses[0])*np.identity(3))
    
    Dx = np.concatenate([Ku*np.cos(a),Kv*np.cos(a),Rx,np.zeros((n,n))]).transpose()
    Dy = np.concatenate([Ku*np.sin(a),Kv*np.sin(a),np.zeros((n,n)),Ry]).transpose()
    Rx = np.concatenate([Rx, np.zeros((3*n, n))]).transpose()
    Ry = np.concatenate([np.zeros((n, n)), Ry, np.zeros((2*n, n))]).transpose()
    
    print(Dy)
    
    A = np.vstack((Dx, Dy, Rx, Ry))
    mask_row = ~np.all(A==0, axis=0)
    mask_col = ~np.all(A==0, axis=1)
    A = A[mask_row]
    A = A[:,mask_col]
    #print(A, A.shape)
    
    Bx, By = loads(nodes)
    B = np.concatenate([Bx,By])
    
    B.resize((A.shape[1]))
    #print(np.linalg.det(A))
    
    #x = np.linalg.solve(A, B)
    return None