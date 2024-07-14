import numpy as np
from src.types import Node
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
    # Make A matrxi
    Kx, Ky = trussK(nodes, trusses, elastic_modulus)
    Vx, Vy = constrain(nodes)
    adju = np.triu(trusses[0])
    
    n = len(nodes)
    Ro = np.eye(n,n)
    Rx = np.multiply(Ro, Vx)
    Ry = np.multiply(Ro, Vy)
    
    Rx = np.concatenate([Rx, np.zeros((n,n))])
    Ry = np.concatenate([np.zeros((n,n)), Ry])
    
    R = np.hstack((Rx, Ry))
    
    Au = np.multiply(Kx - np.matmul(Kx, trusses[0])*np.identity(3), np.matmul(adju, Vx))
    Av = np.multiply(Ky - np.matmul(Ky, trusses[0])*np.identity(3), np.matmul(adju, Vy))
    A = np.concatenate([Au,Av])
    A = np.hstack((A, R))
    
    #d_idx = np.argwhere(np.all(A[..., :] == 0, axis=0))
    #A = np.delete(A, d_idx, axis=1)
    print(A) 

    Bx, By = loads(nodes)
    B = np.concatenate([Bx,By])
    
    #Normalize dimension
    e, i = A.shape
    #A = np.stack((A, np.zeros((i-e, i))))
    print(i-e)
    B.resize((i))
    #print(B.shape, A.shape)
    x = np.linalg.solve(A, B)
    return None