import numpy as np  
    
def Node(x: float, y: float, vx: bool, vz: bool, Px: float, Pz: float):
    return np.array([x,y,0,0,int(vx),int(vz),0,0,Px,Pz], np.float64)