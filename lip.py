import numpy as np
import math
import random
from scipy.stats import dirichlet

N=3 #Consider a NxN grid
basis = [] #Current basis

#Basically, we're saying that if dx < eps, |f(x + dx) - f(x)| < kdx
eps = 2 #How "local" our Lipschitz continuity constraint is
k = .1 #How quickly it can change

#Check lipschitz condition
def check_bound(v):
    for x in range(v.shape[0] - 1): #grid dimension 1
        for y in range(v.shape[1] - 1): #grid dimension 2
            for dx in range(int(eps)): #x component of dx
                for dy in range(int(math.sqrt(eps**2-dx**2))): #y component of dx
                    if dx == 0 and dy == 0:
                        continue
                    if abs(v[x][y] - v[x+dx][y+dy]) >= k*math.sqrt(dx**2+dy**2): #Lipschitz constraint
                        return False
    return True

#Check if vector is LI wrt our current basis
def independent(v):
    ini = np.linalg.norm(v) #initial mag of vector
    if len(basis) == 0:
        return True
    for i in range(len(basis)): #subtract out vector projections for each basis vector
        v -= np.dot(basis[i][0],v[0]) * basis[i][0] / np.linalg.norm(basis[i][0])
        
    if np.linalg.norm(v)/ini > .8: #did the vector survive
        return True
    return False

for i in range(10000): #generate 10000 random vectors
    alpha = [1 for i in range(N**2)] #let's use an NxN grid
    P = dirichlet.rvs(alpha) #uniformly generate a value from the N^2 simplex

    if not independent(np.copy(P)): #is our vector independent from the rest
        continue
    
    P.shape = (N, N)
    
    if check_bound(np.copy(P)): #does it meet lipschitz constraint
        P.shape = (1, N**2)
        basis.append(P) #add it to the basis
        print(P)

print(len(basis))
