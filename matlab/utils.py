import numpy as np

def transpose(x:np.ndarray):
    if len(x.shape) > 1: return x.T 
    else : return x.reshape((-1,1))
    
def SkewSym(x:np.ndarray): # SPART LIBRARY
    # Computes the skew-symmetric matrix of a vector, which is also the
    # left-hand-side matricial equivalent of the vector cross product
    #
    # [x_skew] = SkewSym(x)
    #
    # :parameters:
    #	* x -- [3x1] column matrix (the vector).
    #
    # :return:
    #	* x_skew -- [3x3] skew-symmetric matrix of x.

    return np.block([[0, -x[2], x[1]] , [x[2], 0, -x[0]] , [-x[1], x[0], 0] ])

def Omega(x:np.ndarray):
    x1, x2, x3 = x
    return np.block([[0  , x3 , -x2, x1],
                     [-x3, 0  , x1 , x2],
                     [x2 , -x1, 0  , x3],
                     [-x1, -x2, -x3, 0]])
    
class AttributeDict(dict):
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
