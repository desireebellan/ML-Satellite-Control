from casadi import *
from copy import copy
import numpy as np

class AttributeDict(dict):
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    
def cross(x, y):
    assert x.shape[0] == 3
    return SkewSym(x) @ y

def euler2quat(e, theta):
    e = e.flatten() / np.linalg.norm(e)
    q = np.block([e * sin(theta/2), cos(theta/2)])
    if q[3] < 0 : q = -q
    return q / np.linalg.norm(q)

def quat_conjugate(quaternion):
    """
    Compute the conjugate of a quaternion.
    
    Args:
        quaternion (np.ndarray): Quaternion [w, x, y, z]
    
    Returns:
        conjugate (np.ndarray): Conjugate quaternion [w, -x, -y, -z]
    """
    conjugate = copy(quaternion)
    conjugate[:3] *= -1.0
    return conjugate

def quat_error(q, qd):
    #qe = quat_product(q, quat_conjugate(qd))
    qe = quat_product(q, quat_conjugate(qd))
    #if qe[3] < 0 :  return -qe 
    return qe

def quat_product(q1, q2):
    """
    Multiply two quaternions.
    
    Args:
        quat1 (np.ndarray): Quaternion [x, y, z, w]
        quat2 (np.ndarray): Quaternion [x, y, z, w]
    
    Returns:
        product (np.ndarray): Quaternion product [x, y, z, w]
    """
    '''x1, y1, z1, w1 = quat1
    x2, y2, z2, w2 = quat2
    product = np.array([      
        w1*x2 + w2*x1 + y1*z2 - z1*y2,
        w1*y2 + w2*y1 + z1*x2 - x1*z2  ,
        w1*z2 + w2*z1 + x1*y2 - y1*x2 ,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])'''
    w1, w2 = q1[3], q2[3]
    v1 = q1[:3].reshape((3, 1))
    v2 = q2[:3].reshape((3, 1))
    
    product = vertcat(
        w1 * v2 + w2 * v1 - cross(v1, v2), 
        w1 * w2 - dot(v1, v2)
    )
    product = product / norm_2(product)
    return product

def scalar_quat_error(q, qd, case:int=1):
    qe = quat_error(q, qd)
    if case == 1 : return arctan2(sqrt(norm_2(qe[:3])), q[3])
    elif case == 2 : return norm_2(qe[:3])*q[3]
    else : raise NotImplementedError

def SkewSym(x): # SPART LIBRARY
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

    return blockcat([[0, -x[2], x[1]] , [x[2], 0, -x[0]] , [-x[1], x[0], 0] ])

def transpose(x):
    if len(x.shape) > 1: return x.T 
    else : return x.reshape((-1,1))
    
