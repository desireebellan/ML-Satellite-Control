import numpy as np

from math import sin, cos, pi, sqrt, acos

def scale(x, min = 0, max = 1):
    return x * (max - min) + min

def scalar_quat_error(q, qd, alpha = 1):
    qe = quat_error(q, qd)
    #return np.arctan2(np.sqrt(np.linalg.norm(qe[:3])), q[3]).squeeze()
    #return np.absolute(q[:3]).sum() * alpha
    return np.linalg.norm(qe[:3])*alpha

def quat_error(q, qd):
    #qe = quat_product(q, quat_conjugate(qd))
    qe = quat_product(q, quat_conjugate(qd))
    if qe[3] < 0 :  return -qe 
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
    v1 = q1[:3].reshape(3, 1)
    v2 = q2[:3].reshape(3, 1)
    product = np.vstack([
        w1 * v2 + w2 * v1 - np.cross(v1, v2, axis = 0), 
        w1 * w2 - np.dot(v1.T, v2)
    ])
    product = product / np.linalg.norm(product)
    return product

'''def rand_quat(min = 0.0, max = 1.0):
   x = np.random.uniform(low = min, high = max, size = 3)
   theta = 2*pi*x[1:] 
   q = np.array([sin(theta[0])*sqrt(1-x[0]), cos(theta[0])*sqrt(1-x[0]), sin(theta[1])*sqrt(x[0]), cos(theta[1])*sqrt(x[0])])
   q = q/np.norm(q)
   if q[3] < 0 : q = -q
   return q'''

def random_quaternion(max = 1.0, sigma = 1.0, goal = np.array([0, 0, 0, 1])):
    _, e_mean = quat2euler(goal)
    theta = np.random.uniform(high = max, size = 1) * 2 * pi 
    #theta = np.random.normal(loc = theta_mean, scale = sigma) % 2*pi
    e = np.random.normal(loc = e_mean, scale = sigma)
    q = np.block([e * sin(theta/2), cos(theta/2)])
    if q[3] < 0 : q = -q
    return q / np.linalg.norm(q)

def quat2euler(q):
    alpha = 2*acos(q[3])
    if alpha % pi == 0: euler_axis = np.array([0, 0, 1])
    else : euler_axis = 1/sin(alpha/2) * q[0:3]
    return alpha, euler_axis/np.linalg.norm(euler_axis)

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
    conjugate = quaternion.copy()
    conjugate[:3] *= -1.0
    return conjugate

def rad2grad(x):
    return x * 180 / pi

def angle_diff(x1, x2):
    e = x1 - x2 
    e_ = 2 * pi - e
    idx = abs(e) > abs(e_)
    e[idx] = e_[idx]
    return e

def angle_error(x1:np.ndarray, x2:np.ndarray) -> np.ndarray:
    e1 = np.abs(x1 - x2)
    e2 = 2 * pi - e1
    sign = ~ np.logical_xor(x1 < x2, e1 < e2)
    sign = sign * 2 - 1
    return np.minimum(e1, e2) * sign

def sinusoids_trajectory(n_joints:int, T:float, q0:np.ndarray, harmonics:int = 5, coefMax:int=1):
    wf = 2*pi/T
    n = np.random.randint(1, harmonics)
    a = np.random.rand(n_joints, n)*(2*coefMax) - coefMax
    b = np.random.rand(n_joints, n)*(2*coefMax) - coefMax

    # Truncated Fourier series
    q_ = lambda t : sum([a[:,k-1]/(wf*k)*sin(wf*k*t) - b[:,k-1]/(wf*k)*cos(wf*k*t) for k in range(1, n+1)]).reshape((n_joints, 1))
    qdot_ = lambda t : sum([a[:,k-1]*cos(wf*k*t) + b[:,k-1]*sin(wf*k*t) for k in range(1, n+1)]).reshape((n_joints, 1))
    qdotdot_ = lambda t : sum([-a[:,k-1]*wf*k*sin(wf*k*t) + b[:,k-1]*wf*k*cos(wf*k*t) for k in range(1, n+1)]).reshape((n_joints, 1))
    
    
    
    # Fifth order polynomial 
    # The free-coefficients cj should minimize the conition number of the regressor matrix
    # In this case are dtermined by using the desired initial and final conditions
    # q(0) = 0, q(N) = 0, qdot(0) = 0, qdot(N) = 0, qdotdot(0) = 0, qdotdot(N) = 0
    # The minimization process is left to the algorithm
    c = np.zeros((n_joints, 6))
    M = np.eye(6)
    M[2,2] = 2
    M[3:,:] = np.array([[0 if k > i else np.prod([i-j for j in range(0, k)])*T**(i-k) for i in range(6)] for k in range(3)])
    Minv = np.linalg.inv(M)
    qv = np.array([-q_(0) + q0, -qdot_(0), -qdotdot_(0), -q_(T) + q0, -qdot_(T), -qdotdot_(T)]).reshape((6, n_joints)).swapaxes(1,0)
    for i in range(n_joints):
        c[i, :] = Minv @ qv[i]  
    q = lambda t: sum([c[:, k]*t**k for k in range(6)]).reshape((n_joints, 1)) + q_(t)
    qdot = lambda t : sum([c[:,k]*k*t**(k-1) for k in range(1,6)]).reshape((n_joints, 1)) + qdot_(t)
    qdotdot = lambda t : sum([c[:,k]*k*(k-1)*t**(k-2) for k in range(2,6)]).reshape((n_joints, 1)) + qdotdot_(t)
    return q, qdot, qdotdot

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
