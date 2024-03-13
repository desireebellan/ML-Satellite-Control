# python3.9
# Control Strategies used for the Planar Robotic Arm

import numpy as np

from typing import Callable
from utils import angle_diff

class PD(object):
    def __init__(self, qd:Callable[[float],float], qddot:Callable[[float], float], qdddot:Callable[[float], float] = None, 
                 num_links:int = 2, Kd = 20, Kp = 50):
        self.qd = qd # function of time 
        self.qddot = qddot # function of time
        self.qdddot = qdddot
        self.Kd = np.diagflat(np.full((1,num_links), Kd))
        self.Kp = np.diagflat(np.full((1,num_links), Kp))
        self.num_links = num_links
    def next(self, q, qdot, t):
        # u = Kp(qd-q) + Kd(qddot-qdot)
        #return self.Kp @ (self.qd(t)-q) + self.Kd @ (self.qddot(t) - qdot)
        #print("desired", self.qdddot(t), self.qd(t), self.qddot(t))
        #print("real", q, qdot)
        return self.qdddot(t) + \
            self.Kp @ angle_diff(self.qd(t),q[-self.num_links:, :])\
                + self.Kd @ (self.qddot(t) - qdot[-self.num_links:, :])
    
class ComputedTorque(object):
    def __init__(self, qd:Callable[[float],float], qddot:Callable[[float], float], qdddot:Callable[[float], float],
                 M:Callable[[float], np.ndarray], n:Callable[[float, float], np.ndarray], num_links:int = 2, Kd = 1, Kp = 10):
        '''
            M = inertia matrix of the manipulator without the payload (known) -> function depending on q
            n = coriolis term of the manipulator without the payload (known) -> function depending on q,qdot
            qd = desired trajectory
            qddot = desired angular velocity
            qdddot = desired acceleration
            Kd = 
        '''
        self.qd = qd
        self.qddot = qddot
        self.qdddot = qdddot
        self.M = M
        self.n = n
        self.Kd = np.diagflat(np.full((1,num_links), Kd))
        self.Kp = np.diagflat(np.full((1,num_links), Kp))
        
        self.num_links = num_links
        
    def next(self, q, qdot, t):
        # u = M(q)[qdddot + Kp(qd-q) + Kd(qddot-qdot)] + n(q,qdot)
        return (self.M(q)@( self.qdddot(t) 
                           + self.Kp @ (self.qd(t)    - q[-self.num_links:, :])
                           + self.Kd @ (self.qddot(t) - qdot[-self.num_links:, :]))
                + self.n(q, qdot)).reshape(self.num_links, 1)
        
class Control(object):
    def __init__(self, qd:Callable[[float],float], qddot:Callable[[float], float], qdddot:Callable[[float], float],
                 ID:Callable[[np.ndarray], np.ndarray], num_links:int = 2, Kd = 1, Kp = 10):
        self.control_law = PD(qd=qd, qddot=qddot, qdddot=qdddot, num_links=num_links, Kd=Kd, Kp=Kp)
        self.ID = ID
    def next(self, q, qdot, t):
        qddot = self.control_law.next(q, qdot, t)
        return self.ID(q, qdot, qddot)