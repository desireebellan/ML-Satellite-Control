# python3.9
# Control Strategies used for the Planar Robotic Arm

import numpy as np

from typing import Callable
from utils import angle_diff
from SPART.attitude_transformations import quat_Angles321

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
        return self.Kp @ angle_diff(self.qd(t),q[-self.num_links:, :])\
                + self.Kd @ (self.qddot(t) - qdot[-self.num_links:, :])
                
class PDFFW(object):
    def __init__(self, qd:Callable[[float],float], qddot:Callable[[float], float], qdddot:Callable[[float], float] = None, 
                 num_links:int = 2, Kd = 20, Kp = 50, rw:bool=False, thrusters:bool=False):
        self.qd         = qd # function of time 
        self.qddot      = qddot # function of time
        self.qdddot     = qdddot
        self.Kd         = np.diagflat(np.full((1,num_links), Kd))
        self.Kp         = np.diagflat(np.full((1,num_links), Kp))
        self.num_links  = num_links
        self.rw         = rw
        self.thrusters  = thrusters
        
        self.Kp_rw      = np.diagflat(np.full((1, 3), 30))
        self.Kd_rw      = np.diagflat(np.full((1, 3), 10))
        self.Kp_t       = np.diagflat(np.full((1, 3), 30))
        self.Kd_t       = np.diagflat(np.full((1, 3), 10))
        
    def next(self, q, qdot, t):
        # u = Kp(qd-q) + Kd(qddot-qdot)
        #return self.Kp @ (self.qd(t)-q) + self.Kd @ (self.qddot(t) - qdot)
        #print("desired", self.qdddot(t), self.qd(t), self.qddot(t))
        #print("real", q, qdot)
        u = self.qdddot(t) + \
            self.Kp @ angle_diff(self.qd(t),q[-self.num_links:, :])\
                + self.Kd @ (self.qddot(t) - qdot[-self.num_links:, :])
        if self.rw:
            # reaction wheels are used 
            u0 = - self.Kp_rw @ quat_Angles321(q[:4,:]) - self.Kd_rw @ qdot[:3,:]
            if self.thrusters:
                u0_ = - self.Kp_t @ q[4:7,:] - self.Kd_t @ qdot[3:6,:]
                u0 = np.block([u0.flatten(), u0_.flatten()]).reshape(6, 1)
            u = np.block([[u0], [u]])
        return u
        
        
class Control(object):
    def __init__(self, qd:Callable[[float],float], qddot:Callable[[float], float], qdddot:Callable[[float], float],
                 ID:Callable[[np.ndarray], np.ndarray], num_links:int = 2, Kd = 1, Kp = 10, rw:bool=False, thrusters:bool=False):
        self.control_law = PDFFW(qd=qd, qddot=qddot, qdddot=qdddot, num_links=num_links, Kd=Kd, Kp=Kp, rw=rw, thrusters=thrusters)
        self.ID = ID
    def next(self, q, qdot, t):
        qddot = self.control_law.next(q, qdot, t)
        return self.ID(q, qdot, qddot)


    
    
        


