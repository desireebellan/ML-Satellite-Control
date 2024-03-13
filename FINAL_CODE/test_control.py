import numpy as np

from SPART.robot_model import *
from SPART.spacecraftStep import *
from SPART.dynamics import *
from SPART.kinematics import *

from utils import *

def ID_(q, qdot, qddot):
        # no external forces and torques
        wF0 = np.zeros((6, 1))
        wFm = np.zeros((6, robot.n_links_joints))
        
        q0, qm = q[:7,:], q[7:,:]
        R0 = transpose(quat_DCM(transpose(q0[:4,:]).squeeze()))
        r0 = q0[4:, :]  
        
        u0, um = qdot[:6,:], qdot[6:,:]

        # Floating case 
        if  rw and not  thrusters:
            u0dot, umdot = np.block([[qddot[:3, :]],[zeros((3, 1))]]), qddot[3:,:]
        if  rw and  thrusters:
            u0dot, umdot = qddot[:6,:], qddot[6:,:]
        else :
            assert not (not  rw and  thrusters)
            u0dot, umdot = zeros((6,1)), qddot
                
        _,RL,_,rL,e,g = Kinematics(R0, r0, qm, robot) 
        Bij, Bi0, P0, pm = DiffKinematics(R0, r0, rL, e, g, robot)
        t0, tL = Velocities(Bij, Bi0, P0, pm, u0, um, robot)
        t0dot, tLdot = Accelerations(t0, tL, P0, pm, Bi0, Bij, u0, um, u0dot, umdot, robot)
        I0, Im = I_I(R0, RL, robot)
        
        tau0, taum = ID(wF0,wFm,t0,tL,t0dot,tLdot,P0,pm,I0,Im,Bij,Bi0,robot)
        # floating case
        return taum
    
if __name__=="__main__":
    from classic_control import Control
    from copy import deepcopy
    from tqdm import tqdm
    
    filename = "./matlab/SC_3DoF.urdf"
    robot, _ = urdf2robot(filename=filename)
    
    rw = True 
    thrusters = False 
    harmonics = 5
    T = 20
    n = robot.n_q
    steps = 2000
    dt = 0.01
    max_action = 400
    
    initial_state = {
                    'q0' : np.block([np.zeros(3), 1 ,np.zeros(3)]).reshape((7, 1)).astype(np.float32), # base quaternion
                    'qm': np.zeros((n, 1)).astype(np.float32), # Joint variables [rad]
                    'u0': np.zeros((6,1)).astype(np.float32), # Base-spacecraft velocity
                    'um': np.zeros((n, 1)).astype(np.float32), # Joint velocities
                    'u0dot': np.zeros((6,1)).astype(np.float32), # Base-spacecraft acceleration
                    'umdot': np.zeros((n, 1)).astype(np.float32), # Joint acceleration
                    'tau0': np.zeros((6,1)).astype(np.float32), # base RW
                    'taum' : np.zeros((n, 1)).astype(np.float32) # manipulator joint torques
    }
    q0 = initial_state["qm"]
    data = deepcopy(initial_state)
    
    qd, qddot, qdddot = sinusoids_trajectory(n_joints= n, 
                                            T        = T, 
                                            q0       =  q0, 
                                            harmonics=  harmonics) 
    control = Control( qd =  qd, 
                        qddot     =  qddot, 
                        qdddot    =  qdddot, 
                        ID        =  ID_, 
                        num_links =  n,
                        rw        = True,
                        thrusters = False)
    
    # for number of steps
    for t in tqdm(range(steps)):
        tau = control.next(np.block([[data["q0"]], [data["qm"]]]), 
                           np.block([[data["u0"]], [data["um"]]]), t*dt)
        
        # next state            
        # _, _, done, is_terminated, _ = self.step(tau.flatten())
        if rw:
            tau0 = np.block([[tau[:3,:]], [np.zeros((3,1))]])
            taum = tau[3:,:]
        if rw and thrusters:
            tau0 = tau[6:,:]
            taum = tau[6:,:]
        else:
            tau0 = np.zeros((6, 1))
            taum = tau
        taum = np.clip(taum, -max_action, max_action)
        
        wf0  = np.zeros((6, 1))
        wfm = np.zeros((6, robot.n_links_joints))
        
        data = spacecraftStep(tau0, taum, wf0, wfm, data, robot, dt = dt)