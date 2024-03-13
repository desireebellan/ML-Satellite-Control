# pybullet test

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
        elif  rw and  thrusters:
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
        if rw and not thrusters:
            return np.block([[tau0[:3,:]], [taum]])
        if rw and thrusters:
            return np.block([[tau0], [taum]])
        else:
            return taum

def get_params():
        # get random mass from a uniform distribution
        m = np.random.uniform(low = 0, high = 10) 
        # get random center of mass (com) position wrt the end effector com
        # gaussian distribution over a circle (2D) with center aligned with its x axis            
        rcom_x = np.abs(np.random.normal(loc = 0.3, scale = 0.1, size = 1))   
        rcom_y = np.random.normal(loc = 0.0, scale = 0.1, size = 1)
        rcom_z = np.random.normal(loc = 0.0, scale = 0.1, size = 1)
            
        rcom = np.array([rcom_x, rcom_y, rcom_z]).squeeze()
        # get random moment of inertia : 
        # (i) random principal moment of inertia 
        j = np.random.normal(loc = 0.0, scale = 1.0, size = 3)
        j = np.array(list(map(lambda x : x**2, j)))
        E = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        J = np.diag(np.matmul(E, j))
        # (ii) random rotation matrix
        Q, _ = np.linalg.qr(np.random.randn(3, 3))
        #J = np.transpose(Q) @ J @ Q
        # Final random inertia matrix rotated by Q
        # I = np.transpose(Q) @ J @ Q
         
        #return {'mp': m, 'rp': rcom, 'Ip': J, 'R': Q}  
        return {'m': m, 'r': rcom, "J":J, 'Q': Q}  

def initialize_robot():
    filename = "./matlab/SC_3DoF.urdf" 
    robot, _ = urdf2robot(filename)
    
    robot = setParams(params, robot)
    n = robot.n_q
    
    initial_state = {
                    'q0' : np.block([np.array([0,0,0,1]),zeros(3)]).reshape((7, 1)).astype(np.float32), # base quaternion
                    'qm': zeros((n, 1)).astype(np.float32), # Joint variables [rad]
                    'u0': zeros((6,1)).astype(np.float32), # Base-spacecraft velocity
                    'um': zeros((n, 1)).astype(np.float32), # Joint velocities
                    'u0dot': zeros((6,1)).astype(np.float32), # Base-spacecraft acceleration
                    'umdot': zeros((n, 1)).astype(np.float32), # Joint acceleration
                    'tau0': zeros((6,1)).astype(np.float32),                    'taum': zeros((n,1)).astype(np.float32) # manipulator joint torques
            }
    data = deepcopy(initial_state)
    wf0 = zeros((6,1))
    wfm = zeros((6,robot.n_links_joints))
    return robot, data, wf0, wfm


if __name__ == "__main__":
    import numpy as np
    from numpy import zeros
    from SPART.robot_model import *
    from copy import deepcopy
    import pybullet as p
    import pybullet_data
    from SPART.spacecraftStep import spacecraftStep
    from tqdm import tqdm
    import time
    import gc
    from classic_control import Control
    
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    timeStep = 1/240.0
    p.setTimeStep(timeStep) 
    #planeId = p.loadURDF("plane.urdf")
    #start_pos = [1,-1,1]
    start_pos = [0.,0.,0.]
    #start_orientation = p.getQuaternionFromEuler([0,0,pi/2 + pi/6])
    start_orientation = p.getQuaternionFromEuler([0,0,0])
    robotId = p.loadURDF("./matlab/SC_3DoF.urdf",start_pos, start_orientation, useFixedBase=False, physicsClientId=physicsClient)
    filename = "./matlab/SC_3DoF.urdf" 
    #robot, _ = urdf2robot(filename)
    m = p.getNumJoints(robotId)
    n = []
    for i in range(m):
        if p.getJointInfo(robotId, i)[2] == 0:
            n.append(i)
    
    for i in range(m):
        dynamics = p.getDynamicsInfo(robotId, i)
        if dynamics[0] == 0.0:
            params=get_params()
            p.changeDynamics(robotId, i, mass=params["m"], localInertiaDiagonal=np.diag(params["J"]))
        dynamics = p.getDynamicsInfo(robotId, i)
    #n = robot.n_q
    
    p.setRealTimeSimulation(0)
    gc.collect()
    # This step is required to enable torque control. Refer to the documentation for more details.
    for i in range(m):
        p.setJointMotorControl2(robotId, i, p.VELOCITY_CONTROL, force=0.0)

    filename = "./matlab/SC_3DoF.urdf"
    robot, _ = urdf2robot(filename=filename)
    
    rw = True 
    thrusters = False 
    harmonics = 5
    T = 20
    dt = 0.01
    steps = int(T/dt)
    max_action = 400
    
    '''qb = p.getBasePositionAndOrientation(robotId)
    q0 = []
    for i in n:
        q0.append(p.getJointState(robotId, i)[0])
    q0 = np.array(q0).reshape((len(n), 1))'''
    
    '''qd, qddot, qdddot = sinusoids_trajectory(n_joints= len(n), 
                                                T        = T, 
                                                q0       =  q0, 
                                                harmonics=  harmonics) 
    control = Control( qd =  qd, 
                        qddot     =  qddot, 
                        qdddot    =  qdddot, 
                        ID        =  ID_, 
                        num_links =  len(n),
                        rw        = True,
                        thrusters = False)'''
    
    '''from PIL import Image
    
    p.resetJointState(robotId, n[0], pi/4)
    p.resetJointState(robotId, n[1], -pi/4)
    p.resetJointState(robotId, n[2], pi/2)
    rgb = p.getCameraImage(1024,1024)[2] 
    #np_img_arr = np.reshape(rgb, (1024, 1024, 4))
    rbg = np.array(rgb, dtype=np.uint8)
    print(rgb)
    img = Image.fromarray(rbg, "RGBA")
    img.show()'''
        
    '''for i in tqdm(range(2000)):
        
        
        
        qm = np.array(list(map(lambda x : x[0], p.getJointStates(robotId, n)))).reshape(len(n), 1)
        um = np.array(list(map(lambda x : x[1], p.getJointStates(robotId, n)))).reshape(len(n), 1)
        qb = p.getBasePositionAndOrientation(robotId)
        qb = np.array([list(qb[1]) + list(qb[0])]).reshape(7,1)
        ub = p.getBaseVelocity(robotId)
        ub = np.array([list(ub[1]) + list(ub[0])]).reshape(6,1)
        
        tau = control.next(np.block([[qb], [qm]]), 
                           np.block([[ub], [um]]), i*dt)
        if rw and not thrusters:
            tau0 = np.block([[tau[:3,:]], [np.zeros((3,1))]])
            taum = tau[3:,:]
            p.applyExternalTorque(robotId, -1, torqueObj = list(tau[:3]), flags=p.LINK_FRAME)
            
        elif rw and thrusters:
            tau0 = tau[6:,:]
            taum = tau[6:,:]
            p.applyExternalTorque(robotId, -1, torqueObj = list(tau[:3]), flags=p.LINK_FRAME)
            p.applyExternalForce(robotId, -1, forceObj = list(tau[3:6]), flags=p.LINK_FRAME)
        else:
            tau0 = np.zeros((6, 1))
            taum = tau
        taum = np.clip(taum, -max_action, max_action)   
        p.setJointMotorControlArray(robotId, n, controlMode=p.TORQUE_CONTROL, forces=taum.flatten().tolist())
        p.stepSimulation()
        
        time.sleep(timeStep)
        gc.collect()'''
    p.disconnect()

    