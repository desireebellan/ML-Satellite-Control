from utils import *
from SPART.attitude_transformations import quat_DCM
from SPART.kinematics import Kinematics, DiffKinematics, Velocities
from SPART.dynamics import I_I, FD

def spacecraftStep(tau0, taum, wf0, wfm, data, robot, dt:float = 0.01, n_steps:int = 1):
    data["taum"] = taum 
    data["tau0"] = tau0
    #np.block([[tau0],[taum]])
    data["wF"] = np.block([wf0, wfm])
    
    for i in range(n_steps):
        R0 = transpose(quat_DCM(transpose(data["q0"]).squeeze()))
        r0 = data["q0"][4:, :]
        RJ,RL,rJ,rL,e,g = Kinematics(R0, r0, data["qm"], robot) 
        Bij, Bi0, P0, pm = DiffKinematics(R0, r0, rL, e, g, robot)
        t0, tL = Velocities(Bij, Bi0, P0, pm, data["u0"], data["um"], robot)
        #t0dot, tLdot = Accelerations(t0, tL, P0, pm, Bi0, Bij, u0, um, u0dot, umdot, robot)
        I0, Im = I_I(R0, RL, robot)
        u0dot, umdot = FD(tau0, taum, wf0, wfm, t0, tL, P0, pm, I0, Im, Bij, Bi0, data["u0"], data["um"], robot)
        data = integrate(u0dot, umdot, data, dt)
    return data

def integrate(u0dot, umdot, data, dt):
    data["u0dot"] = u0dot
    data["umdot"] = umdot 
    
    data["um"] += umdot*dt 
    data["u0"] += u0dot*dt 
    data["u0"] = data["u0"]
    
    data["qm"] += data["um"]*dt 
    data["qm"] = normalize_angle(data["qm"], 0)
    data["q0"][4:] += data["u0"][3:]*dt 
    
    #q0dot = 1/2*Omega(data["u0"][:3])@data["q0"][:4,:]
    #data["q0"][:4] += q0dot * dt
    data["q0"][:4] = quaternionIntegrate(data["q0"][:4,:], data["u0"][:3], dt)
    return data

if __name__ == "__main__":
    
    import numpy as np
    from numpy import zeros
    from robot_model import urdf2robot
    from tqdm import tqdm
    from copy import deepcopy
    
    
    filename = "./matlab/SC_3DoF.urdf" 
    robot, _ = urdf2robot(filename)
    n = robot.n_q
    
    initial_state = {
                    'q0' : np.block([np.array([0,0,0,1]),zeros(3)]).reshape((7, 1)).astype(np.float32), # base quaternion
                    'qm': zeros((n, 1)).astype(np.float32), # Joint variables [rad]
                    'u0': zeros((6,1)).astype(np.float32), # Base-spacecraft velocity
                    'um': zeros((n, 1)).astype(np.float32), # Joint velocities
                    'u0dot': zeros((6,1)).astype(np.float32), # Base-spacecraft acceleration
                    'umdot': zeros((n, 1)).astype(np.float32), # Joint acceleration
                    'tau': zeros((6 + n,1)).astype(np.float32) # manipulator joint torques
            }
    data = deepcopy(initial_state)
    tau0 = zeros((6,1))
    wf0 = zeros((6,1))
    wfm = zeros((6,robot.n_links_joints))
    for i in tqdm(range(2000)):
        tau = np.random.rand(n, 1) * 2 - 1
        data = spacecraftStep(tau0, tau, wf0, wfm, data, robot)
    print(data)
    
    import matlab.engine
        
    engine = matlab.engine.start_matlab()
    engine.addpath('./matlab')
    engine.workspace["filename"] = filename
    engine.eval("[robot, ~] = urdf2robot(filename);", nargout = 0)
    param =  {"m": 0.0 , "rcom": zeros(3).astype(np.float32), "I": zeros((3,3)).astype(np.float32)}
    engine.workspace["param"] = param
    engine.eval("robotp = setParam(robot, param.m, param.rcom', param.I);", nargout = 0)
    data = deepcopy(initial_state)
    
    for i in tqdm(range(2000)):
        taum = np.random.rand(n, 1) * 2 - 1
        data_ = engine.spacecraftStep(tau0, taum, data, 0.01, 1)
        
        assert data is not None

        for key in data:     
            data[key] = np.asarray(data_[key]).reshape(data[key].shape)