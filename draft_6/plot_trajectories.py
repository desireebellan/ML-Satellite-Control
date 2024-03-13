from utils import sinusoids_trajectory

if __name__=="__main__":
    import matplotlib.pyplot as plt 
    import numpy as np
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--T", default = 20, type = int)
    parser.add_argument("--dt", default = 0.01, type = float)
    parser.add_argument("--njoints", default=3, type=int)
    parser.add_argument("--harmonics", default=5, type=int)
    parser.add_argument("--coef-max", default=1, type=int)
    
    args = parser.parse_args()
    
    T = args.T
    dt = args.dt
    n_joints = args.njoints
    harmonics = args.harmonics
    coefMax = args.coef_max
    q0 = np.zeros((n_joints,1))
    qd, qddot, qdddot = sinusoids_trajectory(n_joints, T, q0=q0, harmonics=harmonics, coefMax=coefMax)
    traj_size = int(T/dt)
    q, qdot, qdotdot = [],[],[]
    for t in range(1, traj_size+1):
        q.append(qd(t*dt))
        qdot.append(qddot(t*dt))
        qdotdot.append(qdddot(t*dt))
    
    fig, axs = plt.subplots(1,3)
    q = np.array(q).squeeze()
    qdot = np.array(qdot).squeeze()
    qdotdot = np.array(qdotdot).squeeze()
    axs[0].plot(q)
    axs[1].plot(qdot)
    axs[2].plot(qdotdot)
    
    plt.show()
    plt.close()
    