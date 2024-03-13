from filterpy.kalman import UnscentedKalmanFilter
from SPART_CASADI.attitude_transformations import quat_DCM, Angles321_DCM
from SPART.attitude_transformations import Euler_Angles321
from SPART_CASADI.kinematics import Kinematics, DiffKinematics, Velocities
from SPART_CASADI.dynamics import I_I, FD
from SPART_CASADI.robot_model import urdf2robot
from SPART_CASADI.utils import *
from agent import Planar2R
from math import pi
import numpy as np
from hyperparameters import HyperParams

def setParams(params:dict, robot:AttributeDict):
    if "mp" not in params.keys() or "rp" not in params.keys() or "Ip" not in params.keys():
        raise Exception("Params in the wrong format!")
    robot.links[-1].mass = params["mp"]
    robot.links[-1].inertia = params["Ip"]
    robot.links[-1].T = blockcat([[SX.eye(3), params["rp"].reshape((3, 1))],[SX.zeros((1,3)), 1]])
    return robot

def spacecraftFD(tau0, taum, q0, qm, u0, um, robot, mp, rp, Ip):
    robot = setParams({'mp':mp, 'rp': rp, 'Ip': Ip}, robot)
    R0 = transpose(Angles321_DCM(transpose(q0[:3,:].reshape((-1,1)))))
    r0 = q0[3:, :]
    _,RL,_,rL,e,g = Kinematics(R0, r0, qm, robot) 
    Bij, Bi0, P0, pm = DiffKinematics(R0, r0, rL, e, g, robot)
    t0, tL = Velocities(Bij, Bi0, P0, pm, u0, um, robot)
    #t0dot, tLdot = Accelerations(t0, tL, P0, pm, Bi0, Bij, u0, um, u0dot, umdot, robot)
    I0, Im = I_I(R0, RL, robot)
    u0dot, umdot = FD(tau0, taum, np.zeros((6, 1)), np.zeros((6, robot.n_links_joints)), t0, tL, P0, pm, I0, Im, Bij, Bi0, u0, um, robot)    
    return u0dot, umdot

def unscented_kalman_filter(x0, P0, Q, R, N, dynamic_model, measurement_model, dt, num_steps):
    """
    Unscented Kalman Filter for N states.

    Parameters:
        x0: Initial state estimate
        P0: Initial state covariance matrix
        Q: Process noise covariance matrix
        R: Measurement noise covariance matrix
        N: Number of states
        dynamic_model: Function defining the dynamic model
        measurement_model: Function defining the measurement model
        dt: Time step
        num_steps: Number of time steps

    Returns:
        Fisher matrix
    """
    alpha = 1e-3
    beta = 2
    kappa = 0

    # Calculate sigma points
    lambda_ = alpha ** 2 * (N + kappa) - N
    sigma_points = np.zeros((2 * N + 1, N))
    Wm = np.zeros(2 * N + 1)
    Wc = np.zeros(2 * N + 1)

    for i in range(2 * N + 1):
        sigma_points[i, :] = x0
        Wm[i] = 1 / (2 * (N + lambda_))
        Wc[i] = 1 / (2 * (N + lambda_))

    sigma_points[1:N+1, :] += np.sqrt(N + lambda_) * np.linalg.cholesky(P0).T
    sigma_points[N+1:, :] -= np.sqrt(N + lambda_) * np.linalg.cholesky(P0).T

    # Initialize variables
    x_hat = x0
    P_hat = P0
    fisher_matrix = np.zeros((N, N))

    for k in range(num_steps):
        # Prediction step
        sigma_points = dynamic_model(sigma_points, dt)
        x_hat, P_hat = predict(sigma_points, Wm, Wc, Q)

        # Update step
        z = measurement_model(x_hat)
        fisher_matrix += compute_fisher_matrix(sigma_points, Wm, Wc, x_hat, P_hat, R, measurement_model, z)

    return fisher_matrix

def predict(sigma_points, Wm, Wc, Q):
    """
    Predict step of the Unscented Kalman Filter.

    Parameters:
        sigma_points: Sigma points
        Wm: Weights for mean
        Wc: Weights for covariance
        Q: Process noise covariance matrix

    Returns:
        Predicted mean and covariance
    """
    N = sigma_points.shape[1]
    x_hat = sum(Wm[i] * sigma_points[i, :] for i in range(2 * N + 1))
    P_hat = Q + sum(Wc[i] * np.outer(sigma_points[i, :] - x_hat, sigma_points[i, :] - x_hat) for i in range(2 * N + 1))

    return x_hat, P_hat

def compute_fisher_matrix(sigma_points, Wm, Wc, x_hat, P_hat, R, measurement_model, z):
    """
    Compute Fisher matrix for the Unscented Kalman Filter update step.

    Parameters:
        sigma_points: Sigma points
        Wm: Weights for mean
        Wc: Weights for covariance
        x_hat: Predicted mean
        P_hat: Predicted covariance
        R: Measurement noise covariance matrix
        measurement_model: Function defining the measurement model
        z: Measurement vector

    Returns:
        Fisher matrix for the update step
    """
    N = sigma_points.shape[1]
    H = np.zeros((len(z), N))

    for i in range(2 * N + 1):
        z_sigma = measurement_model(sigma_points[i, :])
        H += Wc[i] * np.outer(z_sigma - z, sigma_points[i, :] - x_hat)

    S = R + H @ P_hat @ H.T
    K = P_hat @ H.T @ np.linalg.inv(S)

    fisher_matrix = K.T @ np.linalg.inv(S) @ K

    return fisher_matrix

def MPC():
    
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)
    
    filename = "./matlab/SC_3DoF.urdf"
    robot, _ = urdf2robot(filename=filename)
    n = robot.n_q
     
    qb = model.set_variable(var_type='_x', var_name = 'qb', shape=(3,1))
    rb = model.set_variable(var_type='_x', var_name = 'rb', shape=(3,1))
    qm = model.set_variable(var_type='_x', var_name = 'qm', shape=(n,1))
    dqb = model.set_variable(var_type='_x', var_name = 'dqb', shape=(3,1))
    drb = model.set_variable(var_type='_x', var_name = 'drb', shape=(3,1))
    um = model.set_variable(var_type='_x', var_name = 'um', shape=(n,1))  
    taum = model.set_variable(var_type='_u', var_name = 'taum', shape=(n,1))
    taub = model.set_variable(var_type='_u', var_name = 'taub', shape=(3,1))
    
    mp = model.set_variable(var_type='parameter', var_name='mp', shape=(1,1))
    '''rpx = model.set_variable(var_type='parameter', var_name='rpx', shape=(1,1))
    rpy = model.set_variable(var_type='parameter', var_name='rpy', shape=(1,1))
    rpz = model.set_variable(var_type='parameter', var_name='rpz', shape=(1,1))
    Ipxx = model.set_variable(var_type='parameter', var_name='Ipxx', shape=(1,1))
    Ipyy = model.set_variable(var_type='parameter', var_name='Ipyy', shape=(1,1))
    Ipzz = model.set_variable(var_type='parameter', var_name='Ipzz', shape=(1,1))
    
    rp = vertcat(rpx, rpy, rpz)
    Ip = diag(vertcat(Ipxx, Ipyy, Ipzz))'''
    
    #mp = np.array([5.0])
    rpx = np.array([0.5])
    rpy= np.array([0.0])
    rpz= np.array([0.0])
    Ipxx = np.array([3.0])
    Ipyy = np.array([7.0])
    Ipzz = np.array([4.5])
    rp = np.vstack((rpx, rpy, rpz))
    Ip = np.diag(np.vstack((Ipxx, Ipyy, Ipzz)))
    
    tau0 = vertcat(taub, SX.zeros((3,1)))
    
    ubdot, umdot = spacecraftFD(tau0=tau0, taum=taum, q0=vertcat(qb, rb), qm=qm, u0=vertcat(dqb, drb), um=um, mp=mp, rp=rp, Ip=Ip, robot=robot)
    model.set_rhs('um', umdot)
    model.set_rhs('dqb', ubdot[:3,:])
    model.set_rhs('drb', ubdot[3:,:])
    model.set_rhs('qm', um)

    #model.set_rhs('qb', 1/2*quat_product(qb, vertcat(dqb, SX.zeros(1))))
    model.set_rhs('qb', dqb)
    model.set_rhs('rb', drb)
     
    model.setup()
    
    mpc = do_mpc.controller.MPC(model)
    
    T = 15 #
    dt = 0.1 # s
    
    setup_mpc = {
        'n_horizon': int(T/dt),
        't_step': dt,
        'n_robust': 0,
        'open_loop':0,
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
    }
    mpc.set_param(**setup_mpc)
    
    #lterm = qb[0]**2 + qb[1]**2 + qb[2]**2
    #mterm = qb[0]**2 + qb[1]**2 + qb[2]**2
    
    lterm = dqb[0]**2 + dqb[1]**2 + dqb[2]**2
    mterm = dqb[0]**2 + dqb[1]**2 + dqb[2]**2
    
    mpc.set_objective(lterm=lterm, mterm=mterm)

    mpc.set_rterm(
        taum=1e-2,
        taub=1e-2
    )
    
    # Lower bounds on states:
    mpc.bounds['lower', '_x', 'qm'] = -2*pi
    mpc.bounds['lower', '_x', 'um'] = -4*pi
    # Upper bounds on states
    mpc.bounds['upper','_x', 'qm'] = 2*pi
    mpc.bounds['upper', '_x', 'um'] = 4*pi

    # Lower bounds on inputs:
    mpc.bounds['lower', '_u', 'taum'] = -200*pi 
    # Lower bounds on inputs:
    mpc.bounds['upper', '_u', 'taum'] = 200*pi 
    
    mp_values = np.arange(0.0, 10.0, 2)
    #mp_values = np.array([0.0, 10.0])
    #mp_values = np.array([5.0])
    '''mp_values = np.array([5.0])
    rpx_values = np.array([0.5])
    rpy_values = np.array([0.0])
    rpz_values = np.array([0.0])
    Ipxx_values = np.array([3.0])
    Ipyy_values = np.array([7.0])
    Ipzz_values = np.array([4.5])'''
    
    mpc.scaling['_x', 'qm'] = 2
    mpc.scaling['_x', 'qb'] = 2

    mpc.set_uncertainty_values(
        mp = mp_values    
    )
    '''rpx = rpx_values,
        rpy = rpy_values,
        rpz = rpz_values,
        Ipxx = Ipxx_values,
        Ipyy = Ipyy_values,
        Ipzz = Ipzz_values'''
    
    mpc.setup()
    
    simulator = do_mpc.simulator.Simulator(model)
    
    # Instead of supplying a dict with the splat operator (**), as with the optimizer.set_param(),
    # we can also use keywords (and call the method multiple times, if necessary):
    params_simulator = {
        # Note: cvode doesn't support DAE systems.
        'integration_tool': 'idas',
        'abstol': 1e-3,
        'reltol': 1e-3,
        't_step': dt,
    }   

    simulator.set_param(**params_simulator)
    
    p_template = simulator.get_p_template()
    
    # case constant parameters
    def p_fun(t_now):    
        p_template['mp'] = 2.0 if t_now <= 5 else 8.0
        '''p_template['rpx'] = 0.3
        p_template['rpy'] = 0.0
        p_template['rpz'] = 0.0 
        p_template['Ipxx'] = 3.0 
        p_template['Ipyy'] = 7.0 
        p_template['Ipzz'] = 4.5'''
        return p_template
    
    simulator.set_p_fun(p_fun)
    simulator.setup()
    
    initial_state = {
        #'qb' : euler2quat(np.ones((3, 1)), pi/4).reshape((-1,1)).astype(np.float32), # base quaternion
        'qb': Euler_Angles321(np.ones((3, 1)), pi/4).reshape((-1,1)).astype(np.float32),
        'rb' : np.zeros((3,1)).astype(np.float32),
        #'qm': np.array([0, 5/4*pi, -5/4*pi]).reshape((n, 1)).astype(np.float32), # Joint variables [rad]
        'qm': np.zeros((n,1)).astype(np.float32),
        'dqb': np.array([0.2, -0.15, 0.18]).reshape((3,1)).astype(np.float32), # Base-spacecraft velocity
        #'dqb': 1e-8*np.ones((3,1)).astype(np.float32),
        'drb': np.zeros((3,1)).astype(np.float32),
        #'u0': np.zeros((6,1)).astype(np.float32),
        'um': np.zeros((n, 1)).astype(np.float32) # Joint velocities
        }
    
    x0 = np.vstack([initial_state["qb"], initial_state["rb"], initial_state["qm"], initial_state["dqb"], initial_state['drb'], initial_state["um"]]).reshape(-1,1)
    simulator.x0 = x0
    mpc.x0 = x0
    mpc.set_initial_guess()
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    # Customizing Matplotlib:
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['axes.grid'] = True    
    
    mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
    sim_graphics = do_mpc.graphics.Graphics(simulator.data)
    
    # We just want to create the plot and not show it right now. This "inline magic" supresses the output.
    fig, ax = plt.subplots(4, sharex=True, figsize=(32,9))
    fig.align_ylabels()

    for g in [sim_graphics, mpc_graphics]:
        # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
        g.add_line(var_type='_x', var_name='qm', axis=ax[0])
        #g.add_line(var_type='_x', var_name='phi_2', axis=ax[0])
        #g.add_line(var_type='_x', var_name='phi_3', axis=ax[0])
        #g.add_line(var_type='_x', var_name='qb', axis=ax[1])
        g.add_line(var_type='_x', var_name='dqb', axis=ax[1])

        # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
        g.add_line(var_type='_u', var_name='taum', axis=ax[2])
        g.add_line(var_type='_u', var_name='taub', axis=ax[3])
        #g.add_line(var_type='_u', var_name='phi_m_2_set', axis=ax[1])


        ax[0].set_ylabel(r'$q_m$ [rad]')
        ax[1].set_ylabel(r'$\dot{q}_b$ [rad]')
        ax[2].set_ylabel(r'$\tau_m$ [rad]')
        ax[3].set_ylabel(r'$\tau_b$ [rad]')
        ax[3].set_xlabel('time [s]')    
        
    '''taum = np.zeros((3,1))
    from tqdm import tqdm
    print("Starting simulation ...")
    for i in tqdm(range(8000)):
        simulator.make_step(taum)
    print("Simulation ended. ")
        
    sim_graphics.plot_results()
    # Reset the limits on all axes in graphic to show the data.
    sim_graphics.reset_axes()
    # Show the figure:
    plt.show()
    fig.save('./output/images/mpc.png')'''
    
    u0 = mpc.make_step(x0)
    mpc_graphics.plot_predictions()
    mpc_graphics.reset_axes()
    plt.show()
    fig.savefig('./output/images/mpc.png')
    
    '''simulator.reset_history()
    simulator.x0 = x0
    mpc.reset_history()

    for i in range(20):
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0)'''
        
def UKF():
    
    from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
    from filterpy.common import Q_discrete_white_noise
    from gymnasium.spaces.box import Box
    
    # 1DOF robotic arm 
    # [x,xdot, xddot, I] = [x + xdot*dt, xdot + xddot, I(Kp(x-xd) - Kd*xdot), I]
    
    hparams = HyperParams()
    shape = 4 + 2
    n = 2
    observation_space = Box(low = -np.inf, high = np.inf, shape = (shape + n,), dtype = np.float32)
    action_space = Box(low = -hparams.max_action, high = hparams.max_action, shape = (n,1))
    hparams.n_joints = 2

    space_dim = {"observation-space":observation_space, "action-space":action_space}

    # initialize environment
    initial_state = {
            'q':np.zeros((hparams.n_joints,1)).astype(np.float32),
            'qdot':np.zeros((hparams.n_joints,1)).astype(np.float32),
            'qdotdot':np.zeros((hparams.n_joints,1)).astype(np.float32),
            'tau':np.zeros((hparams.n_joints,1)).astype(np.float32)
            }
    # taken from "Payload Estimation Based on Identified Coefficients of Robot Dynamics —with an Application to Collision Detection" 
    # https://www.diag.uniroma1.it/~labrob/pub/papers/IROS17_PayloadEstimation_2199.pdf
    robot = {
            'm': [3, 2], #kg
            'l': [1, 0.5], #m
            'rcom': [[-0.6, 0.01], [-0.2, 0.02]], #m
            'I': [1.3303, 0.1225] #kg*m^2
            }
    payload_ranges = {'m':[0.,5.]}
    robot = Planar2R(hparams, space_dim=space_dim, initial_state=initial_state, robot=robot, payload_ranges=payload_ranges)
    
    def fx(x, dt):
        # state transition function - predict next state based
        # on constant velocity model x = vt + x_0    
        x[2] = x[2]
        tau = (0.3*abs(x[0] - pi/2) - 0.3*x[1])
        xddot = tau/x[2]
        #x[1] = 
        return np.dot(F, x)
    
    def hx(x):
        # measurement function - convert state into a measurement
        # where measurements are [x_pos, y_pos]
        return np.array([x[0], x[2]])
    
    X = 4
    Z = 2
    ALPHA = .1
    BETA=2.
    KAPPA=-1
    dt=0.1
    # create sigma points to use in the filter. This is standard for Gaussian processes
    points = MerweScaledSigmaPoints(X, alpha=ALPHA, beta=BETA, kappa=KAPPA)
    
    kf = UnscentedKalmanFilter(dim_x=X, dim_z=Z, dt=dt, fx=fx, hx=hx, points=points)
    
    kf.x = np.array([-1., 1., -1., 1]) # initial state
    kf.P *= 0.2 # initial uncertainty
    z_std = 0.1
    kf.R = np.diag([z_std**2, z_std**2]) # 1 standard
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01**2, block_size=2)
    zs = [[i+np.random.randn()*z_std, i+np.random.randn()*z_std] for i in range(50)] # measurements
    for z in zs:
        kf.predict()
        kf.update(z)
        print(kf.x, 'log-likelihood', kf.log_likelihood)
    
    
def test_MPC():
    
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)
    
    phi_1 = model.set_variable(var_type='_x', var_name='phi_1', shape=(1,1))
    phi_2 = model.set_variable(var_type='_x', var_name='phi_2', shape=(1,1))
    phi_3 = model.set_variable(var_type='_x', var_name='phi_3', shape=(1,1))
    # Variables can also be vectors:
    dphi = model.set_variable(var_type='_x', var_name='dphi', shape=(3,1))
    # Two states for the desired (set) motor position:
    phi_m_1_set = model.set_variable(var_type='_u', var_name='phi_m_1_set')
    phi_m_2_set = model.set_variable(var_type='_u', var_name='phi_m_2_set')
    # Two additional states for the true motor position:
    phi_1_m = model.set_variable(var_type='_x', var_name='phi_1_m', shape=(1,1))
    phi_2_m = model.set_variable(var_type='_x', var_name='phi_2_m', shape=(1,1))
    
    # As shown in the table above, we can use Long names or short names for the variable type.
    Theta_1 = model.set_variable('parameter', 'Theta_1')
    Theta_2 = model.set_variable('parameter', 'Theta_2')
    Theta_3 = model.set_variable('parameter', 'Theta_3')

    c = np.array([2.697,  2.66,  3.05, 2.86])*1e-3
    d = np.array([6.78,  8.01,  8.82])*1e-5
    
    model.set_rhs('phi_1', dphi[0])
    model.set_rhs('phi_2', dphi[1])
    model.set_rhs('phi_3', dphi[2])
    
    dphi_next = vertcat(
        -c[0]/Theta_1*(phi_1-phi_1_m)-c[1]/Theta_1*(phi_1-phi_2)-d[0]/Theta_1*dphi[0],
        -c[1]/Theta_2*(phi_2-phi_1)-c[2]/Theta_2*(phi_2-phi_3)-d[1]/Theta_2*dphi[1],
        -c[2]/Theta_3*(phi_3-phi_2)-c[3]/Theta_3*(phi_3-phi_2_m)-d[2]/Theta_3*dphi[2],
    )

    model.set_rhs('dphi', dphi_next)
    
    tau = 1e-2
    model.set_rhs('phi_1_m', 1/tau*(phi_m_1_set - phi_1_m))
    model.set_rhs('phi_2_m', 1/tau*(phi_m_2_set - phi_2_m))
    
    model.setup()
    
    mpc = do_mpc.controller.MPC(model)
    
    setup_mpc = {
        'n_horizon': 20,
        't_step': 0.1,
        'n_robust': 1,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)
    
    mterm = phi_1**2 + phi_2**2 + phi_3**2
    lterm = phi_1**2 + phi_2**2 + phi_3**2

    mpc.set_objective(mterm=mterm, lterm=lterm)
    
    mpc.set_rterm(
        phi_m_1_set=1e-2,
        phi_m_2_set=1e-2
    )
    
    # Lower bounds on states:
    mpc.bounds['lower','_x', 'phi_1'] = -2*np.pi
    mpc.bounds['lower','_x', 'phi_2'] = -2*np.pi
    mpc.bounds['lower','_x', 'phi_3'] = -2*np.pi
    # Upper bounds on states
    mpc.bounds['upper','_x', 'phi_1'] = 2*np.pi
    mpc.bounds['upper','_x', 'phi_2'] = 2*np.pi
    mpc.bounds['upper','_x', 'phi_3'] = 2*np.pi

    # Lower bounds on inputs:
    mpc.bounds['lower','_u', 'phi_m_1_set'] = -2*np.pi
    mpc.bounds['lower','_u', 'phi_m_2_set'] = -2*np.pi
    # Lower bounds on inputs:
    mpc.bounds['upper','_u', 'phi_m_1_set'] = 2*np.pi
    mpc.bounds['upper','_u', 'phi_m_2_set'] = 2*np.pi
    
    mpc.scaling['_x', 'phi_1'] = 2
    mpc.scaling['_x', 'phi_2'] = 2
    mpc.scaling['_x', 'phi_3'] = 2
    
    inertia_mass_1 = 2.25*1e-4*np.array([1., 0.9, 1.1])
    inertia_mass_2 = 2.25*1e-4*np.array([1., 0.9, 1.1])
    inertia_mass_3 = 2.25*1e-4*np.array([1.])

    mpc.set_uncertainty_values(
        Theta_1 = inertia_mass_1,
        Theta_2 = inertia_mass_2,
        Theta_3 = inertia_mass_3
    )
    
    mpc.setup()
    
    simulator = do_mpc.simulator.Simulator(model)
    
    # Instead of supplying a dict with the splat operator (**), as with the optimizer.set_param(),
    # we can also use keywords (and call the method multiple times, if necessary):
    simulator.set_param(t_step = 0.1)
    
    p_template = simulator.get_p_template()
    def p_fun(t_now):    
        p_template['Theta_1'] = 2.25e-4
        p_template['Theta_2'] = 2.25e-4
        p_template['Theta_3'] = 2.25e-4
        return p_template
    
    simulator.set_p_fun(p_fun)
    simulator.setup()
    
    np.pi*np.array([1, 1, -1.5, 1, -1, 1, 0, 0]).reshape(-1,1)
    simulator.x0 = x0
    mpc.x0 = x0
    mpc.set_initial_guess()
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    # Customizing Matplotlib:
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['axes.grid'] = True    
    
    mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
    sim_graphics = do_mpc.graphics.Graphics(simulator.data)
    
    # We just want to create the plot and not show it right now. This "inline magic" supresses the output.
    fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
    fig.align_ylabels()

    for g in [sim_graphics, mpc_graphics]:
        # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
        g.add_line(var_type='_x', var_name='phi_1', axis=ax[0])
        g.add_line(var_type='_x', var_name='phi_2', axis=ax[0])
        g.add_line(var_type='_x', var_name='phi_3', axis=ax[0])

        # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
        g.add_line(var_type='_u', var_name='phi_m_1_set', axis=ax[1])
        g.add_line(var_type='_u', var_name='phi_m_2_set', axis=ax[1])


        ax[0].set_ylabel('angle position [rad]')
        ax[1].set_ylabel('motor angle [rad]')
        ax[1].set_xlabel('time [s]')    
        
    u0 = np.zeros((2,1))
    for i in range(200):
        simulator.make_step(u0)
        
    sim_graphics.plot_results()
    # Reset the limits on all axes in graphic to show the data.
    sim_graphics.reset_axes()
    # Show the figure:
    #plt.show()
    u0 = mpc.make_step(x0)
    sim_graphics.clear()
    mpc_graphics.plot_predictions()
    mpc_graphics.reset_axes()
    #plt.show()
    
    # Change the color for the three states:
    for line_i in mpc_graphics.pred_lines['_x', 'phi_1']: line_i.set_color('#1f77b4') # blue
    for line_i in mpc_graphics.pred_lines['_x', 'phi_2']: line_i.set_color('#ff7f0e') # orange
    for line_i in mpc_graphics.pred_lines['_x', 'phi_3']: line_i.set_color('#2ca02c') # green
    # Change the color for the two inputs:
    for line_i in mpc_graphics.pred_lines['_u', 'phi_m_1_set']: line_i.set_color('#1f77b4')
    for line_i in mpc_graphics.pred_lines['_u', 'phi_m_2_set']: line_i.set_color('#ff7f0e')

    # Make all predictions transparent:
    for line_i in mpc_graphics.pred_lines.full: line_i.set_alpha(0.2)
    
    # Get line objects (note sum of lists creates a concatenated list)
    lines = sim_graphics.result_lines['_x', 'phi_1']+sim_graphics.result_lines['_x', 'phi_2']+sim_graphics.result_lines['_x', 'phi_3']

    ax[0].legend(lines,'123',title='disc')

    # also set legend for second subplot:
    lines = sim_graphics.result_lines['_u', 'phi_m_1_set']+sim_graphics.result_lines['_u', 'phi_m_2_set']
    ax[1].legend(lines,'12',title='motor')
    
    simulator.reset_history()
    simulator.x0 = x0
    mpc.reset_history()

    for i in range(20):
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0)
        
    # Plot predictions from t=0
    mpc_graphics.plot_predictions(t_ind=0)
    # Plot results until current time
    sim_graphics.plot_results()
    sim_graphics.reset_axes()
    #plt.show()
    
    # save results
    '''from do_mpc.data import save_results, load_results
    # Note that by default results are stored in the subfolder results under the name results.pkl. Both can be changed and the folder is created if it doesn’t exist already.
    save_results([mpc, simulator])
    with open(file_name, 'rb') as f:
        results = pickle.load(f)
    results = load_results('./results/results.pkl')'''
    
    from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter

    def update(t_ind):
        sim_graphics.plot_results(t_ind)
        mpc_graphics.plot_predictions(t_ind)
        mpc_graphics.reset_axes()
    anim = FuncAnimation(fig, update, frames=20, repeat=False)
    gif_writer = ImageMagickWriter(fps=3)
    anim.save('anim.gif', writer=gif_writer)

if __name__=="__main__":

    # Example usage:
    # Define dynamic_model and measurement_model functions
    '''def dynamic_model(sigma_points, dt):
        # Define your dynamic model here
        return sigma_points

    def measurement_model(x):
        # Define your measurement model here
        return np.array([x[0]])

    # Set initial parameters
    N = 1  # Number of states
    x0 = np.array([0.0])
    P0 = np.eye(N)
    Q = np.eye(N)
    R = np.eye(1)
    dt = 0.1
    num_steps = 100

    # Compute Fisher matrix
    fisher_matrix = unscented_kalman_filter(x0, P0, Q, R, N, dynamic_model, measurement_model, dt, num_steps)

    print("Fisher Matrix:")
    print(fisher_matrix)'''
    
    '''from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
    from filterpy.common import Q_discrete_white_noise
    
    def fx(x, dt):
        # state transition function - predict next state based
        # on constant velocity model x = vt + x_0
        F = np.array([[1, dt, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, dt],
                    [0, 0, 0, 1]], dtype=float)
        return np.dot(F, x)
    
    def hx(x):
        # measurement function - convert state into a measurement
        # where measurements are [x_pos, y_pos]
        return np.array([x[0], x[2]])
    
    X = 4
    Z = 2
    ALPHA = .1
    BETA=2.
    KAPPA=-1
    dt=0.1
    # create sigma points to use in the filter. This is standard for Gaussian processes
    points = MerweScaledSigmaPoints(X, alpha=ALPHA, beta=BETA, kappa=KAPPA)
    
    kf = UnscentedKalmanFilter(dim_x=X, dim_z=Z, dt=dt, fx=fx, hx=hx, points=points)
    
    kf.x = np.array([-1., 1., -1., 1]) # initial state
    kf.P *= 0.2 # initial uncertainty
    z_std = 0.1
    kf.R = np.diag([z_std**2, z_std**2]) # 1 standard
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01**2, block_size=2)
    zs = [[i+np.random.randn()*z_std, i+np.random.randn()*z_std] for i in range(50)] # measurements
    for z in zs:
        kf.predict()
        kf.update(z)
        print(kf.x, 'log-likelihood', kf.log_likelihood)'''
        
    import do_mpc
    from casadi import *
    MPC()