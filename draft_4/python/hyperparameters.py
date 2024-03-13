class HyperParams:
    is_discrete = True
    
    max_iter = int(1e7)
    max_traj_size = 2000
    update_timestep = 6*max_traj_size
    action_std_decay_freq = int(2.5e5)
    action_std_decay_rate = 0.05
    min_action_std = 0.01
    action_std = 0.6
    threshold_decay_rate = 0.5
    threshold_decay_freq = 500 * max_traj_size
    gamma = 0.99
    lamda = 0.95
    threshold = 0.01
    K_epochs = 10
    epsilon = 0.01
    reward = "inertia"
    update_advantage = False
    max_grad_norm = 0.5
    
    robot_name = "planar"  
    n_joints = 2
    state_dim = 3
    action_dim = 3
    hidden_dim = 256
    dt = 0.1
    
    inertia_ckpt = './output/epoch=38-valid_error=0.1752.ckpt'
    inertia_dim = 4
    gamma_p = 1
    
    device = 'cpu'
    save_model_freq = int(1e4) 
    
    seq_len = 5