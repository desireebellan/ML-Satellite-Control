if __name__== "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction
    from hyperparameters import HyperParams
    from gymnasium.spaces.box import Box
    from math import ceil
    from SPART.robot_model import urdf2robot
    from agent import Spacecraft
    
    import numpy as np
    
    import torch
    
    #
    
    parser = ArgumentParser(description = "Recurrent PPO sb3")
    
    hparams = HyperParams
    
    parser.add_argument("--model-ckpt", default=None, type=str, help="pretrained weights checkpoint")
    parser.add_argument("--metric-path", default=None, type=str, help="path of saved metric")
    parser.add_argument("--dataset-dir", default="./output/dataset", type=str, help="Dataset directory")
    parser.add_argument("--patience", default=20, type=int, help="number of non-decreasing epochs before early stopping")
    parser.add_argument("--max-epochs", default=10000, type=int, help="maximum number of training epochs")
    parser.add_argument("--num-workers", default=1, type=int, help="number of training workers")
    
    parser.add_argument("--mode", default = "train", type = str, help = "script mode [train, test]", choices=["train", "test"])
    parser.add_argument("--max-action", default = 10, type = float, help = "max torque value")
    parser.add_argument("--seed", default = 0, type = int, help = "seed value")
    parser.add_argument("--ckpt", default=None, type = str, help = "loading checkpoint path")
    parser.add_argument('--batch-size', default = None, type = int, help = "size of update minibatch")
    parser.add_argument('--update-freq', default = 32, type = int, help = "inertia dataset size")
    
    parser.add_argument('--lr', default = 1e-3, type = float, help = "learning rate")
    parser.add_argument('--epochs', default=hparams.train_K, type=int, help="number of training epochs for each batch")
    parser.add_argument('--compute-variance', default = False, type = bool, action=BooleanOptionalAction)
    parser.add_argument('--num-layers', default = 1, type = int, help = "number of lstm layers inertia estimator")
    parser.add_argument('--hidden-dim', default=128, type=int, help="size lstm hidden layer")
    parser.add_argument('--feature-extraction', default = False, type = bool, help = "if True, a MLP layer is added after both the observation and the action ", \
        action=BooleanOptionalAction)
    parser.add_argument('--dropout', default = 0.1, type = float, help = "dropout probability ")
    parser.add_argument('--reduction', default = "sum", type=str, help = "loss reduction")
    parser.add_argument('--loss', default = "mse", type=str, help="loss function used", choices=["mse", "nmse"])
    parser.add_argument('--hw', default=False, type=bool, help="if True, use highway connections on the lstm layers", action=BooleanOptionalAction)
    parser.add_argument('--bidirectional', default=False, type=bool, action=BooleanOptionalAction)
    parser.add_argument('--optimizer', default="adam", type=str, help="optimizer class", choices=["adam", "sgd"])
    
    parser.add_argument('--reset-control-rate', default = 1, type = int, help = "number of epochs before a desired trajectory update")
    parser.add_argument('--harmonics', default = 5, type = int, help = "maximum number of harmonics in the desired trajectory")
    
    parser.add_argument('--hardness-rate', default = 50, type = int, help = "rate of hardness increase")
    parser.add_argument('--continuous-trajectory', default = False, type = bool, 
                        help = "if true, trajectory continuous until goal is reached", action=BooleanOptionalAction)
    parser.add_argument('--init', default = "constant", type = str, help = "how the environment is reset for each episode", 
                        choices=["constant", "random"])
    parser.add_argument('--start-hardness', default = 0.0, type = float, help = "initial hardness when using init = random")
    parser.add_argument('--reset-rate', default = 1, type = int, help = "number of epochs to change the initial state")
    
    parser.add_argument('--n-steps', default = 1, type = int, 
                        help = "number of steps taken by the environment for each action generated by the model")
    parser.add_argument('--max-time', default = 20, type = float, help = "time of execution of a single trajectory in seconds")
    parser.add_argument('--dt', default = 0.01, type = float, help = "time step of the environment")
    
    parser.add_argument('--rw', default = False, type=bool, help="if True the base reaction wheels are controlled", action=BooleanOptionalAction)
    parser.add_argument('--thrusters', default=False, type=bool, help="if True the base thrusters are controlled", action=BooleanOptionalAction)
    parser.add_argument('--rotation', default=False, type=bool, help="if True add rotation to inertia matrix", action=BooleanOptionalAction)
    
    args = parser.parse_args()
    
    # hyperparameters
    hparams.embedding_checkpoint = args.model_ckpt
    
    hparams.max_traj_size    = ceil((args.max_time // args.dt)/args.n_steps)
    #hparams.update_timestep  = hparams.max_traj_size * args.training_steps
    #hparams.update_epochs    = args.training_steps
    hparams.n_steps          = args.n_steps

    hparams.max_action       = args.max_action
    hparams.seed             = args.seed
    hparams.payload          = None
    hparams.update_payload_freq = 1
    hparams.estimate_inertia = True
    hparams.inertia_buffer_size= args.update_freq * hparams.max_traj_size
    #hparams.T               = hparams.max_traj_size * hparams.dt / 2
    hparams.T                = args.max_time
    
    hparams.robot_filename   = "./matlab/SC_3DoF.urdf"
    hparams.checkpoint_path  = './output/' 
    hparams.inertia_batch_size= None if args.batch_size is None else args.batch_size * hparams.max_traj_size
    hparams.lr               = args.lr 
    hparams.train_K          = args.epochs
    hparams.compute_variance = args.compute_variance
    hparams.optimizer_class  = args.optimizer
    
    hparams.mode             = args.mode
    
    hparams.min_reward       = -1e3
    
    hparams.num_layers       = args.num_layers
    hparams.hidden_dim       = args.hidden_dim
    hparams.feature_extraction = args.feature_extraction
    hparams.dropout          = args.dropout
    hparams.reduction        = args.reduction
    hparams.loss             = args.loss
    hparams.hw               = args.hw
    hparams.bidirectional    = args.bidirectional
    
    hparams.device           = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    hparams.reset_control_rate = args.reset_control_rate
    hparams.harmonics        = args.harmonics
    
    hparams.device           = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #hparams.curriculum_learning = args.curriculum_learning
    hparams.start_hardness   = args.start_hardness
    hparams.hardness_rate    = args.hardness_rate
    hparams.continuous_trajectory = args.continuous_trajectory
    hparams.initialize       = args.init
    hparams.reset_rate       = args.reset_rate
    
    hparams.plot_freq        = 10
    
    hparams.rw               = args.rw
    hparams.thrusters        = args.thrusters
    hparams.rotation         = args.rotation
    
    hparams.plot_reward      = False
    
    hparams.is_train_inertia = False
    
  
    filename = "./matlab/SC_3DoF.urdf"
    robot, _ = urdf2robot(filename=filename)
    n = robot.n_q
    hparams.n_joints = n
        
    state_dim = 7 + n*2
    observation_space = Box(low = -np.inf, high = np.inf, shape = (state_dim,), dtype = np.float32)
    action_space = Box(low = -hparams.max_action, high = hparams.max_action, shape = (n,))
    space_dim = {"observation-space":observation_space, "action-space":action_space}
        
    hparams.state_dim        = state_dim
    hparams.action_dim       = n
    hparams.inertia_dim      = 7

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
    env = Spacecraft(hparams, space_dim=space_dim, initial_state=initial_state, robot=robot, control = True)
    env.goal = np.array([0.,0.,0.,1.])
         
    '''    if args.ckpt is not None:
        print("Loading Environment data")
        
        env.load(args.ckpt + '/inertia_estimate')'''
                
    if args.mode == "train":
        print("---------------------------------------------------------")
        print("Learning Phase Started")
        with torch.autograd.set_detect_anomaly(True):
            trainer = env.train_inertia(dir = args.dataset_dir, max_epochs=args.max_epochs, batch_size=args.batch_size, 
                          metric_path=args.ckpt, patience=args.patience, num_workers=args.num_workers)
        ckpt_path = "best"
        print('---------------------------------------------------------')
        print('Learning phase compleated')
        
    elif args.mode == "test":   
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        accelerator = "gpu" if device == "cuda" else "cpu"
        trainer = None 
        ckpt_path = args.ckpt
        if args.ckpt is None : 
            raise Exception("Testing mode has been selected, but no checkpoint is defined!")    

    print('----------------------------------------------------------')
    print('Evaluation phase started')
    env.test_inertia(trainer, dir = args.dataset_dir, ckpt_path=ckpt_path, num_workers=args.num_workers)
    print('Evaluation phase compleated')