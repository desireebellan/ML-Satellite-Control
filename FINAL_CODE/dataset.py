# creare dataset

# python3.9
# Create the dataset to train the Parameter Identification  Neural Network



from robot import PlanarRoboticArm
from control import PD
from tqdm import tqdm
from math import sin, pi


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction
    from hyperparameters import HyperParams
    from gymnasium.spaces.box import Box
    from math import ceil
    from SPART.robot_model import urdf2robot
    from agent import Spacecraft
    
    import numpy as np
    
    import torch
    
    
    parser = ArgumentParser(description = "Create Spacecraft Parameter Identification Training Dataset")
    
    hparams = HyperParams

    parser.add_argument('--filename','-f', default = 'train', type = str, help = 'Dataset Filename', choices=["train", "val", "test"])
    parser.add_argument('--dir', default='./output/dataset', type=str, help='Dataset directory')
    parser.add_argument('-s', '--samples', default = 25000, type = int, help = 'Number of different payloads')

    parser.add_argument("--max-action", default = 30, type = float, help = "max torque value")
    parser.add_argument("--seed", default = 0, type = int, help = "seed value")
    parser.add_argument("--ckpt", default=None, type = str, help = "loading checkpoint path")
    
    parser.add_argument('--reset-control-rate', default = 1, type = int, help = "number of epochs before a desired trajectory update")
    parser.add_argument('--harmonics', default = 5, type = int, help = "maximum number of harmonics in the desired trajectory")
    
    parser.add_argument('--init', default = "constant", type = str, help = "how the environment is reset for each episode", 
                        choices=["constant", "random"])
    parser.add_argument('--reset-rate', default = 1, type = int, help = "number of epochs to change the initial state")
    
    parser.add_argument('--steps', default = 400, type = int, help = "number of steps saved")
    parser.add_argument('--max-time', default = 20, type = float, help = "time of execution of a single trajectory in seconds")
    parser.add_argument('--dt', default = 0.01, type = float, help = "time step of the environment")
    
    parser.add_argument('--rw', default = False, type=bool, help="if True the base reaction wheels are controlled", action=BooleanOptionalAction)
    parser.add_argument('--thrusters', default=False, type=bool, help="if True the base thrusters are controlled", action=BooleanOptionalAction)
    parser.add_argument('--rotation', default=False, type=bool, help="if True add rotation to inertia matrix", action=BooleanOptionalAction)
    
    args = parser.parse_args()
    
    args = parser.parse_args()
    
    # hyperparameters
    
    hparams.max_traj_size    = ceil((args.max_time // args.dt))
    #hparams.update_timestep  = hparams.max_traj_size * args.training_steps
    #hparams.update_epochs    = args.training_steps
    hparams.n_steps          = 1
    hparams.steps            = args.steps

    hparams.max_action       = args.max_action
    hparams.seed             = args.seed
    hparams.payload          = None
    hparams.estimate_inertia = False
    hparams.T                = args.max_time
    
    hparams.robot_filename   = "./matlab/SC_3DoF.urdf" 
    hparams.checkpoint_path  = args.dir + '/'
    
    hparams.device           = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    hparams.reset_control_rate = args.reset_control_rate
    hparams.harmonics        = args.harmonics
    
    hparams.device           = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #hparams.curriculum_learning = args.curriculum_learning
    hparams.start_hardness   = 0
    hparams.hardness_rate    = 0.1
    hparams.continuous_trajectory = False
    hparams.initialize       = args.init
    hparams.reset_rate       = args.reset_rate
    
    hparams.rw               = args.rw
    hparams.thrusters        = args.thrusters
    hparams.rotation         = args.rotation
    
    hparams.plot_reward      = False
    hparams.verbose          = False
    
    robot, _ = urdf2robot(filename=hparams.robot_filename)
    
    n = robot.n_q
    hparams.n_joints = n
    hparams.action_dim = n + 3*args.rw + 3*args.thrusters
        
    state_dim = 4 + 3*args.thrusters + 6 + n*2  
    hparams.state_dim        = state_dim
    observation_space = Box(low = -np.inf, high = np.inf, shape = (state_dim,), dtype = np.float32)
    action_space = Box(low = -hparams.max_action, high = hparams.max_action, shape = (n,))
    space_dim = {"observation-space":observation_space, "action-space":action_space}
        
    hparams.inertia_dim      = 1 + 3 + 3 + 3 * args.rotation
    hparams.is_train_inertia = False
    
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
    
    env.is_train_inertia = False
    
    env.create_dataset(samples=args.samples, filename=args.filename)