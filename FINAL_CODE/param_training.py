from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.spaces.box import Box

from SPART.robot_model import urdf2robot
from SPART.attitude_transformations import Euler_Angles321
from agent import Spacecraft
from hyperparameters import HyperParams

import numpy as np
import torch
import pytorch_lightning as pl

from math import pi, ceil

class CostumeCallback(BaseCallback):
    def __init__(self):
        super(CostumeCallback, self).__init__()
        self.rewards = []
    def on_rollout_end(self):
        self.rewards.append(self.locals["reward"])
        
if __name__ == "__main__":
    
    
    from argparse import ArgumentParser, BooleanOptionalAction
    
    hparams = HyperParams
    
    parser = ArgumentParser(description = "PPO sb3")
    
    parser.add_argument("--mode", default = "train", type = str, help = "script mode [train, test]", choices=["train", "test", "eval-inertia", "render"])
    parser.add_argument("--reward", default="attitude-stabilization", type = str, help="type of reward", choices=["attitude-stabilization", "attitude-detumbling", "ee-tracking"])
    parser.add_argument("--dataset-dir", default="dataset", type=str, help="dataset used to train the inertia estimator", choices=["dataset", "dataset_rotation", "dataset_rw", "dataset_rw_thrusters"])
    parser.add_argument("--max-action", default = 30, type = float, help = "max torque value")
    parser.add_argument("--seed", default = 0, type = int, help = "seed value")
    
    # TRAJECTORY SPECIFICS
    
    parser.add_argument('--n-steps', default = 5, type = int, help = "number of steps taken by the environment for each action generated by the model")
    parser.add_argument('--max-time', default = 20, type = float, help = "time of execution of a single trajectory in seconds")
    parser.add_argument('--dt', default = 0.01, type = float, help = "time step of the environment")
    parser.add_argument('--rotation', default=False, type=bool, help="if True add rotation to inertia matrix", action=BooleanOptionalAction)
    
    # LEARNING & CURRICULUM LEARNING 
    
    #parser.add_argument("--control-ckpt", default=None, type = str, help = "loading ppo checkpoint path")
    #parser.add_argument("--inertia-ckpt", default=None, type=str, help="pretrained weights checkpoint")
    parser.add_argument('--ckpt', default=None, type=str, help="directory of pretrained controller and inertia estimator")
    parser.add_argument('--training-steps', default = 6, type = int, help = "number of episodes before learning")
    parser.add_argument('--epochs', default=hparams.train_K, type=int, help="number of training epochs for each batch")
    parser.add_argument('--hardness-rate', default = 50, type = int, help = "rate of hardness increase")
    parser.add_argument('--continuous-trajectory', default = False, type = bool, help = "if true, trajectory continuous until goal is reached", action=BooleanOptionalAction)
    parser.add_argument('--init', default = "constant", type = str, help = "how the environment is reset for each episode", choices=["constant", "random"])
    parser.add_argument('--start-hardness', default = 0.0, type = float, help = "initial hardness when using init = random")
    parser.add_argument('--reset-rate', default = 1, type = int, help = "number of epochs to change the initial state")
    parser.add_argument('--max-epochs', default=10000, type=int, help="maximum number of training epochs")
    parser.add_argument("--num-workers", default=1, type=int, help="number of training workers")
    parser.add_argument('--inertia-lr', default = 1e-3, type = float, help = "learning rate")
    parser.add_argument('--control-lr', default = 3e-5, type = float, help = "learning rate")

    
    # NETWORK SPECIFICS
    
    parser.add_argument('--inertia-num-layers', default = 1, type = int, help = "number of lstm layers inertia estimator")
    parser.add_argument('--inertia-hidden-dim', default=512, type=int, help="size lstm hidden layer")
    parser.add_argument("--inertia-batch-size", default=128, type=int, help="batch size inertia training procedure")
    parser.add_argument('--control-num-layers', default = 1, type = int, help = "number of lstm layers inertia estimator")
    parser.add_argument('--control-hidden-dim', default=256, type=int, help="hidden size agent network")
    parser.add_argument("--control-batch-size", default=128, type=int, help="controller batch size")
    parser.add_argument('--shared-lstm', default=False, type=bool, help="if True, the agent and critic lstm are shared", action=BooleanOptionalAction)
    parser.add_argument('--inertia-feature-extraction', default = False, type = bool, help = "if True, a MLP layer is added after both the observation and the action ", \
        action=BooleanOptionalAction)
    parser.add_argument('--dropout', default = 0.1, type = float, help = "dropout probability ")
    parser.add_argument('--reduction', default = "sum", type=str, help = "loss reduction")
    parser.add_argument('--loss', default = "mse", type=str, help="loss function used", choices=["mse", "nmse"])
    parser.add_argument('--hw', default=False, type=bool, help="if True, use highway connections on the lstm layers", action=BooleanOptionalAction)
    parser.add_argument('--bidirectional', default=False, type=bool, action=BooleanOptionalAction)
    parser.add_argument('--optimizer', default="adam", type=str, help="optimizer class", choices=["adam", "sgd"])
    
    # GAMMA 
    parser.add_argument('--gamma', default=1.0, type=float, help="initial value of gamma")
    parser.add_argument('--gamma-rate', default=0.995, type=float, help="decreasing rate of gamma")
    
    
    args = parser.parse_args()

    assert not((args.reward == 'attitude-stabilization') ^ (args.dataset_dir in {'dataset', 'dataset_rotation'}))
    assert not((args.reward == 'attitude-detumbling') ^ (args.dataset_dir == 'dataset_rw'))
    assert not((args.reward == 'ee-tracking') ^ (args.dataset_dir == 'dataset_rw_thrusters'))
    
    control_name_prefix  = 'control_' + args.reward + '_shared=' + ("True" if args.shared_lstm else "False") 
    control_name_prefix += "_layers=" + str(args.control_num_layers) + '_init=' + args.init + '_maxtime=' + str(args.max_time) + '_payload=True' 
    
    inertia_name_prefix = 'inertia_' + args.dataset_dir + '_batch-size=' + str(args.inertia_batch_size) + '_hw=' + ("True" if args.hw else "False") + '_loss=' + args.loss + '_num-layers=' + str(args.inertia_num_layers)
    inertia_name_prefix += '_lr=' + str(args.inertia_lr)+ '_dropout=' + str(args.dropout) + '_bidirectional=' + ('True' if args.bidirectional else 'False') + '_hidden-dim=' + str(args.inertia_hidden_dim)
    
    dataset_dir = './output/' + args.dataset_dir
    
    pretrain = True
    
    rw = True if args.reward in {"attitude-detumbling", "ee-tracking"} else False
    thrusters = True if args.reward == "ee-tracking" else False
    
    # hyperparameters
    hparams.checkpoint_path = args.ckpt + '/inertia_control'
    
    hparams.max_traj_size    = ceil((args.max_time // args.dt)/args.n_steps)
    hparams.update_timestep  = hparams.max_traj_size * args.training_steps
    hparams.update_epochs    = args.training_steps
    hparams.n_steps          = args.n_steps
    
    hparams.reward           = args.reward
    hparams.max_action       = args.max_action
    hparams.seed             = args.seed
    hparams.payload          = None
    
    hparams.robot_filename    = "./matlab/SC_3DoF.urdf"
    hparams.checkpoint_dir    = args.ckpt
    hparams.embedding_checkpoint = args.ckpt + '/' + inertia_name_prefix
    
    # INERTIA TRAINING 
    #hparams.inertia_buffer_size= args.update_freq * hparams.max_traj_size
    hparams.estimate_inertia = True
    hparams.inertia_batch_size= args.inertia_batch_size
    hparams.lr               = args.inertia_lr
    hparams.train_K          = args.epochs
    hparams.compute_variance = True
    hparams.optimizer_class  = args.optimizer   
    hparams.num_layers       = args.inertia_num_layers
    hparams.hidden_dim       = args.inertia_hidden_dim
    hparams.feature_extraction = args.inertia_feature_extraction
    hparams.dropout          = args.dropout
    hparams.reduction        = args.reduction
    hparams.loss             = args.loss
    hparams.hw               = args.hw
    hparams.bidirectional    = args.bidirectional
    
    hparams.mode             = args.mode
    
    hparams.max_reward = None 
    hparams.min_reward = None
    if args.reward == "attitude-stabilization": 
        pass
    elif args.reward == "attitude-detumbling":
        pass
    elif args.reward == "ee-tracking":
        pass
    else:
        raise NotImplementedError
    
    hparams.device           = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    hparams.start_hardness   = args.start_hardness
    hparams.hardness_rate    = args.hardness_rate
    hparams.update_payload   = 1
    hparams.continuous_trajectory = args.continuous_trajectory
    hparams.initialize       = args.init
    hparams.reset_rate       = args.reset_rate
    
    hparams.plot_freq        = 10
    
    hparams.rw               = rw
    hparams.thrusters        = thrusters
    hparams.rotation         = args.rotation
    
    hparams.plot_reward      = True
    hparams.is_train_inertia = True
    hparams.init_gamma       = args.gamma 
    hparams.gamma_rate       = args.gamma_rate
      
    # Algorithm hyperparameters
    LR            = args.control_lr
    N_STEPS       = hparams.max_traj_size * args.training_steps
    BATCH_SIZE    = args.control_batch_size
    N_EPOCHS      = 50
    GAMMA         = 0.99
    GAE_LAMBDA    = 0.99
    CLIP_RANGE    = 0.2
    ENT_COEF      = 0.01
    VF_COEF       = 0.5
    MAX_GRAD_NORM = 0.5
    WINDOW_SIZE   = 100
    DEVICE        = hparams.device
    
    filename = "./matlab/SC_3DoF.urdf"
    robot, _ = urdf2robot(filename=filename)
    n = robot.n_q
    hparams.n_joints = n

    #shape = 4 + 3*thrusters + 6 + n*2   
    shape = 3 + 3*thrusters + 6 + n*2   
    action_shape = n + 3*rw + 3*thrusters

    hparams.state_dim        = 4 + 3*thrusters + 6 + n*2 
    hparams.action_dim       = action_shape
    hparams.inertia_dim      = 7

    observation_space = Box(low = -np.inf, high = np.inf, shape = (shape + action_shape + 1,), dtype = np.float32)
    action_space = Box(low = -hparams.max_action, high = hparams.max_action, shape = (action_shape,))
    space_dim = {"observation-space":observation_space, "action-space":action_space}
    
    #policy = policy(observation_space, action_space, lr_schedule, lstm_hidden_size=args.hidden_size, \
    #    n_lstm_layers=args.num_layers, shared_lstm=args.shared_lstm)
    
    
    if args.init == "constant" : 
        #q0 = np.block([random_quaternion(goal = euler2quat(np.ones((3, 1)), pi/4), sigma = 0.0),
        #                             np.zeros(3)]).reshape((7, 1)).astype(np.float32)
        #q0 = np.block([euler2quat(np.ones((3, 1)), pi/4), np.zeros(3)]).reshape((7, 1)).astype(np.float32) \
        #    if args.reward == "attitude-stabilization" else np.array([0,0,0,1,0,0,0]).reshape((7, 1)).astype(np.float32)
        q0 = np.vstack((Euler_Angles321(np.ones((3, 1)), pi/4), np.zeros((3,1)))).reshape((-1,1)).astype(np.float32)\
            if args.reward == "attitude-stabilization" else np.zeros((6,1)).astype(np.float32)
        u0 = np.array([0.2, -0.15, 0.18, 0., 0., 0.]).reshape((6,1)).astype(np.float32)     
        qm = np.array([0, 5/4*pi, -5/4*pi]).reshape((n, 1)).astype(np.float32)
    elif args.init == "random":
        #q0 = np.array([0,0,0,1,0,0,0]).reshape((7, 1)).astype(np.float32)
        q0 = np.zeros((6,1)).astype(np.float32)
        u0 = np.block([[np.random.normal(size = (3,1), loc = 0.0, scale = 0.1)], [np.zeros((3, 1))]]).reshape((6, 1)).astype(np.float32) \
            if args.reward == "attitude-detumbling" else np.zeros((6,1)).astype(np.float32)
        qm = np.clip(np.absolute(np.random.nomal(size = (n,1), loc = 0.0, scale = 0.1)), 0 , 1) * 2 * pi 
    initial_state = {
                    'q0' : q0, # base quaternion
                    'qm': qm, # Joint variables [rad]
                    'u0': u0, # Base-spacecraft velocity
                    'um': np.zeros((n, 1)).astype(np.float32), # Joint velocities
                    'u0dot': np.zeros((6,1)).astype(np.float32), # Base-spacecraft acceleration
                    'umdot': np.zeros((n, 1)).astype(np.float32), # Joint acceleration
                    'tau': np.zeros((6 + n,1)).astype(np.float32) # manipulator joint torques
            }
    env = Spacecraft(hparams, space_dim=space_dim, initial_state=initial_state, robot=robot)
    qm = np.array([pi/2, 0.0, 0.0]).reshape((3, 1)).astype(np.float32) if args.reward == "ee-tracking" else np.zeros((n, 1)).astype(np.float32) 
    env.goal = {
        #'q0' : np.array([0,0,0,1,0,0,0]).reshape((7, 1)).astype(np.float32), # base quaternion
        'q0' : np.zeros((6,1)).astype(np.float32),
        'qm': qm, # Joint variables [rad]
        'u0': np.zeros((6,1)).astype(np.float32), # Base-spacecraft velocity
        'um': np.zeros((n, 1)).astype(np.float32), # Joint velocities
    }
    env.c = {"c_w": 1e2 if args.reward == "attitude-detumbling" else 1e-3, "c_oob": 1, "c_en": 1e-3, "c1":1, "c2":1, "c3":1, "c_b":0.5}
    env.threshold = {"attitude-stabilization":1e-3, "attitude-detumbling":1e-3, "ee-tracking":1e-3}
        
    if pretrain:
        print("Loading Environment data")
        env.load(args.ckpt + '/' + control_name_prefix)
    
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=hparams.save_model_freq,
        save_path=args.ckpt + '/',
        name_prefix=control_name_prefix,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    if pretrain : 
        print(args.ckpt + '/' + control_name_prefix)
        try:
            model = RecurrentPPO.load(args.ckpt + '/' + control_name_prefix)
            model.set_env(env)
        except Exception as e:
            print(e)
            print('Model file not existing')
            #exit(0)
            args.ckpt = None
            
    print(args.ckpt)
            
    if args.ckpt is None : 
        if args.mode in {"test", "render"} : 
            raise Exception("Testing mode has been selected, but no checkpoint is defined!") 
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, learning_rate=LR, n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=GAE_LAMBDA, 
                            clip_range=CLIP_RANGE, ent_coef=ENT_COEF, vf_coef=VF_COEF, max_grad_norm=MAX_GRAD_NORM, use_sde=False, stats_window_size=WINDOW_SIZE, seed=hparams.seed, 
                            device=DEVICE, policy_kwargs={"lstm_hidden_size":args.control_hidden_dim, "n_lstm_layers":args.control_num_layers, "shared_lstm":args.shared_lstm, 
                                                          "enable_critic_lstm":(not args.shared_lstm)})
                
    if args.mode == "train":
        print("---------------------------------------------------------")
        print("Learning Phase Started")
        args.ckpt = '/output' if args.ckpt is None else args.ckpt 
        try:
            model.learn(int(1e7), callback = checkpoint_callback, reset_num_timesteps = False)
            model.save(args.ckpt + '/' + control_name_prefix)
        except KeyboardInterrupt:
            model.save(args.ckpt + '/' + control_name_prefix)
        print('---------------------------------------------------------')
        print('Learning phase compleated')
    
    env.plot_reward = False
        
    if args.mode in {"train", "test"}:        
        print('----------------------------------------------------------')
        print('Control evaluation phase started')
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, warn=False, return_episode_rewards = True)
        print('Control evaluation phase compleated')

        print(mean_reward)
        
        print('----------------------------------------------------------')
        print('Inertia evaluation phase started')
        env.test_inertia(trainer=None, dir=dataset_dir, ckpt_path=hparams.embedding_checkpoint, num_workers=args.num_workers)
        print('Inertia evaluation phase compleated')
             

    obs, _ = env.reset()
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        action = action.squeeze() 
        if rw and thrusters:
            pass
        if rw and not thrusters:
            pass
        else:
            print(env.data["q0"][:4])
            
        obs, rewards, dones, is_terminated, info = env.step(action)
        episode_starts = dones or is_terminated
        env.render()
        if dones : break
        
    



    
    
    