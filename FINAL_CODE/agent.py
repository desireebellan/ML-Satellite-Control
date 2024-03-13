# Gymnasium environment expansion
# Parent Class Manipulator 
# Childrens classes Planar (2D) and Spacecraft (3D)

from utils import *
from classic_control import Control
from networks import InertiaEstimator
from hyperparameters import HyperParams
from typing import Optional, List, Union
from copy import deepcopy
from math import sin, cos, pi, exp, floor
from tqdm import tqdm

from SPART.robot_model import *
from SPART.spacecraftStep import *
from SPART.dynamics import *
from SPART.kinematics import *
from SPART.attitude_transformations import *

import numpy as np
import matplotlib.pyplot as plt

import gymnasium
import torch
import pickle
import sys
import os

def get_parameters(object, hparams):
    object.__dict__.update({name: getattr(hparams, name) for name in hparams.__dict__})
        

class Manipulator(gymnasium.Env):
    def __init__(self, hparams:HyperParams, space_dim:dict, control:bool = False,
                 initial_state: Optional[dict] = {}, robot: Optional[dict] = {}, payload_ranges: Optional[dict] = {}):
        
        self.observation_space = space_dim["observation-space"]
        self.action_space      = space_dim["action-space"]
        
        self.payload_ranges = {'m':(0.,10.), 'rcomx':(0.3, 0.1), 'rcomy': (0., 0.1), 'rcomz' : (0., 0.1)}
        self.payload_ranges.update(payload_ranges)
        
        self.initial_state = initial_state
        
        # set hyperparameters
        self.__dict__.update({name: getattr(hparams, name) for name in hparams.__dict__})

        # set payload
        self.ee = self.configure_payload(self.payload)
        
        # initialize plot arrays
        self.rewards = []
        self.losses  = []
        self.errors  = []
        self.vars    = []
        
        self.inertia_steps = 0
        self.epochs = {'reset'   : 0, 
                       'hardness': 0, 
                       'payload' : 0, 
                       'training': 0,
                       'control' : 0}
        
        # curriculum learning parameters
        self.hardness = self.start_hardness
        self.done     = True
        self.is_oob   = False
        
        # inertia estimator
        if self.embedding_checkpoint is not None:
            #checkpoints = torch.load(self.embedding_checkpoint)
            #self.inertia_estimator = InertiaEstimator(checkpoints["hyper_parameters"])\
            #    .load_from_checkpoint(checkpoints["state_dict"]).to(self.device)
            self.inertia_estimator = InertiaEstimator.load_from_checkpoint(self.embedding_checkpoint + '.ckpt', map_location=self.device).to(self.device)
            update_hparams(self.inertia_estimator, hparams)
            self.inertia_estimator.state_dim = 13
            self.inertia_estimator.setup()
        else : self.inertia_estimator = InertiaEstimator(hparams=hparams).to(self.device)
        self.best_loss         = sys.float_info.max
        self.patience          = 0
        self.end_train         = False
        
        # Training control routine 
        if control:
            self.qd, self.qddot, self.qdddot = sinusoids_trajectory(n_joints = hparams.n_joints, 
                                                                    T        = hparams.T, 
                                                                    q0       = self.q0, 
                                                                    harmonics= self.harmonics) 
            self.control = Control( qd = self.qd, 
                                    qddot     = self.qddot, 
                                    qdddot    = self.qdddot, 
                                    ID        = self.ID, 
                                    num_links = self.n_joints,
                                    rw        =self.rw,
                                    thrusters =self.thrusters)
        else : self.control = None
 
    def reset(self, seed:int = None, options: Optional[dict] = None):
        super().reset(seed=seed)
 
        if (self.continuous_trajectory and (self.done or self.is_oob)) or not self.continuous_trajectory:
            self.t = 1
            # Update epoch
            for key in self.epochs : 
                    if key != "hardness" : self.epochs[key] += 1
            # Reset the environment to its initial state
            if self.verbose:
                print('--------------------------------------------------')
                print('Reset environment')
            self.random_reset()  

            if self.update_payload_freq > 0 and self.epochs['payload'] >= self.update_payload_freq :
                print('Update Payload')
                self.ee = self.configure_payload(self.payload)
                self.epochs['payload'] = 0
                
            self.state        = self.get_joints(base=True).reshape(self.n_joints + 6, 1)
            self.actions      = np.zeros((self.n_joints + 3 * self.rw + 3*self.thrusters, 1))
            self.error_reward = []
                
            #self.episode_start = 1
            reward, _, _         = self.calculate_reward()
            self.data["reward"]  = reward
            self.episode_rewards = [reward]
            self.episode_errors  = []
            self.episode_vars    = []
            self.current_reward  = reward
            self.absolute_error  = None
            
            self.gamma = self.initial_gamma
            
            if hasattr(self, "prev_error"): del self.prev_error
            
            print('--------------------------------------------------')
        
        if self.control is not None and self.epochs["control"] >= self.reset_control_rate: 
            print("Reset control")
            self.control.qd, \
                self.control.qddot, \
                    self.control.qdddot = sinusoids_trajectory(n_joints =self.n_joints, 
                                                               T        =self.T, 
                                                               q0       = self.q0, 
                                                               harmonics= self.harmonics)
            self.epochs["control"] = 0

        if not hasattr(self.data, "reward"):
            reward, _, _         = self.calculate_reward()
        initial_state = self.get_input_parameters(self.data)
        initial_state = np.block([initial_state.flatten(), self.actions[:,-1].flatten(), reward])   
        return initial_state, {}
        
    def configure_payload(self, payload: Optional[dict] = None):
        if payload is not None : return payload
        
        # get random mass from a uniform distribution
        m = np.random.uniform(low = self.payload_ranges['m'][0], high = self.payload_ranges['m'][1]) 
        # get random center of mass (com) position wrt the end effector com
        # gaussian distribution over a circle (2D) with center aligned with its x axis            
        rcom_x = np.abs(np.random.normal(loc = self.payload_ranges['rcomx'][0], \
            scale = self.payload_ranges['rcomx'][1], size = 1))   
        rcom_y = np.random.normal(loc = self.payload_ranges['rcomy'][0], \
            scale = self.payload_ranges['rcomy'][1], size = 1)
        rcom_z = np.random.normal(loc = self.payload_ranges['rcomz'][0], \
                scale = self.payload_ranges['rcomz'][1], size = 1)
            
        rcom = np.array([rcom_x, rcom_y, rcom_z]).squeeze()
        # get random moment of inertia : 
        # (i) random principal moment of inertia 
        j = np.random.normal(loc = 0.0, scale = 1.0, size = 3)
        j = np.array(list(map(lambda x : x**2, j)))
        E = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        J = np.diag(np.matmul(E, j))
        # (ii) random rotation matrix
        Q, _ = np.linalg.qr(np.random.randn(3, 3))
        if self.rotation:
            J = np.transpose(Q) @ J @ Q
        # Final random inertia matrix rotated by Q
        # I = np.transpose(Q) @ J @ Q
         
        #return {'mp': m, 'rp': rcom, 'Ip': J, 'R': Q}  
        return {'mp': m, 'rp': rcom, 'Ip': J}  
        
    def step(self, action):
        # Define the dynamics of your custom environment based on the action taken
        try:
            data_next = self.__step__(tau = action)
        except AssertionError:
            if self.mode == "train" : 
                if self.plot_reward : self.plot()
                self.save(self.checkpoint_path)
            exit(0)
        next_state = self.get_input_parameters(data_next)
        
        self.data = data_next
        self.state = np.block([self.state, self.get_joints(base=True).reshape(self.n_joints + 6, 1)])
        self.actions = np.block([self.actions, action.reshape(self.n_joints + 3*self.rw + 3*self.thrusters, 1)])

        # Define the reward function based on the current state and 
        # Define if the episode is done (terminal state reached)
        reward, done, is_terminated = self.calculate_reward()
        self.data["reward"] = reward
        
        observation = next_state
        self.estimate, error, self.var = self.update_inertia(observation, action, is_terminal=is_terminated or done)
        if self.estimate_inertia:
            reward += self.gamma * self.var.mean().item() * 1e-2
            self.gamma *= self.gamma_rate

        self.update(reward, error, self.var, done, is_terminated)

        self.log(done, is_terminated)
        
        if done:
            print("GOAL REACHED !!!")
        elif not done and is_terminated:
            print("Trajectory concluded w/o reaching goal")

        if done or is_terminated:
            print("Episode length : " + str(self.t))
            
        if self.continuous_trajectory and self.t % self.max_traj_size == 0:
            print("Current trajectory size : " + str(self.t))
            
        self.done = done
        
        next_state = np.block([next_state.flatten(), action.flatten(), reward])

        return next_state, reward, done, is_terminated, {}        
   
    def update(self, reward, error, var, done, is_terminated):
        self.t += 1
        self.current_reward = reward

        if not(done or is_terminated):
            self.episode_rewards.append(reward)
            if self.estimate_inertia :
                if error is not None : self.episode_errors.append(error.cpu())
                if var is not None : self.episode_vars.append(var.cpu())
                
    def update_inertia(self, observation, action, is_terminal):
        target = self.get_payload() if self.is_train_inertia else None
        inertia_estimate, error, var = None, None, None
        
        if self.estimate_inertia: 
            if not self.is_oob:
                if self.is_train_inertia:
                    self.inertia_steps += 1
                    
                observation = np.hstack((Angles321_quat(observation[:3]).flatten(), observation[3:].flatten()))
            
                inertia_estimate, var = self.inertia_estimator.predict(observation = observation,  action=action, is_terminal=is_terminal, target=target)
                error = torch.abs(inertia_estimate - self.get_payload())/torch.abs(self.get_payload())
                self.absolute_error = torch.abs(inertia_estimate - self.get_payload())
                #if self.episode_start == 1 : self.episode_start = 0
            else : self.inertia_estimator.episode_start = 1

        if self.is_train_inertia and is_terminal and self.epochs["training"] % self.inertia_batch_size == 0:
            # Training Inertia Model
            print('----------------------------------')
            print("Updating Inertia Estimator ")
            loss = self.inertia_estimator.online_training_phase()
            
            assert not torch.isnan(loss), print("Loss is NaN!!!")
            
            #torch.save(self.inertia_estimator.state_dict(), self.checkpoint_path + '_model{}.pth'.format(self.epochs['training']))
            dir = self.embedding_checkpoint if self.embedding_checkpoint is not None else './output/'
            torch.save({'state_dict': self.inertia_estimator.state_dict(), 'hyper_parameters': self.inertia_estimator.hparams}
                       , dir + '_epoch={}.ckpt'.format(self.epochs["training"]))
            self.plot_inertia()
            
            self.losses.append(loss.item())
            
            print("loss : {:.4f}".format(loss.item()))
            
            if self.losses[-1] > self.best_loss:
                print("Loss function increased " + str(self.patience))
                if self.patience == self.max_patience:
                    self.end_train = True
                else : self.patience += 1
            else: 
                self.patience = 0   
                self.best_loss = self.losses[-1]
                
        return inertia_estimate, error, var

    def log(self, done:bool, is_terminated:bool):
        if (done or is_terminated) and self.verbose: 
   
            # epoch 
            if self.mode == 'train' : print('Epoch ', self.epochs['training'])
            
            # average reward
            average_reward = sum(self.episode_rewards)/len(self.episode_rewards)
            self.rewards.append(average_reward)
            print("Average Episode Reward :", average_reward)
            
            if self.estimate_inertia:
                
                # payload target
                print("Target Payload : ", self.get_payload().flatten().tolist())
                           
                # average relative error
                average_error = self.episode_errors[-1]
                self.errors.append(average_error)
                print("Last Relative Error :", average_error.tolist(), " Mean :", average_error.mean().item())
                
                # last absolute error
                if self.absolute_error is not None:
                    print("Last Absolute Error :", self.absolute_error.tolist(), " Mean :", self.absolute_error.mean().item())
            
                # average precision
                average_precision = self.episode_vars[-1]
                self.vars.append(average_precision)
                print("Average Episode precision :", average_precision.mean().item())
            
            if self.is_train_inertia:
                if len(self.losses) > 0 : 
                    print("Estimator Loss :", self.losses[-1], "Best Loss :", self.best_loss, "N° epochs without decreasing loss :", self.patience) 
            
            print(self.epochs["training"])
            if self.epochs["training"] % self.plot_freq == 0 and (done or (self.is_oob if self.continuous_trajectory else is_terminated)):
                    self.save(self.checkpoint_path) 
                    if self.plot_reward: 
                        self.plot()
                        self.plot_reward_error()
                        self.plot_state()

    def plot(self):
        rewards = np.array(self.rewards).astype(np.float32)
        min_reward, max_reward = np.min(self.rewards), np.max(self.rewards)
        min_reward = min_reward if self.min_reward is None else max(self.min_reward, min_reward)
        max_reward = max_reward if self.max_reward is None else min(self.max_reward, max_reward)
        rewards = (rewards - min_reward)/(max_reward - min_reward) * 100
        rewards[rewards < 0] = 0.0
        rewards[rewards > 100] =  100.0
        
        fig, axs = plt.subplots(1,1)
        
        x = np.arange(0,len(self.rewards),1)
        #x = np.linspace(0, len(rewards), len(rewards))
        
        # real reward
        axs.scatter(x,rewards, color = 'b', label = "episode reward", alpha = 0.3)
        
        # average reward
        window = 100
        average = []
        for ind in range(len(rewards)):
            start = max(0, ind-window)
            average.append(np.mean(rewards[start:ind+1]))
        axs.plot(x, average, 'r', label = "mean reward")
        axs.set(xlabel='episodes', ylabel='reward', title='reward')
        axs.legend(loc='lower left')
        axs.grid()
        if self.min_reward is not None and self.max_reward is not None:
            axs.set_ylim(bottom = -10, top = 110)

        fig.suptitle('PPO-RECURRENT : {}'.format(self.reward))
        fig.tight_layout()
        
        fig.savefig(self.checkpoint_path + "_reward.png")
        plt.close(fig)
        
    def plot_inertia(self):
            
        fig, axs = plt.subplots(1, 3)  
        axs[0].plot(self.losses)
        axs[0].set_title("Loss")
        try:
            errors = torch.stack(self.errors, dim = 1)
        except:
            errors = []
        try:
            vars = torch.stack(self.vars).flatten()
        except:
            vars = []
         
        for error in errors:
            axs[1].plot(error*100)
        axs[1].legend(self.legend)
        axs[1].set_ylim(bottom = 0, top = 50)
        axs[1].set_title("Error")
        
        axs[2].plot(vars)
        axs[2].set_title("precision")
        
        fig.suptitle('PPO-RECURRENT INERTIA ESTIMATOR TRAINING')
        fig.tight_layout()
        dir = self.embedding_checkpoint if self.embedding_checkpoint is not None else './output'
        fig.savefig(dir + '_losserrorplot_steps=' + str(self.epochs['training']) + '.png')
        plt.close(fig)  
        
    def plot_state(self):
        fig, axs = plt.subplots(1, self.n_joints)
        for i in range(self.n_joints):
            axs[i].plot(self.state[i])
            axs[i].set_title("q" + str(i))
        fig.suptitle('PLOT')
        fig.tight_layout()
        dir = self.embedding_checkpoint if self.embedding_checkpoint is not None else './output'
        fig.savefig(dir + '_state_steps=' + str(self.epochs['training']) + '.png')
        plt.close(fig)
        
    def plot_reward_error(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.error_reward)
        fig.suptitle('PLOT ERROR')
        fig.tight_layout()
        dir = self.embedding_checkpoint if self.embedding_checkpoint is not None else './output'
        fig.savefig(dir + '_reward-error_steps='+ str(self.epochs['training']) + '.png')
        plt.close(fig)
        
    def plot_action(self):
        fig, axs = plt.subplots(1, self.n_joints + self.rw * 3)
        for i in range(self.n_joints):
            axs[i].plot(self.actions[i])
            axs[i].set_title("q" + str(i))
        fig.suptitle('PLOT ACTIONS')
        fig.tight_layout()
        fig.savefig("./output/images/plot_actions_" + str(self.epochs['training']) + '.png')
        plt.close(fig)        
        
    def save(self, checkpoint_path):
        with open(checkpoint_path + '.pickle', 'wb') as f:
            d = {'rewards':self.rewards, 'ee':self.ee, 'epochs': self.epochs, 'hardness':self.hardness}
            if self.estimate_inertia : d.update({'errors':self.errors, 'vars':self.vars})
            if self.is_train_inertia : d.update({'losses':self.losses, 'best_loss':self.best_loss, 'patience':self.patience})
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    def load(self, checkpoint_path):
        #if self.is_train_inertia : 
        #    self.inertia_estimator.load_state_dict(torch.load(self.checkpoint_path + '_model.pth', map_location=lambda storage, loc: storage))
        try:
            with open(checkpoint_path + '.pickle', 'rb') as f:
                while True:             
                    try:
                        data = pickle.load(f)
                        for key, val in data.items():
                            if isinstance(val, torch.Tensor): val = val.to(self.device)
                            setattr(self, key, val)
                    except EOFError:
                        break    
        except:
            print("File " + checkpoint_path + '.pickle not existing!')
            
    def create_dataset(self, samples:int, filename:str)->None:
        from os.path import exists
        import csv
        
        steps = self.steps 
        delay = int(self.max_traj_size / steps)
        
        # pickle trjectories file
        data_path =self.checkpoint_path + filename + '.pickle'
        start_samples = 0
        if exists(data_path):
            header, data = read_pickle(data_path)
            start_samples = len(data)
            print('starting sample ',start_samples)
            f = open(data_path, 'wb')
            pickle.dump(header, f, protocol=pickle.HIGHEST_PROTOCOL)   
            write_pickle(data, f, steps)
        else : 
            f = open(data_path, 'wb')
            header = {'output-dim': self.inertia_dim, 'input-dim': self.state_dim, 'steps': steps}
            pickle.dump(header, f, protocol=pickle.HIGHEST_PROTOCOL)  
            
        print("Creating samples ...")
        # for number_of_robots
        for sample in tqdm(range(start_samples, samples)):             
            trajectory = []
            self.data = deepcopy(self.initial_state)
            self.data.update({'q0': np.block([random_quaternion(goal = euler2quat(np.ones((3, 1)), pi/4), sigma = 2),np.zeros(3)]).reshape((7, 1)).astype(np.float32)})
            self.ee = self.configure_payload()
            self.control.qd, \
                            self.control.qddot, \
                                self.control.qdddot = sinusoids_trajectory(n_joints =self.n_joints, 
                                                                        T        =self.T, 
                                                                        q0       = self.q0, 
                                                                        harmonics= self.harmonics)
            # for number of steps
            for t in range(steps):
                # save state
                trajectory.append(self.get_input_parameters(self.data).tolist())
                # generate action 
                for s in range(delay):
                    tau0, taum = self.control.next(self.get_joints(base = True), self.get_joints_velocity(base = True), (t*delay + s)*self.dt)
                    tau0 = np.clip(tau0, -20*pi*self.dt, 20*pi*self.dt)
                    taum = np.clip(taum, -4*pi*self.dt, 4*pi*self.dt)
                    if self.rw and self.thrusters : tau = np.block([[tau0], [taum]]).reshape((6+self.n_joints, 1))
                    elif self.rw and not self.thrusters : tau = np.block([[tau0[:3,:]], [taum]]).reshape((3+self.n_joints, 1))
                    else : tau = taum
                    # next state
                    #_, _, done, is_terminated, _ = self.step(tau.flatten())
                    self.data = self.__step__(tau = tau)
       
            data = torch.FloatTensor(trajectory).squeeze()
            data = {'input' : data, 'output': self.get_payload()}
            pickle.dump({sample:data}, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        
    def train_inertia(self, dir, max_epochs, batch_size:int, metric_path:str=None, patience:int=20, num_workers:int=1):
        train_path = dir + '/train.pickle'
        dev_path = dir + '/val.pickle'
        test_path = dir +'/test.pickle'
        data_module = OnlineDataLoader(pkl_train=train_path, pkl_dev=dev_path, pkl_test=test_path, batch_size=batch_size, num_workers=num_workers)
    
        checkpoint_dir = './output'
        check_point_callback = pl.callbacks.ModelCheckpoint(
            monitor='valid_error',  
            verbose=True, 
            save_top_k = 2,  
            mode='min', 
            dirpath= checkpoint_dir,  
            filename='{epoch}-{valid_error:.4f}')

        early_stopping = pl.callbacks.EarlyStopping(
            monitor='valid_loss',  
            patience = patience,  
            verbose=True,  
            mode='min',
        )
       
        self.inertia_estimator.train()   
    
        metric = Metrics(name = 'spacecraft', device = 'cpu', parameters=self.legend)
        if metric_path is not None : metric.load(metric_path)
        
        accelerator = 'cpu' if self.device == 'cpu' else 'gpu'
        
        trainer = pl.Trainer(accelerator = accelerator, val_check_interval=1.0, max_epochs = max_epochs, default_root_dir= checkpoint_dir,  
                             callbacks = [early_stopping, check_point_callback, metric])
        trainer.fit(self.inertia_estimator, ckpt_path = metric_path, datamodule = data_module)
        
        return trainer
    
    def test_inertia(self, trainer, dir:str, ckpt_path = "best", num_workers:int=1)->None:
        train_path = dir + '/train.pickle'
        dev_path = dir + '/val.pickle'
        test_path = dir +'/test.pickle'
        data_module = OnlineDataLoader(pkl_train=train_path, pkl_dev=dev_path, pkl_test=test_path, num_workers=num_workers)
        
        accelerator = 'cpu' if self.device == 'cpu' else 'gpu'
        if trainer is None : 
            metric = Metrics(name = 'spacecraft', device = 'cpu', parameters=self.legend)
            metric.load(ckpt_path)
            trainer = pl.Trainer(accelerator = accelerator, callbacks=metric)
        trainer.test(model = self.inertia_estimator, ckpt_path = ckpt_path, datamodule = data_module)

    def render(self, mode=None):
        # Define how to render the environment (optional)
        pass

    def close(self):
        # Define any cleanup actions (optional)
        pass

    def calculate_reward(self):
        # Define the reward function based on the current state
        if self.out_of_bounds():
            print("State out of Bounds!!!")
            return 0, False, True
        
        # velocity penalty
        velocity_penalty = 1e-2 * self.get_velocity_magnitude()
        
        # out of bounds penalty
        q = self.get_joints()
        index = np.where(self.bounds != None)
        assert not any(np.isnan(q))
        
        bounds_distance_max = np.minimum(np.abs(q[index] - self.bounds[index]), 2*pi - np.abs(q[index] - self.bounds[index]))
        bound_distance_min = np.minimum(np.abs(q[index] + self.bounds[index]), 2*pi - np.abs(q[index] + self.bounds[index]))
        bound_distance = np.linalg.norm(np.minimum(bound_distance_min, bounds_distance_max))
        oob_penalty = 1/(1e-7 + bound_distance) * 1e-3
        
        # energy consumptio penalty
        energy_penalty = 1e-3 * self.get_acceleration()
        
        reward = -velocity_penalty * self.c["c_w"] - oob_penalty * self.c["c_oob"] - energy_penalty * self.c["c_en"]
        done = True
        
        is_terminated = False if self.continuous_trajectory else self.t == self.max_traj_size
        
        return reward, done, is_terminated
    
    def get_robot(self, robot = None):
        raise NotImplementedError   
    
    def get_payload(self):
        raise NotImplementedError
    
    def __step__(self, tau: Union[float,  List[float]] = 0., param:str = None):
        raise NotImplementedError
    
    def random_reset(self):
        # Randomly reset the initial state
        self.data = deepcopy(self.initial_state)
    
    def get_dynamic_parameters(self, params):
        pass
    
    def out_of_bounds(self):
        self.is_oob = False
        return False
    
    def get_input_parameters(self):
        raise NotImplementedError
        
    
class Planar2R(Manipulator):
    
    def __init__(self, hparams:HyperParams, space_dim:dict, control:Control = None,
                 initial_state: Optional[dict] = {}, robot: Optional[dict] = {}, payload_ranges: Optional[dict] = {}):
        
        self.q0 = initial_state["q"]      
        
        super().__init__(hparams, space_dim, control, initial_state, robot, payload_ranges)       
        
        self.robot = self.get_robot(robot)
        
    
    def get_robot(self, robot = {}):
        # masses
        if not 'm' in robot:
            robot['m'] = 1 # kg
        robot['m'] = np.array(robot['m']).astype(np.float32) if isinstance(robot['m'], list) else np.array([robot['m']]*self.n).astype(np.float32)    
        # lenghts
        if not 'l' in robot:
            robot['l'] = 0.5 # m
        robot['l'] = np.array(robot['l']) if isinstance(robot['l'], list) else np.array([robot['l']]*self.n)     
        # COMs
        if not 'rcom' in robot:
            robot['rcom'] = np.array([[-robot['l'][i]/2, 0] for i in range(self.n)])
        elif isinstance(robot['rcom'], list) or isinstance(robot['rcom'], tuple):
            robot['rcom'] = np.array(robot['rcom']) if isinstance(robot['rcom'][0], list) or isinstance(robot['rcom'][0], tuple) \
                else np.array([robot['rcom'][i] for i in range(self.n)])
        elif isinstance(robot['rcom'], float) : robot['rcom'] = np.array([[robot['rcom'], 0]]*self.n)
        
        # inertia matrices
        if not 'I' in robot:
            robot['I'] = [1/12*robot['m'][i]*robot['l'][i]**2 for i in range(self.n)]
        if isinstance(robot['I'], list) and isinstance(robot['I'][0], list) : 
            I_ = []
            for i in robot['I']:
                It = np.zeros((3, 3))
                It[np.triu_indices(3, 0)] = i
                It = np.triu(It) + np.tril(It.T, -1)
                I_.append(It)
            robot['I'] = np.array(I_)
        elif isinstance(robot['I'], list) and isinstance(robot['I'][0], float):
            #It = np.zeros((3, 3))
            #It[np.triu_indices(3, 0)] = robot['I']
            #It = np.triu(It) + np.tril(It.T, -1)
            #robot['I']= np.array([It * self.n])
            robot['I'] = np.array([np.array([[0, 0, 0], [0, 0, 0], [0, 0, Ii]]) for Ii in robot['I']])
        elif isinstance(robot['I'], float):
            I_ = np.zeros((3, 3))
            I_[2, 2] = robot['I']
            robot['I'] = np.array([robot['I'] * self.n])
            
        self.legend = ['mass', 'rcom_x', 'rcom_y', 'I_zz']
        self.bounds = np.array([None, pi])
        
        return robot
    
    def __step__(self, tau: Union[float,  List[float]] = 0., dt:float = 0.1, param:str = None):
            '''
            perform one step using the action tau and the parameters param
            if param is None uses the robot's parameters  
            '''     
            if param is None : param = self.ee
            data = {}
            
            # tau vector
            tau = tau.reshape((self.n_joints,1))
            data['tau'] = tau
            
            H = self.get_inertia_matrix(self.data["q"], param)
            C = self.get_coriolis_vector(self.data["q"], self.data["qdot"], param)
            data["qdotdot"] = np.linalg.inv(H) @ (tau - C)
            
            data["qdot"] = np.clip(self.data["qdot"] + data["qdotdot"]*dt, -pi/2*dt, pi/2*dt)
            data["q"] = self.data["q"] + data["qdot"] * dt
            
            # normalize angle
            data["q"] = np.mod(data["q"], 2*pi)
            
            return data
    
    def render(self, mode=None):
        # Define how to render the environment (optional)
        # TO DO : USE PYGAME
        
        left, width = .1, .9
        bottom, height = .1, .9
        right = left + width
        top = bottom + height
        if not hasattr(self, 'figure'):
            self.figure = plt.figure(1)
            self.ax = plt.subplot(111)
            w = self.robot["l"].sum()
            self.ax.set_xlim(xmin=-w, xmax = w)
            self.ax.set_ylim(ymin=-w, ymax=w)
        if hasattr(self, 'q1'): self.q1.remove()
        if hasattr(self, 'q2') : self.q2.remove()
        if hasattr(self, 'text_reward') : self.text_reward.remove()
        #if hasattr(self, )
        o = (0.0, 0.0)
        p1 = (self.robot['l'][0]*cos(self.data['q'][0]), self.robot['l'][0]*sin(self.data['q'][0]))
        p2 = (self.robot['l'][0]*cos(self.data['q'][0]) + self.robot['l'][1]*cos(self.data['q'][0]+self.data['q'][1]), \
            self.robot['l'][0]*sin(self.data['q'][0]) + self.robot['l'][1]*sin(self.data['q'][0]+self.data['q'][1]))
        self.q1, = self.ax.plot([o[0], p1[0]],[o[1], p1[1]], 'k')
        self.q2, = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k')
        self.text_reward = self.ax.text(right, top, '{:.2f}'.format(self.current_reward.item()), horizontalalignment='right', verticalalignment='top', transform=self.ax.transAxes)
        plt.figure(1)
        plt.pause(self.dt)

    def close(self):
        # Define any cleanup actions (optional)
        pass

    def calculate_reward(self):
        
        reward, done, is_terminated = super().calculate_reward()
        if self.is_oob : return 1e-3, False, True
        
        q = self.data["q"]
        qdot = self.data["qdot"]

        if self.reward == "joint-error":
            next_state = q
            joints = np.mod(next_state, 2*pi)
            index = np.where(joints > 2*pi - joints)
            joints[index] = 2*pi - joints[index]
                
            distance = np.linalg.norm(joints - self.goal[:,0])
            done = done and (distance < self.threshold).all()
            reward += -distance + 100*done
            
        elif self.reward == "joint-direction":
            error = np.minimum(np.abs(q.flatten() - self.goal[:,0].flatten()), 2*pi - np.abs(q.flatten()-self.goal[:,0].flatten())).sum()
            error_dot = np.abs(qdot.flatten() - self.goal[:,1].flatten()).sum()
            done = done and error < self.threshold and error_dot < 0.1
            if hasattr(self, "prev_error"):
                reward += -(error >= self.prev_error).astype(np.float32)
            else : 
                reward += 0
            self.prev_error = error  
            
        elif self.reward == "ee-direction":
            error = np.linalg.norm(self.ee_pos(q).flatten() - self.ee_pos(self.goal[:,0]).flatten())
            
            done = done and error < self.threshold
            if hasattr(self, "prev_error"):
                reward += -(error >= self.prev_error).astype(np.float32)
            else : 
                reward += 0
            reward += done.astype(np.float32)
            self.prev_error = error    
        
        reward = np.nanmax([reward.astype(np.float32), -1e4])
                
        return reward, done, is_terminated
    
    def get_input_parameters(self, data: dict, mode:str = None):
        '''
            returns the input parameters to the inertial identifiers
            state = {q1, q2, q1_dot, q2_dot, }
        '''
        q = data["q"].flatten()
        if self.reward in {"joint-direction", "joint-error"}:
            q = angle_error(q, self.goal[:,0].flatten())
        #return np.block([q, data["qdot"].flatten(), data["qdotdot"].flatten()])
        # sine and cosine
        return np.block([np.cos(q), np.sin(q), data["qdot"].flatten()])
    
    def random_reset(self):
        super().random_reset()
        if self.epochs["reset"] == self.reset_rate:
            self.epoch["reset"] = 0
            self.epochs["hardness"] += 1
            if self.initialize == "random":
                print("Update initial state")
                print("Hardness ", min(2.0, self.hardness))
                    
                sigma = min(2.0, self.hardness)
                self.data.update({'q': np.random.normal(loc=self.goal[:,0], scale=sigma).reshape(self.n_joints, 1)})
                    
                if self.epochs['hardness'] == self.hardness_rate:
                    self.epochs['hardness'] = 0
                    self.hardness += 0.1
            elif self.initialize == "constant":
                self.data.update({'q': np.zeros(self.n_joints, 1)})
        print("q0 :", self.data["q"].flatten().tolist())
            

    def ee_pos(self, state):
        return self.robot["l"][0] * np.array([[np.cos(state[0])], [np.sin(state[0])]]) + \
            self.robot["l"][1] * np.array([[np.cos(state[0] + state[1])], [np.sin(state[0] + state[1])]])

    def get_payload(self):
        return torch.tensor([self.ee['mp']] +  self.ee['rp'].flatten().tolist()[:2] + [self.ee['Ip'][2,2]]).to(self.device)
    
    def get_velocity_magnitude(self):
        return np.linalg.norm(self.data["qdot"]) 
    
    def get_joints(self, base = False):
        return self.data["q"]
    
    def get_joints_velocity(self, base = False):
        return self.data["qdot"]
    
    def get_acceleration(self):
        return np.linalg.norm(self.data["qdotdot"])
    
    def out_of_bounds(self):
        self.is_oob = False
        return False
        
    def get_dynamic_parameters(robot = None, params = None):
        l = robot["l"]
        m = robot["m"]
        rcom = robot["rcom"]
        I = robot["I"]
        params = params if params is not None else {"mp":0., "rp":np.zeros(2).astype(np.float32), "Ip":np.zeros((3,3)).astype(np.float32)}
        # I1zz = (Ic1,zz + m1((l1 + rc1,x)**2 + rc1,y**2))
        I1zz = (I[0,2,2] + m[0]*((l[0] + rcom[0,0])**2 + rcom[0,1]**2))
        # a1 = I1zz + (m2+ml)l1**2
        a1 = I1zz + (m[1] +  params["mp"]) * l[0]**2
        # I2,zz = Ic2,zz + m2((l2 + rc2,x)**2 + rc2,y**2)
        I2zz = I[1,2,2] + m[1]*((l[1] + rcom[1,0])**2 + rcom[1,1]**2)
        # Il,zz = Icl,zz + ml((l2 + rcl,x)**2 + rcl,y**2)
        Ilzz =  params["Ip"][2,2] +  params["mp"] * ((l[1] +  params["rp"][0])**2 +  params["rp"][1]**2)
        # a2 = (I2,zz + Il,zz)
        a2 = I2zz + Ilzz
        # a3 = l1*(m2(l2 + rc2,x) + ml(l2 + rcl,x))
        a3 = l[0] * (m[1]*(l[1] + rcom[1,0]) +  params["mp"]*(l[1] +  params["rp"][0]))
        # a4 = l1 * (m2*rc2,y + ml*rcl,y)
        a4 = l[0]*(m[1]*rcom[1,1] +  params["mp"] *  params["rp"][1])
        
        return np.array([a1, a2, a3, a4])
        
    def get_inertia_matrix(self, q, robot = None, param = None):
        if param is None : param = {"mp": 0.0 , "rp": np.zeros(3).astype(np.float32), "Ip": np.zeros((3,3)).astype(np.float32)}
        a = self.get_dynamic_parameters(param=param, robot=robot)
        # m11 = a1 + a2 + 2a3c2 - 2a4s2
        m11 = a[0] + a[1] + 2*a[2]*cos(q[1]) - 2*a[3]*sin(q[1])
        # m12 = m21 = a2 + a3c2 - a4s2
        m12 = a[1] + a[2]*cos(q[1]) - a[3]*sin(q[1])
        # m22 = a2
        m22 = a[1]
        return np.array([[m11, m12],[m12, m22]])
    
    def get_coriolis_vector(self, q, qdot, robot=None, param = None):
        if param is None : param = {"mp": 0.0 , "rp": np.zeros(3).astype(np.float32), "Ip": np.zeros((3,3)).astype(np.float32)}
        a = self.get_dynamic_parameters(param=param, robot=robot)
        # c1 = -(a3s2 + a4c2)(2qdot1*qdot2 + qdot2**2)
        c1 = -(a[2]*sin(q[1]) + a[3]*cos(q[1]))*(2*qdot[0]*qdot[1] + qdot[1]**2)
        # c2 = (a3s2 + a4c2)qdot1**2
        c2 = (a[2]*sin(q[1]) + a[3]*cos(q[1]))*qdot[0]**2
        return np.array([c1, c2]).reshape(2,1) 
    
    def ID(self, q, qdot, qddot, params=None):
        M = self.get_inertia_matrix(q)
        C = self.get_coriolis_vector(q, qdot)
        tau = M@qddot + C @ qdot
        return tau
    
class Spacecraft(Manipulator):

    def __init__(self, hparams:HyperParams, space_dim:dict, control:Control = None,
                 initial_state: Optional[dict] = {}, robot: Optional[dict] = {}, payload_ranges: Optional[dict] = {}):

        self.bounds = np.array([None, pi, pi])
        if hparams.rotation: self.legend = ['mass', 'rcom_x', 'rcom_y', 'rcom_z', 'I_xx', 'I_xy', 'I_xz', 'I_yy', 'I_yz', 'I_zz']
        else : self.legend = ['mass', 'rcom_x', 'rcom_y', 'rcom_z', 'I_xx', 'I_yy', 'I_zz']
        self.q0 = initial_state["qm"]
        self.base_robot = robot

        super().__init__(hparams, space_dim, control, initial_state, robot, payload_ranges)
        
    def configure_payload(self, payload: Optional[dict] = None):
        param = super().configure_payload(payload)
        self.robot = setParams(param, self.base_robot)
        return param
    
    def get_input_parameters(self, data:dict) -> np.ndarray:
        #q0 = data["q0"] if self.reward == "ee-tracking" else data["q0"][:4]
        q0 = data["q0"]if self.reward == "ee-tracking" else data["q0"][:3]
        return np.block([q0.flatten(), data["qm"].flatten(), data["u0"].flatten(), data["um"].flatten()])
    
    def get_payload(self):
        p = torch.tensor([self.ee['mp']] +  self.ee['rp'].flatten().tolist())
        if self.rotation: 
            I = torch.tensor(self.ee['Ip'].tolist()).triu().flatten()
            I = I[I.nonzero()]
        else : 
            I = torch.tensor(np.diag(self.ee['Ip']).flatten().tolist())
        p = torch.cat([p, I.squeeze()]).to(self.device)
        return p
    
    def render(self, mode = None):
        # Define how to render the environment
                
        if not hasattr(self, 'figure'):
            self.figure = plt.figure(1)
            self.axs = self.figure.add_subplot(projection = '3d')
            self.axs.set_xlim(xmin = -2, xmax = 2)
            self.axs.set_ylim(ymin = -2, ymax = 2)
            self.axs.set_zlim(zmin = -2, zmax = 2)
        if hasattr(self, 'lines'):
            self.lines.remove()
        if hasattr(self, 'base_dot'):
            self.base_dot.remove()
        if hasattr(self, 'ee_dot'):
            self.ee_dot.remove()
        
        # base position
        #p1 = self.data["q0"][3:].reshape((3, 1))
        p1 = np.zeros((3,1))
        # joint positions
        #R0 = transpose(quat_DCM(transpose(self.data["q0"][:4,:]).squeeze()))
        #r0 = self.data["q0"][4:, :]
        R0 = transpose(Angles321_DCM(self.data["q0"][:3,:]))
        r0 = np.zeros((3,1))
        qm = self.data["qm"]
        #Kinematics
        _, _, rj, _, _, _ = Kinematics(R0, r0, qm, self.robot)
        rj = np.block([p1, rj]).T
        
        self.lines, = self.axs.plot(rj[:,0], rj[:,1], rj[:,2], 'k')
        self.base_dot = self.axs.scatter(p1[0], p1[1], p1[2], marker = 'o', s = 20, c = 'k')
        self.ee_dot = self.axs.scatter(rj[-1,0], rj[-1,1], rj[-1,2], marker = 'o', s = 20, c = 'r')
        
        plt.figure(1)
        plt.pause(self.dt)
    
    
    def __step__(self, tau: Union[float,  List[float]] = 0., param:dict = None):
        '''
            perform one step using the action tau and the parameters pi
            if pi is None uses the robot's parameters
        '''
        robot = self.robot if param is None else setParams(params=param, robot=self.base_robot)

        if self.rw and not self.thrusters:
            assert tau.shape[0] == self.n_joints + 3, print(tau.shape)
            tau0 = np.block([[tau[:3].reshape((3,1))],[np.zeros((3,1))]]).reshape((6,1)).astype(np.float32)
            taum = tau[3:].reshape((self.n_joints,1)).astype(np.float32)
        elif self.rw and self.thrusters:
            assert tau.shape[0] == self.n_joints + 6
            tau0 = tau[:6].reshape((6,1)).astype(np.float32)
            taum = tau[6:].reshape((self.n_joints,1)).astype(np.float32)
        else:
            assert tau.shape[0] == self.n_joints
            taum = tau.reshape((self.n_joints,1)).astype(np.float32) 
            tau0 = np.zeros((6,1)).astype(np.float32)

        wf0  = np.zeros((6,1))
        wfm = np.zeros((6, self.robot.n_links_joints))
        
        data = spacecraftStep(tau0=tau0, taum=taum, wf0=wf0, wfm=wfm, data=self.data, robot = robot, dt=self.dt, n_steps=self.n_steps)

        return data
    
    def calculate_reward(self):
        reward, done, is_terminated = super().calculate_reward()   
        
        # terminate the sequence if the robot exceed bounds       
        if self.is_oob : return 1e-3, False, True
        
        if self.reward == "attitude-stabilization":
            #error = scalar_quat_error(self.data["q0"][:4], self.goal["q0"][:4]) # error
            #phi, theta, psi = quat_Angles321(self.data["q0"][:4]).flatten()
            phi, theta, psi = self.data["q0"][:3].flatten()
            error = phi**2 + theta**2 + psi**2
            reward += - error * self.c["c1"]
            if hasattr(self, "prev_error"):
                reward += -(error >= self.prev_error).astype(np.float32) * self.c["c2"]
            self.prev_error = error
            done = error < self.threshold["attitude-stabilization"] # and np.linalg.norm(self.data["u0"][:3].flatten()) < 0.1
        elif self.reward == "attitude-detumbling":
            done = self.get_velocity_magnitude() < self.threshold["attitude-detumbling"] 
        elif self.reward == "ee-tracking":
            ee_pos, ee_vel = self.get_ee(self.data)
            ee_pos_goal, ee_vel_goal = self.get_ee(self.goal)
            ee_error = np.linalg.norm(ee_pos.flatten() - ee_pos_goal.flatten())
            ee_vel_error = np.linalg.norm(ee_vel - ee_vel_goal) 
            #attitude_error = scalar_quat_error(self.data["q0"][:4], self.goal["q0"][:4])
            #origin_error = np.linalg.norm(self.data["q0"][4:] - self.goal["q0"][4:])
            attitude_error = np.linalg.norm(self.data["q0"][:3] - self.goal["q0"][:3])
            origin_error = np.linalg.norm(self.data["q0"][3:] - self.goal["q0"][3:])
            reward += - ee_error * self.c["c3"] - (attitude_error + origin_error) * self.c["c_b"]
            done = ee_error < self.threshold["ee-tracking"] and ee_vel_error < self.threshold["ee-tracking"]
        else:
            return 1e-3, False, is_terminated
        
        if done : reward += 1000
        
        reward = np.nanmax([reward.astype(np.float32), -5000])

        return reward, done, is_terminated

    def random_reset(self):
        super().random_reset()
        if self.epochs["reset"] == self.reset_rate:
            self.epochs["reset"] = 0
            self.epochs["hardness"] += 1
            if self.initialize == "random":
                print("Update initial state")
                print("Hardness ", self.hardness)   
                
                sigma = min(5.0, self.hardness)
                if self.reward in {"attitude-stabilization", "attitude-detumbling"}:
                    self.data.update({'q0': np.block([quat_Angles321(random_quaternion(goal = self.goal["q0"][:4], sigma = sigma)),np.zeros(3)]).reshape((7, 1)).astype(np.float32)})
                    #self.data.update({'q0': np.block([random_quaternion(goal = euler2quat(np.ones((3, 1)), pi/4), sigma = sigma),np.zeros(3)]).reshape((7, 1)).astype(np.float32)})
                    self.initial_state.update({'q0' : deepcopy(self.data["q0"])})
                if self.reward == "attitude-detumbling":
                    self.data.update({'u0': np.block([np.random.uniform(size = 3, mean = 0.0, sigma = sigma), np.zeros(3)]).reshape((6,1)).astype(np.float32)})
                    self.initial_state.update({'u0' : deepcopy(self.data["u0"])})
                    
                if self.reward == "ee-tracking":
                    self.data.update({'qm' : np.clip(np.absolute(np.random.nomal(size = (n,1), loc = 0.0, scale = sigma)), 0 , 1) * 2 * pi })
                    self.initial_state.update({'qm' : deepcopy(self.data["qm"])})
                
                if self.epochs['hardness'] >= self.hardness_rate:
                    self.epochs['hardness'] = 0
                    self.hardness += 0.1
                    self.hardness = min(self.hardness, 5.0)
            elif self.initialize == "constant":
                if self.reward == "attitude-stabilization":
                    #self.data.update({'q0': np.block([random_quat(np.ones((3, 1)), pi/4), np.zeros(3)]).reshape((-1, 1)).astype(np.float32)})
                    self.data.update({'q0': np.vstack((Euler_Angles321(np.ones((3, 1)), pi/4), np.zeros((3,1)))).reshape((-1, 1)).astype(np.float32)})
                if self.reward == "attitude-detumbling":
                    self.data.update({'u0': np.array([0.2, -0.15, 0.18, 0., 0., 0.]).reshape((6,1)).astype(np.float32)})
                if self.reward == "ee-tracking":
                    self.data.update({'qm': np.array([0, 5/4*pi, -5/4*pi]).reshape((self.n_joints, 1)).astype(np.float32)})
        #print("qb :", self.data["q0"].flatten()[:4].tolist())
        print("qb :", self.data["q0"].flatten()[:3].tolist())
        if self.reward == "attitude-detumbling": print("ub :", self.data["u0"].flatten()[:3].tolist())
        if self.reward == "ee-tracking" : print('p_ee : ', self.get_ee(self.data)[0].tolist())
        
    def plot_state(self):
        fig, axs = plt.subplots(3, sharex=True, figsize=(24,9))
        axs[0].plot(self.state[:3].swapaxes(1,0))
        axs[0].set_title('base angles [rad]')
        axs[1].plot(self.state[3:6].swapaxes(1,0))
        axs[1].set_title('base position [m]')
        axs[2].plot(self.state[6:].swapaxes(1,0))
        axs[2].set_title('joints angles [rad]')
        fig.suptitle('State')
        fig.tight_layout()
        fig.align_ylabels()
        fig.savefig(self.checkpoint_path + '_state_steps=' + str(self.epochs['training']) + '.png')
        plt.close(fig)
        
    
    def close(self):
        # Define any cleanup actions (optional)
        pass
    
    def get_inertia_matrix(self, q, param = None):
        robot=self.base_robot if param is None else setParams(params=param, robot=self.base_robot)
        # Generalized Inertia Matrix
        _, _, Hm = GIM(q=q, robot=robot)
        return Hm
    
    def get_coriolis_vector(self, q, qdot, param = None):
        # Generalized Convective Inertia Matrix
        robot=self.base_robot if param is None else setParams(params=param, robot=self.base_robot)
        _, _, _, Cm = CIM(q=q, qdot=qdot, robot=robot)     
        return Cm @ qdot[6:,:]
    
    def get_acceleration(self):
        manipulator_acc_magnitude = np.linalg.norm(self.data["umdot"])
        return manipulator_acc_magnitude
        
    
    def get_velocity_magnitude(self):
        manipulator_velocity_magnitude = np.linalg.norm(self.data["um"])
        attitude_velocity_magnitude = np.linalg.norm(self.data["u0"][:3])
        base_velocity_magnitude = np.linalg.norm(self.data["u0"][3:])
        return manipulator_velocity_magnitude + attitude_velocity_magnitude + base_velocity_magnitude
    
    def get_joints(self, base = False):
        if base : return np.block([[self.data["q0"]], [self.data["qm"]]])
        else : return self.data["qm"]
    
    def get_joints_velocity(self, base = False):
        if base : return np.block([[self.data["u0"]], [self.data["um"]]])
        else : return self.data["um"]
    
    def get_ee(self, data, out_velocity:bool=True):
        q0, qm = data["q0"], data["qm"]
        if out_velocity:
            u0, um = data["u0"], data["um"]
        else:
            u0, um = None, None
        return Joints2EE(q0=q0,qm=qm, u0=u0, um=um, robot=self.robot)
    
    def out_of_bounds(self):
        self.is_oob = np.any(np.absolute(self.data["um"]) > 800) \
            or np.any(np.absolute(self.data["u0"]) > 800) \
                or np.any(np.absolute(self.data["q0"][3:]) > 500) \
                    or any([np.any(np.isnan(v)) for v in self.data.values()]) 
        #self.is_oob = np.any(np.absolute(self.data["q0"][4:])>500)
        return self.is_oob
    
    def ID(self, q, qdot, qddot, params=None):
        robot = self.base_robot if params is None else setParams(params, self.base_robot)
        
        # no external forces and torques
        wF0 = np.zeros((6, 1))
        wFm = np.zeros((6, self.robot.n_links_joints))
        
        #q0, qm = q[:7,:], q[7:,:]
        q0, qm = q[:6,:], q[6:,:]
        #R0 = transpose(quat_DCM(transpose(q0[:4,:]).squeeze()))
        #r0 = q0[4:, :]  
        
        R0 = transpose(Angles321_DCM(q0[:3]))
        r0 = q0[3:, :]
        
        u0, um = qdot[:6,:], qdot[6:,:]

        # Floating case 
        if self.rw and not self.thrusters:
            u0dot, umdot = np.block([[qddot[:3, :]],[zeros((3, 1))]]), qddot[3:,:]
        if self.rw and self.thrusters:
            u0dot, umdot = qddot[:6,:], qddot[6:,:]
        else :
            assert not (not self.rw and self.thrusters)
            u0dot, umdot = zeros((6,1)), qddot
                
        _,RL,_,rL,e,g = Kinematics(R0, r0, qm, robot) 
        Bij, Bi0, P0, pm = DiffKinematics(R0, r0, rL, e, g, robot)
        t0, tL = Velocities(Bij, Bi0, P0, pm, u0, um, robot)
        t0dot, tLdot = Accelerations(t0, tL, P0, pm, Bi0, Bij, u0, um, u0dot, umdot, robot)
        I0, Im = I_I(R0, RL, robot)
        
        tau0, taum = ID(wF0,wFm,t0,tL,t0dot,tLdot,P0,pm,I0,Im,Bij,Bi0,robot)
        # floating case
        return tau0, taum
    
                    
    '''def train_inertia(self):
        # torque chosen randomly every N steps in range (-max_torque, max_torque)
        # for each step inertia and var are estimated 
        # implement early stopping
        self.is_train_inertia = True
        self.estimate_inertia = True
        self.inertia_estimator.train()
        
        try:
            while True:
                self.reset()
                done, is_terminated = False, False
                t = 0
                #for t in tqdm(range(1, self.max_traj_size + 1)):
                while not done and not is_terminated:   
                    t += 1              
                    tau = self.control.next(self.get_joints(base = True), self.get_joints_velocity(base = True), t*self.dt*self.n_steps)
                    tau = np.clip(tau, -self.max_action, self.max_action)
                    _, _, done, is_terminated, _ = self.step(tau.flatten())
        
                if self.end_train:
                    print("Stopping training ... ")
                    break

        except KeyboardInterrupt:
            pass 
        
        self.plot_inertia()  
    '''
    '''    def evaluate(self):
        # test with contant velocity and initial random torque
        N = 1//self.dt # frequency joint actuators
        self.is_train_inertia = False
        
        self.reset()
        errors = []
        vars = []
        self.inertia_estimator.eval()
        for t in tqdm(range(1, self.max_traj_size+1)):
            tau = self.control.next(self.get_joints(base = True), self.get_joints_velocity(base = True), t*self.dt*self.n_steps)
            
            self.step(tau.flatten())
            self.render()
            inertia_estimate = self.estimate
            var = self.var
            error = torch.abs((inertia_estimate - self.get_payload()).div(self.get_payload()))
            errors.append(error)
            vars.append(var.flatten())
        
        fig, axs = plt.subplots(1,2)
        errors = torch.stack(errors, dim = 1).cpu()
        for error in errors:
            axs[0].plot(error * 100)
        axs[0].set(xlabel='steps', ylabel='error', title='Error %')
        axs[0].legend(self.legend)
        vars = torch.stack(vars).cpu()
        axs[1].plot(vars)
        axs[1].set(xlabel='steps', ylabel='precision', title='precision')
        
        fig.suptitle('PPO-RECURRENT INERTIA ESTIMATOR')
        fig.tight_layout()
        fig.savefig("./output/images/sb3_ppo_recurrent_inertia_errorvar.png")
        plt.close(fig)  '''