# Gymnasium environment expansion
# Parent Class Manipulator 
# Childrens classes Planar (2D) and Spacecraft (3D)

from utils import *
from control import Control
from inertia_estimator import InertiaEstimator
from hyperparameters import HyperParams
from typing import Optional, List, Union
from copy import deepcopy
from math import sin, cos, pi, exp, floor
from tqdm import tqdm

from robot_model import *
from spacecraftStep import *
from dynamics import *
from kinematics import *

import numpy as np
import matplotlib.pyplot as plt

import gymnasium
import torch
import pickle
import sys

def get_parameters(object, hparams):
    object.__dict__.update({name: getattr(hparams, name) for name in hparams.__dict__})
        

class Manipulator(gymnasium.Env):
    def __init__(self, hparams:HyperParams, space_dim:dict, control:Union[PD, ComputedTorque, Control] = None,
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
        self.inertia_estimator = InertiaEstimator(hparams=hparams)
        self.best_loss         = sys.float_info.max
        self.patience          = 0
        self.end_train         = False
        
        # Training control routine 
        if control is not None :
            self.qd, self.qddot, self.qdddot = sinusoids_trajectory(n_joints = hparams.n_joints, 
                                                                    T        = hparams.T, 
                                                                    q0       = self.q0, 
                                                                    nmax     = self.harmonics) 
            self.control = control( qd        = self.qd, 
                                    qddot     = self.qddot, 
                                    qdddot    = self.qdddot, 
                                    ID        = self.ID, 
                                    num_links = self.n_joints)
        else : self.control = None
 
    def reset(self, seed:int = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.t = 0 
        
        if (self.continuous_trajectory and (self.done or self.is_oob)) or not self.continuous_trajectory:
            # Update epoch
            if self.mode == 'train':
                for key in self.epochs : 
                    if key != "hardness" : self.epochs[key] += 1
            # Reset the environment to its initial state
            print('--------------------------------------------------')
            print('Reset environment')
            self.random_reset()  

            if self.update_payload_freq > 0 and self.epochs['payload'] >= self.update_payload_freq :
                print('Update Payload')
                self.ee = self.configure_payload(self.payload)
                self.epochs['payload'] = 0
                
            self.state        = self.get_joints().reshape(self.n_joints, 1)
            self.actions      = np.zeros((self.n_joints + 3 * self.rw, 1))
            self.error_reward = []
                
            #self.episode_start = 1
            reward, _, _         = self.calculate_reward()
            self.episode_rewards = [reward]
            self.episode_errors  = []
            self.episode_vars    = []
            self.current_reward  = reward
            self.absolute_error  = None
            
            if hasattr(self, "prev_error"): del self.prev_error
            
            print('--------------------------------------------------')
        
        if self.control is not None and self.epochs["control"] >= self.reset_control_rate: 
            print("Reset control")
            self.control.qd, \
                self.control.qddot, \
                    self.control.qdddot = sinusoids_trajectory(n_joints=self.n_joints, 
                                                               T       =self.T, 
                                                               q0      = self.q0, 
                                                               nmax    = self.harmonics)
            self.epochs["control"] = 0

        return self.get_input_parameters(self.data), {}
        
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
        # Final random inertia matrix rotated by Q
        # I = np.transpose(Q) @ J @ Q
         
        return {'mp': m, 'rp': rcom, 'Ip': J, 'R': Q}  
        
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
        self.state = np.block([self.state, self.get_joints().reshape(self.n_joints, 1)])
        self.actions = np.block([self.actions, action.reshape(self.n_joints + 3*self.rw, 1)])

        # Define the reward function based on the current state and 
        # Define if the episode is done (terminal state reached)
        reward, done, is_terminated = self.calculate_reward()
        
        self.estimate, error, self.var = self.update_inertia(next_state, action, is_terminal=is_terminated or done)
        if self.reward == "inertia-variance": 
            reward -= self.var.item()
        if self.reward == "inertia-error":
            reward -= error.sum().item()
        
        self.update(reward, error, self.var, done, is_terminated)
        
        
        self.log(done, is_terminated)
        
        if done:
            print("GOAL REACHED !!!")
        elif not done and is_terminated:
            print("Trajectory concluded w/o reaching goal")

        if done or is_terminated:
            print("Episode length : " + str(self.t))
            
        self.done = done
        
        next_state = np.block([next_state.flatten(), action.flatten()])

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
            
                inertia_estimate, var = self.inertia_estimator.predict(observation = observation,  action=action, is_terminal=is_terminal, target=target)
                error = torch.abs(inertia_estimate - self.get_payload())/torch.abs(self.get_payload())
                self.absolute_error = torch.abs(inertia_estimate - self.get_payload())
                #if self.episode_start == 1 : self.episode_start = 0
            else : self.inertia_estimator.episode_start = 1
        
        if self.is_train_inertia and self.inertia_steps == self.inertia_buffer_size:
            # Training Inertia Model
            print('----------------------------------')
            print("Updating Inertia Estimator ")
            self.inertia_steps = 0
            loss = self.inertia_estimator.training_step()
            
            assert not torch.isnan(loss), print("Loss is NaN!!!")
            
            torch.save(self.inertia_estimator.state_dict(), self.checkpoint_path + '_model{}.pth'.format(self.epochs['training']))
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
        if done or is_terminated: 
                
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
                #average_error = sum(self.episode_errors)/len(self.episode_errors)
                average_error = self.episode_errors[-1]
                self.errors.append(average_error)
                print("Last Relative Error :", average_error.tolist(), " Mean :", average_error.mean().item())
                
                # last absolute error
                if self.absolute_error is not None:
                    print("Last Absolute Error :", self.absolute_error.tolist(), " Mean :", self.absolute_error.mean().item())
            
                # average variance
                #average_variance = sum(self.episode_vars)/len(self.episode_vars)
                average_variance = self.episode_vars[-1]
                self.vars.append(average_variance)
                print("Average Episode Variance :", average_variance.item())
            
            if self.is_train_inertia:
                if len(self.losses) > 0 : 
                    print("Estimator Loss :", self.losses[-1], "Best Loss :", self.best_loss, "N° epochs without decreasing loss :", self.patience) 
            
            if len(self.rewards) % self.plot_freq == 0:
                    self.save(self.checkpoint_path) 
                    if self.plot_reward: 
                        self.plot()
                        self.plot_reward_error()

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
        
        fig.savefig("./output/images/sb3_ppo_recurrent_reward.png")
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
        axs[2].set_title("Variance")
        
        fig.suptitle('PPO-RECURRENT INERTIA ESTIMATOR TRAINING')
        fig.tight_layout()
        fig.savefig("./output/images/sb3_ppo_recurrent_inertia_losserrorval_" + str(self.epochs['training']) + '.png')
        plt.close(fig)  
        
    def plot_state(self):
        fig, axs = plt.subplots(1, self.n_joints)
        for i in range(self.n_joints):
            axs[i].plot(self.state[i])
            axs[i].set_title("q" + str(i))
        fig.suptitle('PLOT')
        fig.tight_layout()
        fig.savefig("./output/images/plot_" + str(self.epochs['training']) + '.png')
        plt.close(fig)
        
    def plot_reward_error(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.error_reward)
        fig.suptitle('PLOT ERROR')
        fig.tight_layout()
        fig.savefig("./output/images/plot_error_" + str(self.epochs['training']) + '.png')
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
        if self.is_train_inertia : 
            self.inertia_estimator.load_state_dict(torch.load(self.checkpoint_path + '_model.pth', map_location=lambda storage, loc: storage))
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
                
    def train_inertia(self):
        # torque chosen randomly every N steps in range (-max_torque, max_torque)
        # for each step inertia and var are estimated 
        # implement early stopping
        self.is_train_inertia = True
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
                    print("tau", tau)
                    tau = np.clip(tau, -self.max_action, self.max_action)
                    #tau = np.random.rand(3, 1)*2-1
                    _, _, done, is_terminated, _ = self.step(tau.flatten())
                    #if done or is_terminated:
                    #    break
        
                if self.end_train:
                    print("Stopping training ... ")
                    break

        except KeyboardInterrupt:
            pass 
        
        self.plot_inertia()  
        
    def evaluate(self):
        # test with contant velocity and initial random torque
        N = 1//self.dt # frequency joint actuators
        self.is_train_inertia = False
        
        self.reset()
        errors = []
        vars = []
        self.inertia_estimator.eval()
        for t in tqdm(range(1, self.max_traj_size+1)):
            tau = self.control.next(self.get_joints(base = True), self.get_joints_velocity(base = True), t*self.dt*self.n_steps)
            #print("tau",tau)
            
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
        axs[1].set(xlabel='steps', ylabel='variance', title='Variance')
        
        fig.suptitle('PPO-RECURRENT INERTIA ESTIMATOR')
        fig.tight_layout()
        fig.savefig("./output/images/sb3_ppo_recurrent_inertia_errorvar.png")
        plt.close(fig)  

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
        
        reward = -velocity_penalty - oob_penalty - energy_penalty
        done = True
        
        if self.reward == "ee-error":
            e_next = np.linalg.norm(self.joint_pos[-1](self.data).flatten() - self.joint_pos[-1](self.goal).flatten())
            done = done and e_next < self.threshold
            reward += -e_next 
        
        return reward, done, self.t == self.max_traj_size
    
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
        
    def get_dynamic_parameters(self, robot = None, params = None):
        robot = self.robot if robot is None else robot
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
        
    def get_inertia_matrix(self, q, param = None):
        if param is None : param = {"mp": 0.0 , "rp": np.zeros(3).astype(np.float32), "Ip": np.zeros((3,3)).astype(np.float32)}
        a = self.get_dynamic_parameters(param=param)
        # m11 = a1 + a2 + 2a3c2 - 2a4s2
        m11 = a[0] + a[1] + 2*a[2]*cos(q[1]) - 2*a[3]*sin(q[1])
        # m12 = m21 = a2 + a3c2 - a4s2
        m12 = a[1] + a[2]*cos(q[1]) - a[3]*sin(q[1])
        # m22 = a2
        m22 = a[1]
        return np.array([[m11, m12],[m12, m22]])
    
    def get_coriolis_vector(self, q, qdot, param = None):
        if param is None : param = {"mp": 0.0 , "rp": np.zeros(3).astype(np.float32), "Ip": np.zeros((3,3)).astype(np.float32)}
        a = self.get_dynamic_parameters(param=param)
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
        self.legend = ['mass', 'rcom_x', 'rcom_y', 'rcom_z', 'I_xx', 'I_yy', 'I_zz']
        self.q0 = initial_state["qm"]
        self.base_robot = robot

        super().__init__(hparams, space_dim, control, initial_state, robot, payload_ranges)
        
    def configure_payload(self, payload: Optional[dict] = None):
        param = super().configure_payload(payload)
        self.robot = setParams(param, self.base_robot)
        return param
    
    def get_input_parameters(self, data:dict) -> np.ndarray:
        # angles
        state = np.block([data["q0"].flatten(), data["qm"].flatten(), data["u0"].flatten(), data["um"].flatten(), 
                          data["u0dot"].flatten(), data["umdot"].flatten()])
        return state 
    
    def get_payload(self):
        return torch.tensor([self.ee['mp']] +  self.ee['rp'].flatten().tolist() + np.diag(self.ee['Ip']).flatten().tolist()).to(self.device)
    
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
        p1 = self.data["q0"][4:].reshape((3, 1))
        # joint positions
        rj = np.block([p1, self.joint_pos(self.data)]).T
        
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
        
        tau = tau.reshape((3*self.rw + self.n_joints,1)).astype(np.float32)
        tau0 = np.zeros((6, 1)) if not self.rw else np.block([[tau[:3].reshape(3, 1)], [np.zeros((3, 1))]])
        taum = tau if not self.rw else tau[3:].reshape(self.n_joints, 1)
        #tau = tau * self.max_action
        
        # no external forces and torques
        wf0 = np.zeros((6, 1))
        wfm = np.zeros((6, self.robot.n_links_joints))
        
        data = spacecraftStep(tau0=tau0, taum=taum, wf0=wf0, wfm=wfm, data=self.data, robot = robot, dt=self.dt, n_steps=self.n_steps)

        return data
    
    def calculate_reward(self):
        reward, done, is_terminated = super().calculate_reward()   
        
        # terminate the sequence if the robot exceed bounds
        
        if self.is_oob : return 1e-3, False, True
        
        threshold = 0.1    
        alpha = 1e4    

        if self.reward in {"attitude-error", "attitude-direction", "attitude"}:
            # compute quaternion error between current attitude and desired attitude
            error = scalar_quat_error(self.data["q0"][:4], self.goal[:4])
            done = done and error <= threshold and self.get_velocity_magnitude() <= 0.1
            reward += done.astype(np.float32)*2
            
            self.error_reward.append(error)

            if self.reward in {"attitude-error", "attitude"}:
                # r \in [-1, 0]
                reward -= error
            if self.reward in {"attitude-direction", "attitude"}:    
                # r = {-1, 0}                    
                if hasattr(self, "prev_error"):
                    reward += -(error >= self.prev_error).astype(np.float32)
                    #diff = alpha * scale((error < self.prev_error).astype(np.float32), max = 1, min = -1) * abs(error - self.prev_error)
                    #reward += diff 
                    #reward += self.prev_error - error     
                    #print(error, self.prev_error, reward)  
            self.prev_error = error    
        
        if self.reward  not in {"attitude-error", "attitude-direction", "ee-error"}: 
            done = False

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
                #self.data.update({'q0': np.block([random_quaternion(goal = self.goal, sigma = sigma),np.zeros(3)]).reshape((7, 1)).astype(np.float32)})
                self.data.update({'q0': np.block([random_quaternion(goal = euler2quat(np.ones((3, 1)), pi/4), sigma = sigma),np.zeros(3)]).reshape((7, 1)).astype(np.float32)})
                self.initial_state.update({'q0' : deepcopy(self.data["q0"])})
                    
                if self.epochs['hardness'] >= self.hardness_rate:
                    self.epochs['hardness'] = 0
                    self.hardness += 0.1
            elif self.initialize == "constant":
                self.data.update({'q0': np.block([euler2quat(np.ones((3, 1)), pi/4), np.zeros(3)]).reshape((7, 1)).astype(np.float32)})
        print("q0 :", self.data["q0"].flatten()[:4].tolist())
        
    
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
    
    def joint_pos(self, data):  
        q0, qm = data["q0"], data["qm"]
        return joint2EE(q0=q0,qm=qm, robot=self.robot)
    
    def out_of_bounds(self):
        self.is_oob = np.any(np.absolute(self.data["um"]) > 800) \
            or np.any(np.absolute(self.data["u0"]) > 800) \
                or np.any(np.absolute(self.data["q0"][4:]) > 500) \
                    or any([np.any(np.isnan(v)) for v in self.data.values()]) 
        #self.is_oob = np.any(np.absolute(self.data["q0"][4:])>500)
        return self.is_oob
    
    def ID(self, q, qdot, qddot, params=None):
        robot = self.base_robot if params is None else setParams(params, self.base_robot)
        
        # no external forces and torques
        wF0 = np.zeros((6, 1))
        wFm = np.zeros((6, self.robot.n_links_joints))
        
        q0, qm = q[:7,:], q[7:,:]
        R0 = transpose(quat_DCM(transpose(q0[:4,:]).squeeze()))
        r0 = q0[4:, :]  
        
        u0, um = qdot[:6,:], qdot[6:,:]

        # Floating case 
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
    from argparse import ArgumentParser, BooleanOptionalAction
    from gymnasium.spaces.box import Box
    from math import ceil
    
    parser = ArgumentParser(description = "Recurrent PPO sb3")
    
    hparams = HyperParams
    
    parser.add_argument("--mode", default = "train", type = str, help = "script mode [train, test]", choices=["train", "test"])
    parser.add_argument("--model", default = "planar", type = str, help = "environment [planar, spacecraft]", choices = ["planar", "spacecraft"])
    parser.add_argument("--max-action", default = 10, type = float, help = "max torque value")
    parser.add_argument("--seed", default = 0, type = int, help = "seed value")
    parser.add_argument("--ckpt", default=None, type = str, help = "loading checkpoint path")
    parser.add_argument('--batch-size', default = None, type = int, help = "size of update minibatch")
    parser.add_argument('--update-inertia', default = 32, type = int, help = "inertia dataset size")
    parser.add_argument('--update-payload-freq', default = -1, type = int, 
                        help = "freqency payload update.If -1 there is no payload, if 0 the payload is constant")
    parser.add_argument('--lr', default = 1e-3, type = float, help = "learning rate")
    parser.add_argument('--epochs', default=hparams.train_K, type=int, help="number of training epochs for each batch")
    parser.add_argument('--compute-variance', default = False, type = bool, action=BooleanOptionalAction)
    parser.add_argument('--num-layers', default = 1, type = int, help = "number of lstm layers inertia estimator")
    
    #parser.add_argument('--control', default = "PD", type = str, help = "type of control during training and evaluation")
    parser.add_argument('--reset-control-rate', default = 1, type = int, help = "number of epochs before a desired trajectory update")
    parser.add_argument('--harmonics', default = 5, type = int, help = "maximum number of harmonics in the desired trajectory")
    
    parser.add_argument('--training-steps', default = 6, type = int, help = "number of episodes before learning")
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
    
    args = parser.parse_args()
    
    # hyperparameters
    
    hparams.max_traj_size    = ceil((args.max_time // args.dt)/args.n_steps)
    hparams.update_timestep  = hparams.max_traj_size * args.training_steps
    hparams.update_epochs    = args.training_steps
    hparams.n_steps          = args.n_steps

    hparams.max_action       = args.max_action
    hparams.seed             = args.seed
    hparams.payload          = None if args.update_payload_freq >= 0 else {"mp": 0.0 , "rp": np.zeros(3).astype(np.float32), "Ip": np.zeros((3,3)).astype(np.float32)}
    hparams.update_payload_freq = args.update_payload_freq
    hparams.estimate_inertia = True
    hparams.inertia_buffer_size= args.update_inertia * hparams.max_traj_size
    #hparams.T               = hparams.max_traj_size * hparams.dt / 2
    hparams.T                = args.max_time
    
    hparams.robot_filename   = "./matlab/SC_3DoF.urdf"
    hparams.checkpoint_path  = './output/inertia_estimate' 
    hparams.inertia_batch_size= args.batch_size
    hparams.lr               = args.lr 
    hparams.train_K          = args.epochs
    hparams.compute_variance = args.compute_variance
    
    hparams.mode             = args.mode
    
    hparams.min_reward       = -1e3
    
    hparams.num_layers       = args.num_layers
    
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
    hparams.thusters         = args.thrusters
    
    hparams.plot_reward      = False
    
    control = Control
    

    if args.model == "planar":
        observation_space = Box(low = -np.inf, high = np.inf, shape = (6,), dtype = np.float32)
        action_space = Box(low = -hparams.max_action, high = hparams.max_action, shape = (2,1))
        hparams.n_joints = 2

        space_dim = {"observation-space":observation_space, "action-space":action_space}
        
        hparams.state_dim        = hparams.n_joints * 3
        hparams.action_dim       = hparams.n_joints
        hparams.inertia_dim      = 4 

        # initialize environment
        initial_state = {
            'q':np.zeros((2,1)).astype(np.float32),
            'qdot':np.zeros((2,1)).astype(np.float32),
            'qdotdot':np.zeros((2,1)).astype(np.float32),
            'tau':np.zeros((2,1)).astype(np.float32)
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
        env = Planar2R(hparams=hparams, space_dim=space_dim, control = control, robot=robot, initial_state=initial_state, payload_ranges=payload_ranges)
        env.goal = np.array([[pi/2, 0, 0], [0, 0, 0]])
        
    elif args.model == "spacecraft":
        # initialize environment
        from utils import random_quaternion
        
        filename = "./matlab/SC_3DoF.urdf"
        robot, _ = urdf2robot(filename=filename)
        n = robot.n_q
        hparams.n_joints = n
        
        state_dim = 7 + 6*2 + n*3
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
                    'tau': np.zeros((6 + n,1)).astype(np.float32) # manipulator joint torques
            }
        env = Spacecraft(hparams, space_dim=space_dim, control = control, initial_state=initial_state, robot=robot)
        env.goal = np.array([0.,0.,0.,1.])
        
    env.is_train_inertia = True
        
    if args.ckpt is not None:
        print("Loading Environment data")
        
        env.load(args.ckpt + '/inertia_estimate')
                
    if args.mode == "train":
        print("---------------------------------------------------------")
        print("Learning Phase Started")
        env.train_inertia()
        print('---------------------------------------------------------')
        print('Learning phase compleated')
        
    elif args.mode == "test":     
        if args.ckpt is None : 
            raise Exception("Testing mode has been selected, but no checkpoint is defined!")    

    print('----------------------------------------------------------')
    print('Evaluation phase started')
    env.is_train_inertia = False
    env.evaluate()
    print('Evaluation phase compleated')