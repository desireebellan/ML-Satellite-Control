from collections import namedtuple
from math import pi, sin, cos, sqrt

import numpy as np

import pickle
import torch

def read_pickle(pkl:str):
    with open(pkl, 'rb') as f:
        header = pickle.load(f)
        data = {}
        i = 0
        while True:
            try:
                data.update(pickle.load(f))
                i += 1
            except EOFError:
                return header, data
            
def write_pickle(data:dict, pkl):
    for d in data:
        pickle.dump({d:data[d]}, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        
def compute_reward(state, new_state, goal, state_pred):
    raise NotImplementedError

# computes the cross product between two quaternions
def quat_cross_product(q1, q2):
    return np.block([q1[3]*q2[:3] + q1[:3]*q2[3] - np.cross(q1[:3], q2[:3]), q1[3]*q2[3] - np.dot(q1[:3], q2[:3])])
def quat_inverse(q):
    return np.block([-q[:3], q[3]])/np.square(q).sum()

def quat2angle(quaternion):
    """
    Convert a quaternion to angle-axis representation.
    
    Args:
        quaternion (np.ndarray): Quaternion [x, y, z, w]
    
    Returns:
        angle (float): Rotation angle in radians
        axis (np.ndarray): Rotation axis [x, y, z]
    """
    quat_norm = np.linalg.norm(quaternion)
    angle = 2.0 * np.arccos(quaternion[3] / quat_norm)
    axis = quaternion[:3] / quat_norm
    return angle, axis

def quat_conjugate(quaternion):
    """
    Compute the conjugate of a quaternion.
    
    Args:
        quaternion (np.ndarray): Quaternion [w, x, y, z]
    
    Returns:
        conjugate (np.ndarray): Conjugate quaternion [w, -x, -y, -z]
    """
    conjugate = quaternion.copy()
    conjugate[:3] *= -1.0
    return conjugate
def quat_product(quat1, quat2):
    """
    Multiply two quaternions.
    
    Args:
        quat1 (np.ndarray): Quaternion [w, x, y, z]
        quat2 (np.ndarray): Quaternion [w, x, y, z]
    
    Returns:
        product (np.ndarray): Quaternion product [w, x, y, z]
    """
    x1, y1, z1, w1 = quat1
    x2, y2, z2, w2 = quat2
    product = np.array([      
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])
    return product

def random_quaternion():
   x = np.random.uniform(size = 3)
   theta = 2*pi*x[1:] 
   q = np.array([sin(theta[0])*sqrt(1-x[0]), cos(theta[0])*sqrt(1-x[0]), sin(theta[1])*sqrt(x[0]), cos(theta[1])*sqrt(x[0])])
   if q[3] < 0 : q = -q
   return q
    

class Trajectory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.new_states = []
        self.rewards = []
        self.action_log_prob = []
        self.masks = []
        
    def add(self,state,action,new_state,reward,mask, action_log_prob = None):
        self.states.append(state)
        self.actions.append(action)
        self.new_states.append(new_state)
        self.rewards.append(reward)
        self.action_log_prob.append(action_log_prob)
        self.masks.append(mask)
    
    def toTorch(self):
        if torch.is_tensor(self.states[0]):
            self.states = torch.stack(self.states).float()
            self.actions = torch.stack(self.actions).float()
            self.new_states = torch.stack(self.new_states).float()
            self.rewards = torch.stack(self.rewards).float()
            self.action_log_prob = torch.stack(self.action_log_prob).float()
            self.masks = torch.stack(self.masks).float()
        else:
            self.states = torch.FloatTensor(self.states)
            self.actions = torch.FloatTensor(self.actions)
            self.new_states = torch.FloatTensor(self.new_states)
            self.rewards = torch.FloatTensor(self.rewards)
            self.masks = torch.FloatTensor(self.masks)
               
    def get_raw(self,idx):
        return [self.states[idx],self.actions[idx],self.new_states[idx],self.rewards[idx]]
    
    def clear(self):
        self.states = []
        self.actions = []
        self.new_states = []
        self.rewards = []
        self.action_log_prob = []
        self.masks = []
    
    def __len__(self):
        return len(self.states)
    
# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    'Experience', field_names=['state', 'next_step', 'action', 'reward', 'done', 'state_val', 'action_log_prob', 'trajectory_marker'])
    
class ReplayBuffer():
    def __init__(self):
        self.buffer = []
    def append(self, experience:Experience):
        self.buffer.append(experience)
    def sample(self):
        state, next_step, action, reward, done, state_val, action_log_prob, trajectory_markers = zip(*self.buffer)
        return torch.stack(list(state)), torch.stack(list(next_step)), torch.stack(list(action)), list(reward), list(done), \
            torch.stack(list(state_val)), torch.stack(list(action_log_prob)), np.array(list(trajectory_markers))
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)
        