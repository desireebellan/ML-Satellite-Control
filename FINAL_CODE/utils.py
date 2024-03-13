import numpy as np
from typing import List, Dict
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from math import sin, cos, pi, sqrt, acos

import torch
import random
import pickle
import io

# CLASSES

class AttributeDict(dict):
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    
'''class dataset(Dataset):
    def __init__(self, data:List[Dict]):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
        
        
class TrajectoryDataLoader(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()     
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers  
        
        self.g = torch.Generator()
        self.g.manual_seed(0)

        with open(self.data_dir, 'r') as f:
            header = f.readline()
            self.dim_input, traj_size, n, self.num_out = tuple(np.array(header.strip().split(','), dtype=np.int32).tolist())
            self.steps = int(traj_size // n)
        data = np.genfromtxt(self.data_dir, delimiter = ',', skip_header = 1)
        self.ids, self.outputs, self.inputs = data[:,0], data[:,1:self.num_out + 1], data[:,self.num_out + 1:]
        
    def prepare_data(self):
        self.data = []
        for input, output in zip(self.inputs, self.outputs):
            shape = (self.step, self.dim_input) if self.dim_input > 1 else self.step
            self.data.append({'input': torch.FloatTensor(input).reshape(shape), 'output': torch.FloatTensor(output)})           
    def setup(self, stage = str):
        if stage == 'fit':
            self.train_dataset = dataset(self.data[:(int)(self.inputs.shape[0]*0.7)])
            self.validation_dataset = dataset(self.data[(int)(self.inputs.shape[0]*0.7):(int)(self.inputs.shape[0]*0.85)])
        elif stage == 'test':
            self.test_dataset = dataset(self.data[(int)(self.inputs.shape[0]*0.85):])
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, \
            num_workers=self.num_workers, worker_init_fn=self.seed_worker, generator=self.g)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers = self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers = self.num_workers)

    def seed_worker(self, worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)'''
            
class OnlineDataset(Dataset):
    def __init__(self, pkl:str):

        try:
            self.header, self.data = read_pickle(pkl)
        except:
            print("pkl does not exist")
            print(pkl)
            self.header, self.data = read_pickle(pkl)
            exit(0)
        self.samples = len(self.data)

    def __getitem__(self, index):   
        return self.data[index]
    def __len__(self):
        return self.samples
            
class OnlineDataLoader(pl.LightningDataModule):
    def __init__(self, pkl_train:str = None, pkl_dev:str = None, pkl_test = None, 
                 batch_size:int = 32, num_workers:int = 1):
        super().__init__()     
        
        self.batch_size = batch_size
        self.num_workers = num_workers  
        
        self.g = torch.Generator()
        self.g.manual_seed(0)
        
        self.pkl_train = pkl_train
        self.pkl_dev = pkl_dev
        self.pkl_test = pkl_test
                
    def setup(self, stage = str):
        if stage == 'fit':
            self.train_dataset = OnlineDataset(pkl = self.pkl_train)
            self.validation_dataset = OnlineDataset(pkl = self.pkl_dev)
        elif stage == 'test':
            self.test_dataset = OnlineDataset(pkl = self.pkl_test)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, \
            num_workers=self.num_workers, worker_init_fn=self.seed_worker, generator=self.g,\
                )

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers = self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers = self.num_workers)

    def seed_worker(self, worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            

class Metrics(pl.Callback):
    def __init__(self, name:str, device, parameters = ['a1', 'a2', 'a3', 'a4']):
    
        self.name = name
        self.device = device
        self.parameters = parameters
        self.output_dim = len(parameters)

        self.collection = {}
        self.val_epoch = {}
        for i in range(self.output_dim):
            self.collection['train_loss'+str(i+1)] = []
            self.collection['train_error'+str(i+1)] = []
            self.collection['dev_loss'+str(i+1)] = []
            self.collection['dev_error'+str(i+1)] = []
            
            self.val_epoch['loss'+str(i+1)] = []
            self.val_epoch['error'+str(i+1)] = []
            
        self.preds = torch.tensor([]).to(device)
        self.targets = torch.tensor([]).to(device)

    def on_train_epoch_end(self, trainer, pl_module):
        for i in range(self.output_dim):
            self.collection['train_loss'+str(i+1)].append(trainer.callback_metrics['train_loss'+str(i+1)].tolist())
            self.collection['train_error'+str(i+1)].append(trainer.callback_metrics['train_error'+str(i+1)].tolist())
        
    def on_validation_epoch_end(self, trainer, pl_module):
        for i in range(self.output_dim):
            self.val_epoch['loss'+str(i+1)].append(trainer.callback_metrics['valid_loss'+str(i+1)].tolist())
            self.val_epoch['error'+str(i+1)].append(trainer.callback_metrics['valid_error'+str(i+1)].tolist())
    
    def on_validation_end(self, trainer, pl_module):
        for i in range(self.output_dim):
            self.collection['dev_loss'+str(i+1)].append(sum(self.val_epoch['loss'+str(i+1)])/len(self.val_epoch['loss'+str(i+1)]))
            self.collection['dev_error'+str(i+1)].append(sum(self.val_epoch['error'+str(i+1)])/len(self.val_epoch['error'+str(i+1)]))
            self.val_epoch['loss'+str(i+1)] = []
            self.val_epoch['error'+str(i+1)] = []

    def on_test_end(self, trainer, pl_module):
      # PLOT LOSSES CURVES
      fig, axs = plt.subplots(1,self.output_dim, figsize = (self.output_dim*3, 3))
      x = range(len(self.collection['train_loss1']))
      if self.output_dim > 1:
        for i in range(self.output_dim):
            axs[i].set_title('Loss ' + self.parameters[i])
            #axs[i].plot(x, self.collection['train_loss'+str(i+1)], label = 'train'+str(i+1))
            axs[i].plot(x,self.collection['dev_loss'+str(i+1)][1:len(self.collection['train_loss'+str(i+1)])+1], label = 'valid'+str(i+1))
            handles, labels = [a for a in axs[0].get_legend_handles_labels()]
      else:
            axs.set_title('Loss')
            #axs.plot(x, self.collection['train_loss1'], label = 'train')
            axs.plot(x,self.collection['dev_loss1'][1:len(self.collection['train_loss1'])+1], label = 'valid')
            handles, labels = [a for a in axs.get_legend_handles_labels()]
      fig.legend(handles, labels, loc='upper right')

      plt.savefig('./output/images/{}_plot_loss(valid).png'.format(self.name))
      
      # PLOT ERROR CURVES
      fig, axs = plt.subplots(1,self.output_dim, figsize = (self.output_dim*3, 3))
      x = range(len(self.collection['train_error1']))
      if self.output_dim > 1:
        for i in range(self.output_dim):    
            axs[i].set_title('Error ' + self.parameters[i])
            #axs[i].plot(x, self.collection['train_error'+str(i+1)], label = 'train'+str(i+1))
            axs[i].plot(x,self.collection['dev_error'+str(i+1)][1:len(self.collection['train_error'+str(i+1)])+1], label = 'valid'+str(i+1))
            handles, labels = [a for a in axs[0].get_legend_handles_labels()]
      else:
            axs.set_title('Error')
            #axs.plot(x, self.collection['train_error1'], label = 'train')
            axs.plot(x,self.collection['dev_error1'][1:len(self.collection['train_error1'])+1], label = 'valid')
            handles, labels = [a for a in axs.get_legend_handles_labels()]
      fig.legend(handles, labels, loc='upper right')

      plt.savefig('./output/images/{}_plot_error(valid).png'.format(self.name))

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        checkpoint['metrics'] = {'collection': self.collection}
        checkpoint['current_epoch'] = trainer.current_epoch
        
    def load(self, ckpt_path) -> None:
        checkpoint = torch.load(ckpt_path, map_location=self.device) 
        self.collection = checkpoint['metrics']['collection']
        
class Safe_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        else:
            return super().find_class(module, name)
    
# FUNCTIONS 

def safe_collate(batch):
    # find min size
    
    min_length = min(batch, key = lambda k: len(k["input"]))
    for i in range(len(batch)):
        batch[i]["input"] = batch[i]["input"][:min_length]
    input = torch.stack([batch[i]["input"] for i in range(len(batch))])
    output = torch.stack([batch[i]["output"] for i in range(len(batch))])
    return{"input": input, "output": output}

def read_pickle(pkl:str):   
    with open(pkl, 'rb') as f:
        #loader=
        header = Safe_Unpickler(f).load()
        data = {}
        i = 0
        while True:
            try:
                #data.update(pickle.load(f))
                data.update(Safe_Unpickler(f).load())
                i += 1
            except EOFError:
                return header, data
            
def write_pickle(data:dict, pkl, step = None):
    for d in data:
        if step is not None:
            data[d]["input"] = data[d]["input"][:step]
        pickle.dump({d:data[d]}, pkl, protocol=pickle.HIGHEST_PROTOCOL)

def scale(x, min = 0, max = 1):
    return x * (max - min) + min

def scalar_quat_error(q, qd, case:int=1):
    qe = quat_error(q, qd)
    if case == 1 : return np.arctan2(np.sqrt(np.linalg.norm(qe[:3])), q[3]).squeeze()
    elif case == 2 : return np.linalg.norm(qe[:3])*q[3]
    else : raise NotImplementedError

def quat_error(q, qd):
    #qe = quat_product(q, quat_conjugate(qd))
    qe = quat_product(q, quat_conjugate(qd))
    if qe[3] < 0 :  return -qe 
    return qe

def quat_product(q1, q2):
    """
    Multiply two quaternions.
    
    Args:
        quat1 (np.ndarray): Quaternion [x, y, z, w]
        quat2 (np.ndarray): Quaternion [x, y, z, w]
    
    Returns:
        product (np.ndarray): Quaternion product [x, y, z, w]
    """
    '''x1, y1, z1, w1 = quat1
    x2, y2, z2, w2 = quat2
    product = np.array([      
        w1*x2 + w2*x1 + y1*z2 - z1*y2,
        w1*y2 + w2*y1 + z1*x2 - x1*z2  ,
        w1*z2 + w2*z1 + x1*y2 - y1*x2 ,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])'''
    w1, w2 = q1[3], q2[3]
    v1 = q1[:3].reshape(3, 1)
    v2 = q2[:3].reshape(3, 1)
    product = np.vstack([
        w1 * v2 + w2 * v1 - np.cross(v1, v2, axis = 0), 
        w1 * w2 - np.dot(v1.T, v2)
    ])
    product = product / np.linalg.norm(product)
    return product

def random_quaternion(max = 1.0, sigma = 1.0, goal = np.array([0, 0, 0, 1])):
    _, e_mean = quat2euler(goal)
    theta = np.random.uniform(high = max, size = 1) * 2 * pi 
    #theta = np.random.normal(loc = theta_mean, scale = sigma) % 2*pi
    e = np.random.normal(loc = e_mean, scale = sigma)
    q = np.block([e * sin(theta/2), cos(theta/2)])
    if q[3] < 0 : q = -q
    return q / np.linalg.norm(q)

def quat2euler(q):
    alpha = 2*acos(q[3])
    if alpha % pi == 0: euler_axis = np.array([0, 0, 1])
    else : euler_axis = 1/sin(alpha/2) * q[0:3]
    return alpha, euler_axis/np.linalg.norm(euler_axis)

def euler2quat(e, theta):
    e = e.flatten() / np.linalg.norm(e)
    q = np.block([e * sin(theta/2), cos(theta/2)])
    if q[3] < 0 : q = -q
    return q / np.linalg.norm(q)

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

def quaternionIntegrate(q, w, dt):
    if np.linalg.norm(w) == 0 : return q
    # Apply angular velocity to compute the updated quaternion
    # Normalize the quaternion
    q = q / np.linalg.norm(q);
    # Compute the rotation angle and axis
    angle = np.linalg.norm(w) * dt;
    axis = w / np.linalg.norm(w);
    # Compute the quaternion exponential
    quat_exp = euler2quat(axis, angle)
    # Multiply the quaternions
    q_new = quat_product(quat_exp, q)
    # Normalize the resulting quaternion
    q_new = q_new / np.linalg.norm(q_new)
    
    return q_new

def rad2grad(x):
    return x * 180 / pi

def angle_diff(x1, x2):
    x1 = normalize_angle(x1)
    x2 = normalize_angle(x2)
    return normalize_angle(x2-x1, center=0)

def normalize_angle(alpha, center = pi):
    return (alpha-center+pi)% (2*pi)+ (center-pi);

def angle_error(x1:np.ndarray, x2:np.ndarray) -> np.ndarray:
    e1 = np.abs(x1 - x2)
    e2 = 2 * pi - e1
    sign = ~ np.logical_xor(x1 < x2, e1 < e2)
    sign = sign * 2 - 1
    return np.minimum(e1, e2) * sign

def transpose(x:np.ndarray):
    if len(x.shape) > 1: return x.T 
    else : return x.reshape((-1,1))
    
def SkewSym(x:np.ndarray): # SPART LIBRARY
    # Computes the skew-symmetric matrix of a vector, which is also the
    # left-hand-side matricial equivalent of the vector cross product
    #
    # [x_skew] = SkewSym(x)
    #
    # :parameters:
    #	* x -- [3x1] column matrix (the vector).
    #
    # :return:
    #	* x_skew -- [3x3] skew-symmetric matrix of x.

    return np.block([[0, -x[2], x[1]] , [x[2], 0, -x[0]] , [-x[1], x[0], 0] ])

def Omega(x:np.ndarray):
    x1, x2, x3 = x
    return np.block([[0  , x3 , -x2, x1],
                     [-x3, 0  , x1 , x2],
                     [x2 , -x1, 0  , x3],
                     [-x1, -x2, -x3, 0]])
    

def sinusoids_trajectory(n_joints:int, T:float, q0:np.ndarray, harmonics:int = 5, coefMax:int=1):
    wf = 2*pi/T
    n = np.random.randint(1, harmonics)
    a = np.random.rand(n_joints, n)*(2*coefMax) - coefMax
    b = np.random.rand(n_joints, n)*(2*coefMax) - coefMax

    # Truncated Fourier series
    q_ = lambda t : sum([a[:,k-1]/(wf*k)*sin(wf*k*t) - b[:,k-1]/(wf*k)*cos(wf*k*t) for k in range(1, n+1)]).reshape((n_joints, 1))
    qdot_ = lambda t : sum([a[:,k-1]*cos(wf*k*t) + b[:,k-1]*sin(wf*k*t) for k in range(1, n+1)]).reshape((n_joints, 1))
    qdotdot_ = lambda t : sum([-a[:,k-1]*wf*k*sin(wf*k*t) + b[:,k-1]*wf*k*cos(wf*k*t) for k in range(1, n+1)]).reshape((n_joints, 1))
    
    # Fifth order polynomial 
    # The free-coefficients cj should minimize the conition number of the regressor matrix
    # In this case are dtermined by using the desired initial and final conditions
    # q(0) = 0, q(N) = 0, qdot(0) = 0, qdot(N) = 0, qdotdot(0) = 0, qdotdot(N) = 0
    # The minimization process is left to the algorithm
    c = np.zeros((n_joints, 6))
    M = np.eye(6)
    M[2,2] = 2
    M[3:,:] = np.array([[0 if k > i else np.prod([i-j for j in range(0, k)])*T**(i-k) for i in range(6)] for k in range(3)])
    Minv = np.linalg.inv(M)
    qv = np.array([-q_(0) + q0, -qdot_(0), -qdotdot_(0), -q_(T) + q0, -qdot_(T), -qdotdot_(T)]).reshape((6, n_joints)).swapaxes(1,0)
    for i in range(n_joints):
        c[i, :] = Minv @ qv[i]  
    q = lambda t: sum([c[:, k]*t**k for k in range(6)]).reshape((n_joints, 1)) + q_(t)
    #qdot = lambda t : sum([c[:,k]*k*t**(k-1) for k in range(1,6)]).reshape((n_joints, 1)) + qdot_(t)
    #qdotdot = lambda t : sum([c[:,k]*k*(k-1)*t**(k-2) for k in range(2,6)]).reshape((n_joints, 1)) + qdotdot_(t)
    qdot = lambda t : np.zeros((n_joints, 1))
    qdotdot = lambda t : np.zeros((n_joints, 1))
    return q, qdot, qdotdot


def update_hparams(object, hparams):
        if not isinstance(hparams, dict):
            object.hparams.update(dict((name, getattr(hparams, name)) for name in dir(hparams) if not name.startswith('__')))
            # set hyperparameters
            object.__dict__.update({name: getattr(hparams, name) for name in hparams.__dict__})          
        else:
            object.hparams.update({key:val for key,val in hparams.items() if not key.startswith('__')})
            # set hyperparameters
            object.__dict__.update(hparams)
    

