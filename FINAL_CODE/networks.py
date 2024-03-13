
from typing import Tuple, Callable, List
from torch import Tensor
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
from buffers import RecurrentBuffer, LSTMState
from tqdm import tqdm
from collections import deque
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import *

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pytorch_lightning as pl

import torch

# Loss functions
def mse(true, pred):
    return (true-pred).square().mean(dim = 1)
class Loss():
    def __init__(self, loss_func:callable, reduction:str = "mean", weights:List[float] = []):
        self.reduction = reduction 
        self.loss_func = loss_func
        self.weights = torch.tensor(weights) if len(weights) >= 1 else torch.ones(1)

    def forward(self, true, pred):
        if self.weights.device != true.device : self.weights = self.weights.to(true.device)
        weights = self.weights
        loss = self.loss_func(true, pred)
        loss = loss.mul(weights.expand(loss.shape))
        '''        if self.reduction == "sum":
            return loss.sum(0)
        elif self.reduction == "mean":
            return loss.mean(0)'''
        return loss
        
    def __call__(self, true, pred):
        return self.forward(true, pred)
        
def MSE(true, pred):
    return (true-pred).square().mean(1)

def NMSE(true, pred):
   return (true-pred).div(true).square().mean(1)
    
def RMSE(true, pred, log_var = 0):
    return (true-pred).square().mean(1).sqrt()

def RRMSE(true, pred, log_var = 0):
    num = (true-pred).square().mean(1)
    den = true.square().sum(1)
    return num.div(den).sqrt()

def RMSRE(true, pred, log_var = 0):
    return (true-pred).div(true).square().mean(1).sqrt()

# Networks

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, dropout = 0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x:Tensor)->Tensor:
        return self.model(x)
    
class LSTM_HWC(nn.Module):
    def __init__(self, input_size, hidden_size, layers = 1, dropout = 0.0, bias = 0.0, bidirectional = True, functional = 'relu'):
        super(LSTM_HWC,self).__init__()

        self.layers           = layers
        self.bidirectional    = bidirectional
        self.input_size       = input_size
        self.D                = 2 if bidirectional else 1
        lstm = []
        transform = []
        self.scale = nn.Linear(input_size, hidden_size)
        for _ in range(layers):
            lstm.append(nn.LSTM(hidden_size, hidden_size, bidirectional=bidirectional))
            output_size = hidden_size if not bidirectional else hidden_size*2
            transform.append(nn.Linear(hidden_size, output_size))
        self.embed = nn.Linear(hidden_size*2, hidden_size) if bidirectional else nn.Identity()
        
        self.func = nn.ModuleList(lstm)
        self.transform = nn.ModuleList(transform)
        
        # self.scale_bilstm = nn.ModuleList([nn.Linear(size*2, size) for _ in range(layers)])
        
        for layer in self.transform:
            layer.bias.data.fill_(bias)
            nn.init.xavier_uniform_(layer.weight)

        self.dropout = nn.Dropout(dropout)
        if functional == 'relu':
            self.h_functional = nn.ReLU()
        elif functional == None:
            self.h_functional = nn.Identity()

    def forward(self, x:torch, lstm_state):
        h, c = lstm_state
        x = self.scale(x)
        for i in range(0, self.layers, self.D):
            T = torch.sigmoid(self.transform[i](x))
            C = 1 - T
            H, (hh, ch) = self.func[i](x, (h[i:i+self.D,:,:].clone(), c[i:i+self.D,:,:].clone()))
            h[i:i+self.D,:,:], c[i:i+self.D,:,:] = hh, ch
            H = self.h_functional(H)
            x = torch.cat([x, x], dim=-1) if self.bidirectional else x
            x = H*T + x*C
            x = self.dropout(x) 
            x = self.embed(x)
        return x, (h,c)

'''class InertiaEstimator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        self.feature_extraction = hparams.feature_extraction
        state_dim             = hparams.state_dim
        action_dim            = hparams.action_dim
        hidden_dim            = hparams.hidden_dim
        num_layers            = hparams.num_layers
        bidirectional         = hparams.bidirectional
        self.inertia_dim      = hparams.inertia_dim
        self.device           = hparams.device
        self.lr               = hparams.lr
        self.betas            = hparams.betas
        self.eps              = 1e-08
        self.weight_decay     = 0
        self.train_K          = hparams.train_K
        self.compute_variance = hparams.compute_variance
        self.len_history      = 100
        self.last_step        = False
        
        dropout               = hparams.dropout
        
        
        if self.feature_extraction:
            input_dims = (state_dim, action_dim)
            output_dims = (int(hidden_dim/2), int(hidden_dim/2))
            self.feature_extractor = nn.ModuleList(
                [FeatureExtractor(input_dim, output_dim) \
                    for (input_dim, output_dim) in zip(input_dims, output_dims)]
                )   
        
        if hparams.hw:
            self.lstm = LSTM_HWC(input_size = state_dim + action_dim, hidden_size = hidden_dim, 
                                 layers=num_layers, dropout=dropout, bidirectional=bidirectional, functional=None)
        else:
            self.lstm = nn.LSTM(state_dim + action_dim, hidden_dim, num_layers = num_layers, dropout = dropout)
        self.estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            #nn.Tanh(),
            nn.Linear(hidden_dim, self.inertia_dim)
        )
    
        self.lstm_state     = self.build_lstm_state(num_layers=num_layers, hidden_dim=hidden_dim)
        self.episode_start  = 1
        self.init_weights()
        
        self.configure_optimizers()
        
        self.buffer = RecurrentBuffer(observation_size=state_dim, 
                                      action_size=action_dim, 
                                      hidden_shape=(num_layers, hidden_dim), 
                                      output_size=self.inertia_dim, 
                                      buffer_size=hparams.inertia_buffer_size,
                                      batch_size=hparams.inertia_batch_size,
                                      device=self.device
                                      )
        
        self.current_history = {'output': deque(maxlen = self.len_history), 
                                'current_target': torch.zeros(self.inertia_dim), 
                                'current_sequence': torch.empty(1, self.state_dim + self.action_dim)}
        
        if hparams.loss == "mse" : loss = MSE 
        elif hparams.loss == "nmse" : loss = NMSE 
        else : raise Exception("the loss function is not recognized")
        
        self.loss = Loss(loss_func=loss, reduction=hparams.reduction)
        
        self.to(self.device)
        
    def build_lstm_state(self, num_layers, hidden_dim):
        return (
            torch.zeros(num_layers, 1, hidden_dim).to(self.device),
            torch.zeros(num_layers, 1, hidden_dim).to(self.device)
        )
      
    # controllare   
    def init_weights(self):
        for name, module in self.named_children():
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name: init.orthogonal_(param)
                    else : init.constant_(param, 0.0)
            elif isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.0)
        
    def forward(self, features:Tensor,lstm_states:Tuple[Tensor, ...], seq_lengths = None) -> [Tensor, Tensor, Tuple[Tensor, ...]]:
        # (sequence length, batch size, features dim)
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        n_seq = lstm_states[0].shape[1]
        features_sequence = features.reshape((n_seq, -1, self.lstm.input_size)).swapaxes(0, 1)
        
        lstm_output, lstm_states = self.lstm(features_sequence, lstm_states)

        # Sequence to batch
        if seq_lengths is not None:
            if self.last_step:
                mask = torch.zeros((lstm_output.shape[0], lstm_output.shape[1]), dtype = lstm_output.dtype, device = lstm_output.device)
                mask[(seq_lengths - 1, torch.arange(lstm_output.shape[1]))] = 1
                mask = mask.bool()
                lstm_output = lstm_output[mask]
            else:
                mask = torch.zeros((lstm_output.shape[0] + 1, lstm_output.shape[1]), dtype = lstm_output.dtype, device = lstm_output.device)
                mask[(seq_lengths, torch.arange(lstm_output.shape[1]))] = 1
                mask = ~ mask.cumsum(dim = 0)[:-1,:].bool().swapaxes(1,0).flatten()
                lstm_output = lstm_output.swapaxes(1,0).flatten(start_dim = 0, end_dim = 1)[mask]
            return self.estimator(lstm_output), lstm_states, mask
        
        else: 
            return self.estimator(lstm_output), lstm_states
    
    def training_step(self) -> Tensor:
        for t in tqdm(range(self.train_K)):
            for batch in self.buffer:
                observations, actions = batch.observations, batch.actions
                if self.feature_extraction:
                    for i, x in enumerate([observations, actions]):
                        x = self.feature_extractor[i](x)
                xt = torch.cat((batch.observations, batch.actions), dim = -1).float().to(self.device)
                
                estimates,  _, mask= self.forward(xt, (batch.lstm_states.hidden, batch.lstm_states.cell), batch.sequence_lengts)
                estimates[:,0] = F.relu(estimates[:,0])
                
                if self.last_step:
                    n_seq = batch.lstm_states.hidden.shape[1]
                    targets = batch.targets.reshape((n_seq, -1, self.inertia_dim)).swapaxes(0, 1)[mask]
                    loss = self.loss(targets, estimates)        
                else:     
                    loss = self.loss(batch.targets[mask], estimates)

                self.optimizer.zero_grad()
                loss.float().backward()
                self.optimizer.step()
                
        self.buffer.reset()
            
        return loss.detach()

    def predict(self, observation, action, is_terminal, target = None) -> [Tensor, Tensor]:
        observation, action = torch.from_numpy(observation).float().to(self.device), \
            torch.from_numpy(action).float().to(self.device)
        
        with torch.no_grad():
            if self.feature_extraction:
                for i, x in enumerate([observation, action]):
                    x = self.feature_extractor[i](x)
            xt = torch.cat((observation, action), dim = -1).float().to(self.device)
            estimate, lstm_state = self.forward(xt, self.lstm_state)
            estimate[:,0] = F.relu(estimate[:,0])
            self.current_history["output"].append(estimate)
        
        if target is not None : self.buffer.add(
            observation=observation, 
            action=action, 
            lstm_state=LSTMState(
                hidden = deepcopy(self.lstm_state[0]),
                cell   = deepcopy(self.lstm_state[1])), 
            episode_start=self.episode_start,
            target=target)
        
        self.episode_start = 0
        if is_terminal : 
            self.lstm_state = self.build_lstm_state(num_layers=self.lstm.num_layers, hidden_dim=self.lstm.hidden_size)
            self.episode_start = 1   
            self.current_history["output"].clear()
        else: self.lstm_state = lstm_state
        
        if self.compute_variance and len(self.current_history["output"]) >= self.len_history: 
            history = torch.stack(list(self.current_history["output"]), dim = 0)
            var = torch.var(history, dim = 0).div(torch.mean(history, dim = 0))
            beta = torch.ones(self.inertia_dim).to(self.device).div(var)
        else: beta = torch.full((1, self.inertia_dim), fill_value = 1e-5)
        
        return estimate.squeeze(), beta
        
    def configure_optimizers(self) -> None:
        self.optimizer = Adam(self.parameters(), lr = self.lr, betas = self.betas, eps = self.eps, weight_decay = self.weight_decay)
        '''
        
class InertiaEstimator(pl.LightningModule):
    def __init__(self, hparams):
        super(InertiaEstimator, self).__init__()
        update_hparams(self, hparams)
                        
        self.save_hyperparameters()

        self.eps              = 1e-08
        self.weight_decay     = 0
        self.len_history      = 20
        self.last_step        = False
        self.max_traj_size    = 801
        self.batch_size       = 128
        
        self.state_dim = 13
        self.action_dim = 3
              
        if self.feature_extraction:
            input_dims = (self.state_dim, self.action_dim)
            output_dims = (int(self.hidden_dim/2), int(self.hidden_dim/2))
            self.feature_extractor = nn.ModuleList(
                [FeatureExtractor(input_dim, output_dim) \
                    for (input_dim, output_dim) in zip(input_dims, output_dims)]
                )   
        
        if self.hw:
            self.lstm = LSTM_HWC(input_size = self.state_dim + self.action_dim, hidden_size = self.hidden_dim, 
                                 layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional, functional=None)
        else:
            self.lstm = nn.LSTM(self.state_dim + self.action_dim, self.hidden_dim, num_layers = self.num_layers, dropout = self.dropout, bidirectional=self.bidirectional)
        lstm_output_dim = self.hidden_dim if self.hw or not self.bidirectional else self.hidden_dim*2
        self.estimator = nn.Sequential(
            nn.Linear(lstm_output_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
            #nn.Tanh(),
            nn.Linear(self.hidden_dim, self.inertia_dim)
        )
    
        self.lstm_state     = self.build_lstm_state(num_layers=self.num_layers, hidden_dim=self.hidden_dim, bidirectional=self.bidirectional)
        self.episode_start  = 1
        self.init_weights()
        
        '''self.buffer = RecurrentBuffer(observation_size=self.state_dim, 
                                      action_size=self.action_dim, 
                                      hidden_shape=(self.num_layers, self.hidden_dim), 
                                      output_size=self.inertia_dim, 
                                      buffer_size=hparams.inertia_buffer_size,
                                      batch_size=hparams.inertia_batch_size,
                                      device=self.device
                                      )'''
        
        if self.loss == "mse" : loss = MSE 
        elif self.loss == "nmse" : loss = NMSE 
        else : raise Exception("the loss function is not recognized")
 
        self.loss_func = Loss(loss_func=loss, reduction=self.reduction)
        if self.is_train_inertia : self.optimizer_dict = self.configure_optimizers()
        
    def setup(self, stage:str=None):
        if stage == 'predict':
            self.build_lstm_state(num_layers=self.num_layers, hidden_dim=self.hidden_dim, bidirectional=self.bidirectional, batch_size=self.batch_size)
            self.current_history = {'input':{'observations': [torch.zeros(self.max_traj_size, self.state_dim, device = self.device)], 
                                            'actions': [torch.zeros(self.max_traj_size, self.action_dim, device = self.device)]}, 
                                    'output': deque(maxlen = self.len_history), 
                                    'target': [torch.zeros(self.inertia_dim, device = self.device)], 
                                    'length': [torch.zeros(1)], 
                                    'step'  : 0}     
        
    def build_lstm_state(self, num_layers, hidden_dim, bidirectional:bool=False, batch_size:int=1):
        D = 2 if bidirectional else 1
        return (
            torch.zeros(num_layers*D, batch_size, hidden_dim).to(self.device),
            torch.zeros(num_layers*D, batch_size, hidden_dim).to(self.device)
        )
      
    # controllare   
    def init_weights(self):
        for name, module in self.named_children():
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name: init.orthogonal_(param)
                    else : init.constant_(param, 0.0)
            elif isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.0)
        
    def forward(self, features:Tensor, lstm_states:Tuple[Tensor, ...] = None, seq_lengths = None) -> [Tensor, Tensor, Tuple[Tensor, ...]]:
        if lstm_states is None : lstm_states = \
            self.build_lstm_state(num_layers=self.num_layers, batch_size=features.shape[1], bidirectional = self.bidirectional, hidden_dim=self.hidden_dim)
        
        lstm_output, lstm_states = self.lstm(features, lstm_states)

        # Sequence to batch
        if seq_lengths is not None:
            if self.last_step:
                mask = torch.zeros((lstm_output.shape[0], lstm_output.shape[1]), dtype = lstm_output.dtype, device = lstm_output.device)
                mask[(seq_lengths - 1, torch.arange(lstm_output.shape[1]))] = 1
                mask = mask.bool()
                lstm_output = lstm_output[mask]
            else:
                mask = torch.zeros((lstm_output.shape[0] + 1, lstm_output.shape[1]), dtype = lstm_output.dtype, device = lstm_output.device)
                mask[(seq_lengths, torch.arange(lstm_output.shape[1]))] = 1
                mask = ~ mask.cumsum(dim = 0)[:-1,:].bool().swapaxes(1,0).flatten()
                lstm_output = lstm_output.swapaxes(1,0).flatten(start_dim = 0, end_dim = 1)[mask]
            return self.estimator(lstm_output), lstm_states, mask
        
        else: 
            return self.estimator(lstm_output), lstm_states
        
    def training_step(self, batch, batch_idx):
        input = batch['input'].swapaxes(1,0)
        output = batch['output']
        
        observation, action = input[:,:,:self.state_dim], input[:,:,self.state_dim:]
        
        if self.feature_extraction:
                for i, x in enumerate([observation, action]):
                    x = self.feature_extractor[i](x)
                    
        xt = torch.cat((observation, action), dim = -1).float().to(self.device)
        estimate, _ = self.forward(xt)
        estimate[:,0] = F.relu(estimate[:,0])
        #self.current_history["output"].append(estimate)
        
        loss = self.loss_func(output, estimate)
        error = (torch.abs((estimate - output).div(output))).mean(1)
        for i in range(output.shape[-1]):
            self.log('train_loss' + str(i+1), loss[i].mean())
            self.log('train_error' + str(i+1), error[i].mean())
            
        self.log('train_loss', loss.mean(0).sum(), prog_bar=True)
             
        #return losses
        return loss.mean(0).sum()
    
    def validation_step(self, batch, batch_idx):
        input = batch['input'].swapaxes(1,0)
        output = batch['output']
        
        if len(input.shape) < 3 : 
            input = input.unsqueeze(1)
            output = output.unsqueeze(0)
        
        output = output.unsqueeze(0)
        
        observation, action = input[:,:,:self.state_dim], input[:,:,self.state_dim:]
        
        if self.feature_extraction:
                for i, x in enumerate([observation, action]):
                    x = self.feature_extractor[i](x)
                    
        xt = torch.cat((observation, action), dim = -1).float().to(self.device)
        estimate, _ = self.forward(xt)
        estimate[:,0] = F.relu(estimate[:,0])
        
        loss = self.loss_func(output, estimate)
        error = torch.abs((output-estimate).div(output)).mean(1)
        for i in range(output.shape[-1]):
            self.log('valid_loss' + str(i+1), loss[:,i].mean())
            self.log('valid_error' + str(i+1), error[:,i].mean())      
        self.log('valid_loss', loss.mean(), prog_bar=True)
        self.log('valid_error', error.mean(), prog_bar=True)  
    
    def test_step(self, batch, batch_idx):
        input = batch['input'].swapaxes(1,0)
        output = batch['output']
        
        if len(input.shape) < 3 : 
            input = input.unsqueeze(1)
            output = output.unsqueeze(0)
        
        output = output.unsqueeze(0)
        
        observation, action = input[:,:,:self.state_dim], input[:,:,self.state_dim:]
        
        if self.feature_extraction:
                for i, x in enumerate([observation, action]):
                    x = self.feature_extractor[i](x)
                    
        xt = torch.cat((observation, action), dim = -1).float().to(self.device)
        estimate, _ = self.forward(xt)
        estimate[:,0] = F.relu(estimate[:,0])
        error = torch.abs((output-estimate).div(output)).mean(1)
        
        for i in range(output.shape[-1]):
            self.log('test_error'+str(i+1), error[:,i].mean())  
            
    def predict(self, observation, action, is_terminal=False, target=None):
        #observation, action = torch.from_numpy(observation).float().to(self.device), \
        #    torch.from_numpy(action).float().to(self.device)
        # PROVVISORIO
        observation = torch.from_numpy(observation).float()
        observation, action = observation[:self.state_dim], observation[self.state_dim:]
        
        with torch.no_grad():
            if self.feature_extraction:
                for i, x in enumerate([observation, action]):
                    x = self.feature_extractor[i](x)
            xt = torch.cat((observation, action), dim = -1).float().to(self.device)
            xt = xt.reshape((1,1,-1)).to(self.device)
            self.lstm_state = self.lstm_state[0].to(self.device), self.lstm_state[1].to(self.device)
            estimate, lstm_state = self.forward(xt, self.lstm_state)
            estimate[:,0] = F.relu(estimate[:,0])
            self.current_history["output"].append(estimate)
        
        if target is not None : 
            self.current_history["input"]["observations"][-1][self.current_history["step"]] = observation
            self.current_history["input"]["actions"][-1][self.current_history["step"]] = action
            self.current_history["target"][-1] = target
            self.current_history["length"][-1] += 1
            
        if is_terminal:
            self.lstm_state = self.build_lstm_state(num_layers=self.num_layers, hidden_dim=self.hidden_dim, bidirectional=self.bidirectional)
            self.current_history["output"].clear()
            self.current_history["input"]["observations"].append(torch.zeros(self.max_traj_size, self.state_dim, device = self.device))
            self.current_history["input"]["actions"].append(torch.zeros(self.max_traj_size, self.action_dim, device = self.device))
            self.current_history["target"].append(torch.zeros(self.inertia_dim, device = self.device))
            self.current_history["length"].append(torch.zeros(1))
            self.current_history["step"] = 0
        else: 
            self.lstm_state = lstm_state
            self.current_history["step"] += 1

        if self.compute_variance and len(self.current_history["output"]) >= self.len_history: 
            history = torch.stack(list(self.current_history["output"]), dim = 0)
            mean = torch.mean(history, dim=0).flatten()
            # avoid nan 
            mean[(mean == 0.0).nonzero()] = 1e-8
            var = torch.var(history, dim = 0).flatten()
            var = var.div(mean)
            var[(var==0.0).nonzero()] = 1e-3
            beta = torch.ones(self.inertia_dim).to(self.device).div(var)
        else: beta = torch.full((1, self.inertia_dim), fill_value = 1e-5)
        
        return estimate.squeeze(), beta
    
    def online_training_phase(self):
        observations = torch.stack(self.current_history["input"]["observations"][:-1], dim=0)
        actions = torch.stack(self.current_history["input"]["actions"][:-1], dim=0)
        targets = torch.stack(self.current_history["target"][:-1], dim=0)
        seq_lengths = torch.stack(self.current_history["length"][:-1]).int()
        for k in tqdm(range(self.train_K)):
            for index in BatchSampler(SubsetRandomSampler(range(targets.shape[0])), self.batch_size, False): 
                observation, action, target, seq_length = observations[index], actions[index], targets[index], seq_lengths[index]
                observation, action = observation.swapaxes(0,1), action.swapaxes(0,1)
                if self.feature_extraction:
                    for i, x in enumerate([observation, action]):
                        x = self.feature_extractor[i](x)
                xt = torch.cat((observation, action), dim = -1).float().to(self.device)
                estimates, _, mask = self.forward(xt, seq_lengths=seq_length)
                estimates[:,0] = F.relu(estimates[:,0])
                n_seq = xt.shape[0]
                
                if self.last_step:
                    target = target.expand((n_seq, -1, self.inertia_dim)).flatten(start_dim=0, end_dim=1)[mask]
                    loss = self.loss_func(target, estimates)        
                else:     
                    target = target.expand((n_seq, -1, self.inertia_dim)).flatten(start_dim=0, end_dim=1)[mask]
                    loss = self.loss_func(target[mask], estimates)
                self.optimizer_dict["optimizer"].zero_grad()
                loss.sum(dim = -1).mean().float().backward()
                self.optimizer_dict["optimizer"].step()
            #self.optimizer_dict["scheduler"].step()
        return loss.mean(dim = 0) 
            

    def configure_optimizers(self):
        if self.optimizer_class == "adam":
            optimizer = Adam(self.parameters(), lr = self.hparams['lr'], weight_decay = self.hparams['weight_decay'])
        if self.optimizer_class == "sgd":
            optimizer = SGD(self.parameters(), lr = self.hparams['lr'], momentum = self.hparams['momentum'], weight_decay=self.hparams['wd'])     
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": "train_loss"}}
        
    '''
    class RecurrentIdentificationActorCritcPolicy(RecurrentActorCriticPolicy):
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_critic_lstm: bool = False,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.lstm_output_dim = lstm_hidden_size
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_lstm_layers,
            shared_critic_lstm,
            enable_critic_lstm,
            lstm_kwargs
        )
        ######
        if not shared_lstm:
            self.lstm_estimator = nn.LSTM(
                self.features_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )
    '''   
        
