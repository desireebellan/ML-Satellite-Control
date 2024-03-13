
from typing import Tuple
from torch import Tensor
from torch.optim import Adam
from copy import deepcopy
from buffers import RecurrentBuffer, LSTMState
from tqdm import tqdm

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import torch

class InertiaEstimator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        state_dim             = hparams.state_dim
        action_dim            = hparams.action_dim
        hidden_dim            = hparams.hidden_dim
        num_layers            = hparams.num_layers
        self.inertia_dim      = hparams.inertia_dim
        self.device           = hparams.device
        self.lr               = 1e-4
        self.train_K          = hparams.train_K
        self.compute_variance = hparams.compute_variance
        self.last_step        = False
        
        dropout = 0.1
        
        self.lstm = nn.LSTM(state_dim + action_dim, hidden_dim, num_layers = num_layers, dropout = dropout)
        self.estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.inertia_dim)
        )
        self.var = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
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
        
        self.loss = nn.MSELoss(reduction = "sum")
        
        self.to(self.device)
        
    def build_lstm_state(self, num_layers, hidden_dim):
        return (
            torch.zeros(num_layers, 1, hidden_dim).to(self.device),
            torch.zeros(num_layers, 1, hidden_dim).to(self.device)
        )
        
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
            return self.estimator(lstm_output), self.var(lstm_output), lstm_states, mask
        
        else: 
            return self.estimator(lstm_output), self.var(lstm_output), lstm_states
    
    def training_step(self) -> Tensor:
        self.train()
        for t in tqdm(range(self.train_K)):
            for batch in self.buffer:
                xt = torch.cat((batch.actions, batch.observations), dim = -1).float().to(self.device)
                
                estimates, log_vars, _, mask= self.forward(xt, (batch.lstm_states.hidden, batch.lstm_states.cell), batch.sequence_lengts)
                estimates[:,0] = F.relu(estimates[:,0])
                
                if self.last_step:
                    n_seq = batch.lstm_states.hidden.shape[1]
                    targets = batch.targets.reshape((n_seq, -1, self.inertia_dim)).swapaxes(0, 1)[mask]
                    if self.compute_variance: 
                        loss = torch.exp(-log_vars) * (estimates - targets).square().sum(dim = -1, keepdim = True) + log_vars
                        loss = loss.mean(dim = 0)
                    else : 
                        loss = (estimates - targets).square().sum(dim = -1).mean(dim = 0)
                else:     
                    if self.compute_variance: 
                        loss = torch.exp(-log_vars) * (estimates - batch.targets[mask]).square().sum(dim = -1, keepdim = True) + log_vars
                        loss = loss.mean(dim = 0)
                    else : 
                        loss = (estimates - batch.targets[mask]).square().sum(dim = -1).mean(dim = 0)

                self.optimizer.zero_grad()
                loss.float().backward()
                self.optimizer.step()
                
        self.buffer.reset()
            
        return loss.detach()

    def predict(self, observation, action, is_terminal, target = None) -> [Tensor, Tensor]:
        self.eval()
        xt = torch.cat((torch.from_numpy(action), torch.from_numpy(observation)), dim = -1).float().to(self.device)
        
        with torch.no_grad():
            estimate, log_var, lstm_state = self.forward(xt, self.lstm_state)
            estimate[:,0] = F.relu(estimate[:,0])
            #print(estimate, target)
        
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
        else: self.lstm_state = lstm_state
        
        return estimate.squeeze(), torch.exp(log_var).sqrt()
        
    def configure_optimizers(self) -> None:
        self.optimizer = Adam(self.parameters(), lr = self.lr)
        
