from collections import namedtuple, deque
from typing import NamedTuple, Tuple, Callable, Generator
from torch import Tensor
from functools import partial

import numpy as np

import torch

class LSTMState(NamedTuple):
    hidden  : Tensor
    cell    : Tensor
    
class InertiaInput(NamedTuple):
    observations   : Tensor
    actions        : Tensor
    lstm_states    : LSTMState
    episode_starts : Tensor
    targets        : Tensor
    sequence_lengts: np.ndarray = np.array([])
 
 
def pad(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: torch.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Chunk sequences and pad them to have constant dimensions.

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device
    :param tensor: Tensor of shape (batch_size, *tensor_shape)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq, max_length, *tensor_shape)
    """
    # Create sequences given start and end
    seq = [tensor[start : end + 1].to(device) for start, end in zip(seq_start_indices, seq_end_indices)]
    return torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value) 

def pad_and_flatten(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: torch.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad and flatten the sequences of scalar values,
    while keeping the sequence order.
    From (batch_size, 1) to (n_seq, max_length, 1) -> (n_seq * max_length,)

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device (cpu, gpu, ...)
    :param tensor: Tensor of shape (max_length, n_seq, 1)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq * max_length,) aka (padded_batch_size,)
    """
    return pad(seq_start_indices, seq_end_indices, device, tensor, padding_value).flatten() 

def create_sequencers(
    episode_starts: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, Callable, Callable]:
    """
    Create the utility function to chunk data into
    sequences and pad them to create fixed size tensors.

    :param episode_starts: Indices where an episode starts
    :param env_change: Indices where the data collected
        come from a different env (when using multiple env for data collection)
    :param device: PyTorch device
    :return: Indices of the transitions that start a sequence,
        pad and pad_and_flatten utilities tailored for this batch
        (sequence starts and ends indices are fixed)
    """
    # First index is always the beginning of a sequence
    episode_starts[0] = True
    # Retrieve indices of sequence starts
    seq_start_indices = np.where(episode_starts.cpu() == True)[0]  # noqa: E712
    seq_lengths = np.diff(np.concatenate([seq_start_indices, np.array([len(episode_starts)])]))
    # End of sequence are just before sequence starts
    # Last index is also always end of a sequence
    seq_end_indices = np.concatenate([(seq_start_indices - 1)[1:], np.array([len(episode_starts)])])

    # Create padding method for this minibatch
    # to avoid repeating arguments (seq_start_indices, seq_end_indices)
    local_pad = partial(pad, seq_start_indices, seq_end_indices, device)
    local_pad_and_flatten = partial(pad_and_flatten, seq_start_indices, seq_end_indices, device)
    return seq_start_indices, seq_lengths, local_pad, local_pad_and_flatten
    
    

class RecurrentBuffer():
    """
    Rollout buffer that also stores the LSTM cell and hidden states.

    :param buffer_size: Max number of element in the buffer
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    """
    def __init__(self, 
                 observation_size:int, 
                 action_size     :int,
                 hidden_shape    :Tuple[int,...],
                 output_size     :int,
                 buffer_size     :int, 
                 batch_size      :int = None, 
                 device          :str = 'cpu'):
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size if batch_size is not None else buffer_size
        self.device = device
        
        self.obs_size = observation_size
        self.action_size = action_size
        self.hidden_shape = hidden_shape
        self.output_size = output_size
        
        self.reset()
        
    def add(self, 
            observation:Tensor, 
            action:Tensor, 
            lstm_state:LSTMState, 
            episode_start:Tensor, 
            target:Tensor) -> None:
        
        assert self.idx < self.buffer_size
        
        self.observations[self.idx] = torch.from_numpy(observation)
        self.actions[self.idx] = torch.from_numpy(action)
        self.hidden_states[self.idx] = lstm_state.hidden.swapaxes(0,1)
        self.cell_states[self.idx] = lstm_state.hidden.swapaxes(0,1)
        self.episode_starts[self.idx] = torch.tensor([episode_start])
        self.targets[self.idx] = target
        
        self.idx += 1
        
    def reset(self):    
        
        self.observations = torch.empty((self.buffer_size, self.obs_size), dtype = torch.float32, device = self.device)
        self.actions = torch.empty((self.buffer_size, self.action_size), dtype = torch.float32, device = self.device)
        self.hidden_states = torch.empty((self.buffer_size,) + self.hidden_shape, dtype = torch.float32, device = self.device)
        self.cell_states = torch.empty((self.buffer_size,) + self.hidden_shape, dtype = torch.float32, device = self.device)
        self.episode_starts = torch.empty((self.buffer_size, 1), dtype = bool, device = self.device)
        self.targets = torch.empty((self.buffer_size, self.output_size), dtype = torch.float32, device = self.device)
        self.idx = 0
        
                
    def __iter__(self) -> Generator[InertiaInput, None, None]:
        
        assert self.idx == self.buffer_size
        
        self.memory = InertiaInput(
            observations=self.observations,
            actions=self.actions,
            lstm_states=LSTMState(
                hidden=self.hidden_states,
                cell = self.cell_states,),
            episode_starts=self.episode_starts,
            targets=self.targets,
            
        )

        # Sampling strategy that allows any mini batch size but requires
        # more complexity and use of padding
        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = np.random.randint(self.buffer_size)
        indices = np.arange(self.buffer_size)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        start_idx = 0
        while start_idx < self.buffer_size:
            batch_inds = indices[start_idx : start_idx + self.batch_size]
            yield self._get_samples(batch_inds)
            start_idx += self.batch_size

            
    def _get_samples(self, batch_inds: np.ndarray) -> InertiaInput:
        # Retrieve sequence starts and utility function
        self.seq_start_indices, self.seq_lengths, self.pad, self.pad_and_flatten = create_sequencers(self.memory.episode_starts[batch_inds], self.device)
        
        # Number of sequences
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.memory.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        lstm_states = LSTMState(
            # 1. (n_steps, n_layers, dim) -> (batch_size, n_layers, dim)
            # 2. (batch_size, n_layers, dim)  -> (n_seq, n_layers, dim)
            # 3. (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
            hidden = self.memory.lstm_states.hidden[batch_inds][self.seq_start_indices].swapaxes(0, 1).contiguous(),
            cell   = self.memory.lstm_states.cell[batch_inds][self.seq_start_indices].swapaxes(0, 1).contiguous()
        )

        return InertiaInput(
            # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
            observations=self.pad(self.memory.observations[batch_inds]).reshape((padded_batch_size, -1)),
            actions=self.pad(self.memory.actions[batch_inds]).reshape((padded_batch_size, -1)),
            lstm_states=lstm_states,
            episode_starts=self.pad_and_flatten(self.memory.episode_starts[batch_inds]),
            targets = self.pad(self.memory.targets[batch_inds]).reshape((padded_batch_size, -1)).float(),
            sequence_lengts = self.seq_lengths
        )