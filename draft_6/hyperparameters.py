
from typing import Dict
class HyperParams(Dict):
    
    max_iter        : int     = int(1e7)
    max_traj_size   : int     = 2000
    update_timestep : int     = 12000
    save_model_freq : int     = int(1e4) 
    
    buffer_size     : int     = 12000  
    hidden_dim      : int     = 128
    inertia_dim     : int     = 4
    lr              : float   = 1e-3
    inertia_buffer_size : int = 24000
    sequence_length : int     = 2000
    train_K         : int     = 10
    max_patience    : int     = 20
    
    reward          : str     = None
    max_action      : str     = 1
    
    dt              : float   = 0.01
    threshold       : float   = 0.01
    
    device          : str     = 'auto'
    seed            : int     = 0
    