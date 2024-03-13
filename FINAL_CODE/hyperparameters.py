
from typing import Dict
class HyperParams(Dict):
    
    max_iter        : int     = int(1e7)
    max_traj_size   : int     = 2000
    update_timestep : int     = 12000
    save_model_freq : int     = int(1e4) 
    update_payload_freq:int  = 1
    
    buffer_size     : int     = 12000  
    hidden_dim      : int     = 128
    inertia_dim     : int     = 4
    lr              : float   = 1e-3
    inertia_buffer_size : int = 24000
    sequence_length : int     = 2000
    train_K         : int     = 10
    max_patience    : int     = 20
    mode            : str     = "test"
    verbose         : bool    = True
    
    reward          : str     = None
    max_action      : str     = 1
    
    dt              : float   = 0.01
    threshold       : float   = 0.01
    
    device          : str     = 'auto'
    seed            : int     = 0
    
    betas           : tuple   = (0.9, 0.999)
    inertia_reduction : str   = "sum"
    feature_extraction : bool = False
    num_layers      : int     = 1
    bidirectional   : bool    = False
    compute_variance: bool    = False
    dropout         : float   = 0.0
    hw              : bool    = False
    inertia_batch_size:int    = 32
    loss            : str     = "mse"
    reduction       : str     = "sum"
    weight_decay    : int     = 0
    
    embedding_checkpoint:str  = None
    initial_gamma   :float    = 0. 
    gamma_rate      :float    = 1.
    