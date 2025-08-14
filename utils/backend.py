import torch
import random
try:
    import cupy as xp
    if not xp.cuda.is_available():
        print("CUDA not available")
        raise ImportError("CUDA not available")
except (ImportError, ModuleNotFoundError):
    import numpy as xp



def set_seed(seed):
    random.seed(seed)
    xp.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




