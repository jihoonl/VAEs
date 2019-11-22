"""
GECO optimization util methods
from https://github.com/applied-ai-lab/genesis
"""

import torch
from . import device

def get_ema(new, old, alpha):
    if old is None:
        return new
    return (1.0 - alpha) * new + alpha * old

def geco_beta_update(beta,
                     error_ema,
                     goal,
                     step_size,
                     min_clamp=1e-10,
                     speedup=None):
    # Compute current constraint value and detach because we do not want to
    # back-propagate through error_ema
    constraint = (goal - error_ema).detach()
    # Update beta
    if speedup is not None and constraint.item() > 0.0:
        # Apply a speedup factor to recover more quickly from undershooting
        beta = beta * torch.exp(speedup * step_size * constraint)
    else:
        beta = beta * torch.exp(step_size * constraint)
    # Clamp beta to be larger than minimum value
    if min_clamp is not None:
        clamp = torch.tensor(min_clamp).to(device)
        beta = torch.max(beta, clamp)
    # Detach again just to be safe
    return beta.detach()
