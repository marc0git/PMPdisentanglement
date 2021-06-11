import torch
import numpy as np
import os
import os
from pathlib import Path
import dotenv

def get_env(env_name: str) -> str:
    """
    Read an environment variable.
    Raises errors if it is not defined or empty.
    :param env_name: the name of the environment variable
    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        raise KeyError(f"{env_name} not defined")
    env_value: str = os.environ[env_name]
    if not env_value:
        raise ValueError(f"{env_name} has yet to be configured")
    return env_value


def load_envs(env_file: str = ".env") -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.
    It is possible to define all the system specific variables in the `env_file`.
    :param env_file: the file that defines the environment variables to use
    """
    assert os.path.isfile(env_file), f"{env_file}"
    env_path = Path('.') / '.env'
    dotenv.load_dotenv(dotenv_path=env_path)




def pairwise_dists(x):
    #D=F.pdist(x)
    D = torch.sum( (x[:,None,:]-x[None,:,:])**2,-1)
    return D






def tv_loss(x, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(x[:,:,:-1] - x[:,:,1:], 2))
    h_variance = torch.sum(torch.pow(x[:,:-1,:] - x[:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss

def full_chamfer_and_Hausdorff_over_sets(U, V, mean_over_batch=True):
    B, nu, d = U.shape
    B, nv, d = V.shape
    ut = U.unsqueeze(2).expand(B,nu,nv,d)
    vt = V.unsqueeze(1).expand(B,nu,nv,d)
    dists = torch.pow( (ut - vt), 2).sum(dim=3)
    init1 = dists.min(dim=1)[0] #.sqrt()
    init2 = dists.min(dim=2)[0] #.sqrt()
    if mean_over_batch:
        c1 = init1.mean()
        c2 = init2.mean()
        h1 = init1.max(dim=1)[0].mean()
        h2 = init2.max(dim=1)[0].mean()
    else:
        c1 = init1.mean(dim=1)
        c2 = init2.mean(dim=1)
        h1 = init1.max(dim=1)[0] #.mean(dim=1)
        h2 = init2.max(dim=1)[0] #.mean(dim=1)
    return c1, c2, h1, h2



def local_dist_over_sets(U, V, mean_over_batch=True):
    B, nu, d = U.shape
    B, nv, d = V.shape
    ut = U.unsqueeze(2).expand(B,nu,nv,d)
    vt = V.unsqueeze(1).expand(B,nu,nv,d)
    dists = torch.pow( (ut - vt), 2).sum(dim=3)
    init1 = dists.argmin(dim=1)[0] #.sqrt()
    init2 = dists.argmin(dim=2)[0] #.sqrt()
    if mean_over_batch:
        c1 = init1.mean()
        c2 = init2.mean()
        h1 = init1.max(dim=1)[0].mean()
        h2 = init2.max(dim=1)[0].mean()
    else:
        c1 = init1.mean(dim=1)
        c2 = init2.mean(dim=1)
        h1 = init1.max(dim=1)[0] #.mean(dim=1)
        h2 = init2.max(dim=1)[0] #.mean(dim=1)
    return c1, c2, h1, h2







def custom_cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res







def local_dist(x,k):
    D = torch.cdist(x, x)
    values, idx = torch.topk(D, dim=2, largest=False, k=k)
    return values




