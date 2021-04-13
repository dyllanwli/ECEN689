
import torch
from utils import to_device


def estimate_net_grad(rewards, masks, values, gamma, tau, device):
    # these computations would be performed on CPU
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)

    """ ESTIMATE RETURNS"""
    #
    #
    #
    #

    # standardize returns for algorithmic stability
    returns = (returns - returns.mean()) / returns.std()

    """ ESTIMATE NET GRADIENT"""
    #
    #
    #
    #
    # Roughly normalize the gradient
    grad = grad / (torch.norm(grad) + 1e-8)

    returns = to_device(device, grad)
    return returns

