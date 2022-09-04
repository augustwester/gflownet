import torch

def detailed_balance_loss(fwd_probs, back_probs, term_probs, R):
    lhs = (R[:, 0:-1] * fwd_probs * term_probs[:, 1:])
    rhs = (R[:, 1:] * back_probs * term_probs[:, :-1])
    lhs[lhs == 0] = 1
    rhs[rhs == 0] = 1
    loss = torch.log(lhs / rhs)**2
    return loss.mean()

def trajectory_balance_loss(total_flow, rewards, fwd_probs, back_probs):
    """
    Computes the mean trajectory balance loss for a collection of samples. For
    more information, see Bengio et al. (2022): https://arxiv.org/abs/2201.13259
    
    Args:
        total_flow: The estimated total flow used by the GFlowNet when drawing
        the collection of samples for which the loss should be computed
        
        rewards: The rewards associated with the final state of each of the
        samples
        
        fwd_probs: The forward probabilities associated with the trajectory of
        each sample (i.e. the probabilities of the actions actuallt taken in
        each trajectory)
        
        back_probs: The backward probabilities associated with each trajectory
    """
    fwd_probs[fwd_probs == 0] = 1
    lhs = total_flow * torch.prod(fwd_probs, dim=1)
    rhs = rewards * torch.prod(back_probs, dim=1)
    loss = torch.log(lhs / rhs)**2
    return loss.mean()