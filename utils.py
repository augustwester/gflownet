import torch

def detailed_balance_loss(fwd_probs, back_probs, term_probs, R):
    lhs = (R[:, 0:-1] * fwd_probs * term_probs[:, 1:])
    rhs = (R[:, 1:] * back_probs * term_probs[:, :-1])
    lhs[lhs == 0] = 1
    rhs[rhs == 0] = 1
    loss = torch.log(lhs / rhs)**2
    return loss.mean()

def trajectory_balance_loss(total_flow, rewards, fwd_probs, back_probs):
    fwd_probs[fwd_probs == 0] = 1
    lhs = total_flow * torch.prod(fwd_probs, dim=1)
    rhs = rewards * torch.prod(back_probs, dim=1)
    loss = torch.log(lhs / rhs)**2
    return loss.mean()