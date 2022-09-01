import torch

def detailed_balance_loss(fwd_probs, back_probs, term_probs, R):
    lhs = (R[:, 0:-1] * fwd_probs * term_probs[:, 1:])
    rhs = (R[:, 1:] * back_probs * term_probs[:, :-1])
    lhs[lhs == 0] = 1
    rhs[rhs == 0] = 1
    loss = torch.log(lhs / rhs)**2
    return loss.mean()