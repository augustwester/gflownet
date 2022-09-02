import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Categorical
from stats import Stats

class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env):
        super().__init__()
        self.total_flow = Parameter(torch.ones(1))
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.env = env
    
    def mask_and_normalize(self, s, probs):
        probs = self.env.mask(s) * probs
        return probs / probs.sum(1)[..., None]
    
    def forward_probs(self, s, gamma):
        probs = self.forward_policy(s)
        unif = torch.rand_like(probs)
        probs = gamma * unif + (1 - gamma) * probs
        probs = self.mask_and_normalize(s, probs)
        return probs
    
    def sample_states(self, s0, explore=False, return_stats=False):
        s = s0.clone()
        done = torch.BoolTensor([False] * len(s))
        gamma = 0.1 if explore else 0
        stats = Stats(s0, self.backward_policy, self.total_flow, self.env) if return_stats else None

        while not done.all():
            probs = self.forward_probs(s[done == False], gamma)
            actions = Categorical(probs).sample()
            s[done == False] = self.env.update(s[done == False], actions)
            
            if return_stats:
                stats.log(s, probs, actions, done)
                
            terminated = actions == probs.shape[-1] - 1
            done[done == False] = terminated
        
        return s, stats