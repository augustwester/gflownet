import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Categorical
from .stats import Stats

class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env):
        super().__init__()
        self.total_flow = Parameter(torch.ones(1))
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.env = env
    
    def mask_and_normalize(self, s, probs):
        probs = self.env.mask(s) * probs
        return probs / probs.sum(1).unsqueeze(1)
    
    def forward_probs(self, s):
        probs = self.forward_policy(s)
        return self.mask_and_normalize(s, probs)
    
    def sample_states(self, s0, return_stats=False):
        s = s0.clone()
        done = torch.BoolTensor([False] * len(s))
        stats = Stats(s0, self.backward_policy, self.total_flow, self.env) if return_stats else None

        while not done.all():
            probs = self.forward_probs(s[done == False])
            actions = Categorical(probs).sample()
            s[done == False] = self.env.update(s[done == False], actions)
            
            if return_stats:
                stats.log(s, probs, actions, done)
                
            terminated = actions == probs.shape[-1] - 1
            done[done == False] = terminated
        
        return s, stats
    
    def evaluate_trajectories(self, s, traj, actions):
        traj = traj.view(traj.shape[0]*traj.shape[1], -1)
        
        _fwd_probs = self.forward_probs(traj)
        _actions = actions.view(len(traj))
        
        fwd_probs = torch.ones(len(traj), 1)
        fwd_probs[_actions != -1] = _fwd_probs[_actions != -1].gather(1, _actions[_actions != -1].unsqueeze(1))
        fwd_probs = fwd_probs.view(len(s), -1)
        
        back_probs = self.backward_policy(traj)
        back_probs = back_probs.view(len(s), -1)
        
        rewards = self.env.reward(s)
        
        return fwd_probs, back_probs, rewards