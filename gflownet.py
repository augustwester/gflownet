import torch
from torch import nn
from torch.distributions import Categorical
from stats import Stats

class GFlowNet(nn.Module):
    def __init__(self,
                 forward_policy,
                 backward_policy,
                 update_fn,
                 reward_fn,
                 mask_fn=None,
                 termination_vector=None) -> None:
        super().__init__()
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.update_fn = update_fn
        self.reward_fn = reward_fn
        self.mask_fn = mask_fn
        self.termination_vector = termination_vector
    
    def mask_and_normalize(self, s, probs) -> torch.Tensor:
        probs = probs if self.mask_fn is None else self.mask_fn(s) * probs
        return probs / probs.sum(1)[..., None]
    
    def forward_probs(self, s, gamma=0) -> torch.Tensor:
        probs = self.forward_policy(s)
        unif = torch.rand_like(probs)
        probs = gamma * unif + (1 - gamma) * probs
        probs = self.mask_and_normalize(s, probs)
        return probs
    
    def sample_states(self, s0, explore=False) -> tuple[torch.Tensor, Stats]:
        s = s0.clone()
        done = torch.BoolTensor([False] * len(s))
        gamma = 0.1 if explore else 0
        stats = Stats(s0, self.backward_policy, self.reward_fn)

        while not done.all():
            probs = self.forward_probs(s[done == False], gamma)
            actions = Categorical(probs).sample()
            s[done == False] = self.update_fn(s[done == False], actions)
            
            terminated = actions == probs.shape[-1] - 1
            stats.log(s, probs, actions, done)
            done[done == False] = terminated
        
        stats.traj = stats.traj[:, :-1, :]
        stats.fwd_prob = stats.fwd_prob[:, :-1]
        stats.rewards = stats.rewards[:, :-1]
        return s, stats