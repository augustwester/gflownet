import torch
from torch.nn.functional import pad

class Stats:
    def __init__(self, s0, backward_policy, reward_fn) -> None:
        self.backward_policy = backward_policy
        self.reward_fn = reward_fn
        self._back_prob = None
        self.traj = s0.view(len(s0), 1, -1)
        self.fwd_prob = torch.empty(len(s0), 0)
        self.term_prob = torch.empty_like(self.fwd_prob)
        self.rewards = reward_fn(s0)
    
    def log(self, s, probs, actions, done):
        self.traj = torch.cat((self.traj, s.view(len(s), 1, -1)), dim=1)
        self.fwd_prob = pad(self.fwd_prob, (0,1,0,0))
        self.term_prob = pad(self.term_prob, (0,1,0,0))
        
        active = actions != probs.shape[-1] - 1
        inactive = done.clone()
        inactive[~done] = ~active
        
        self.fwd_prob[~inactive, -1:] = probs[active].gather(1, actions[active].unsqueeze(1))
        self.term_prob[~done, -1] = probs[:, -1]
        
        rewards = torch.zeros(len(s), 1)
        rewards[~inactive] = self.reward_fn(s[~inactive])
        self.rewards = torch.cat((self.rewards, rewards), dim=1)
        
    @property
    def back_prob(self):
        if self._back_prob is not None:
            return self._back_prob.clone()
        return self.backward_policy(self.traj)