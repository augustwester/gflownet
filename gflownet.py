import torch
from torch import nn
from torch.distributions import Categorical
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
    
    def mask_and_normalize(self, s, probs):
        probs = probs if self.mask_fn is None else self.mask_fn(s) * probs
        return probs / probs.sum(1)[..., None]
    
    def forward_probs(self, s, gamma=0):
        probs = self.forward_policy(s)
        unif = torch.rand_like(probs)
        probs = gamma * unif + (1 - gamma) * probs
        probs = self.mask_and_normalize(s, probs)
        return probs
    
    def sample_states(self, s0, explore=False):
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
    
    def sample_backward_trajectories(self, s):
        traj = s[..., None]
        done = torch.BoolTensor([False] * len(s))
        done[traj[..., -1].sum(-1) == 0] = True
        
        while not done.all():
            probs = self.backward_probs(traj[done == False, :, -1])
            samples = Categorical(probs).sample()
            steps = torch.zeros(traj[done == False].shape[:2])
            steps[samples == 0] = torch.Tensor([0, -1])
            steps[samples == 1] = torch.Tensor([-1, 0])
            traj = torch.cat((traj, traj[..., -1:]), dim=-1)
            traj[done == False, :, -1] += steps
            done[traj[..., -1].sum(-1) == 0] = True
        
        return traj