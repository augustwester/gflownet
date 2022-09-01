import torch

class Stats:
    def __init__(self, s0, backward_policy, env):
        self.backward_policy = backward_policy
        self.env = env
        self._traj = [s0.view(len(s0), 1, -1)]
        self._fwd_probs = []
        self._term_probs = []
        self._rewards = [env.reward(s0)]
        self.num_samples = s0.shape[0]
    
    def log(self, s, probs, actions, done):
        had_terminating_action = actions == probs.shape[-1] - 1
        active = ~done
        active[active == True] = ~had_terminating_action
    
        states = self._traj[-1].clone().squeeze(1)
        states[active] = s[active]
        self._traj.append(states.view(self.num_samples, 1, -1))
        
        fwd_probs = torch.zeros(self.num_samples, 1)
        fwd_probs[active] = probs[~had_terminating_action].gather(1, actions[~had_terminating_action].unsqueeze(1))
        self._fwd_probs.append(fwd_probs)
        
        term_probs = torch.zeros(self.num_samples, 1)
        term_probs[~done] = probs[:, -1:]
        self._term_probs.append(term_probs)
        
        rewards = torch.zeros(self.num_samples, 1)
        rewards[active] = self.env.reward(s[active])
        self._rewards.append(rewards)
    
    @property
    def traj(self):
        return torch.cat(self._traj, dim=1)[:, :-1, :]
    
    @property
    def fwd_probs(self):
        return torch.cat(self._fwd_probs, dim=1)[:, :-1]
          
    @property
    def back_probs(self):
        return self.backward_policy(self.traj)
    
    @property
    def term_probs(self):
        return torch.cat(self._term_probs, dim=1)
    
    @property
    def rewards(self):
        return torch.cat(self._rewards, dim=1)[:, :-1]