import torch

class Stats:
    def __init__(self, s0, backward_policy, total_flow, env):
        self.backward_policy = backward_policy
        self.total_flow = total_flow
        self.env = env
        self._traj = [s0.view(len(s0), 1, -1)]
        self._fwd_probs = []
        self.rewards = torch.zeros(len(s0))
        self.num_samples = s0.shape[0]
    
    def log(self, s, probs, actions, done):
        had_terminating_action = actions == probs.shape[-1] - 1
        active, just_finished = ~done, ~done
        active[active == True] = ~had_terminating_action
        just_finished[just_finished == True] = had_terminating_action
    
        states = torch.zeros_like(self._traj[-1]).squeeze(1)
        states[active] = s[active]
        self._traj.append(states.view(self.num_samples, 1, -1))
        
        fwd_probs = torch.zeros(self.num_samples, 1)
        fwd_probs[~done] = probs.gather(1, actions.unsqueeze(1))
        self._fwd_probs.append(fwd_probs)
        
        self.rewards[just_finished] = self.env.reward(s[just_finished])
    
    @property
    def traj(self):
        return torch.cat(self._traj, dim=1)[:, :-1, :]
    
    @property
    def fwd_probs(self):
        return torch.cat(self._fwd_probs, dim=1)
          
    @property
    def back_probs(self):
        return self.backward_policy(self.traj)