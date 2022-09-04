import torch

class Stats:
    def __init__(self, s0, backward_policy, total_flow, env):
        self.backward_policy = backward_policy
        self.total_flow = total_flow
        self.env = env
        self._traj = [s0.view(len(s0), 1, -1)]
        self._fwd_probs = []
        self._back_probs = []
        self._actions = []
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
        
        fwd_probs = torch.ones(self.num_samples, 1)
        fwd_probs[~done] = probs.gather(1, actions.unsqueeze(1))
        self._fwd_probs.append(fwd_probs)
        
        back_probs = torch.ones(self.num_samples, 1)
        back_probs[~done] = self.backward_policy(s[~done])
        self._back_probs.append(back_probs)
        
        _actions = -torch.ones(self.num_samples, 1).long()
        _actions[~done] = actions.unsqueeze(1)
        self._actions.append(_actions)
        
        self.rewards[just_finished] = self.env.reward(s[just_finished])
    
    @property
    def traj(self):
        if type(self._traj) is list:
            self._traj = torch.cat(self._traj, dim=1)[:, :-1, :]
        return self._traj
    
    @property
    def fwd_probs(self):
        if type(self._fwd_probs) is list:
            self._fwd_probs = torch.cat(self._fwd_probs, dim=1)
        return self._fwd_probs
    
    @property
    def back_probs(self):
        if type(self._back_probs) is list:
            self._back_probs = torch.cat(self._back_probs, dim=1)
        return self._back_probs
    
    @property
    def actions(self):
        if type(self._actions) is list:
            self._actions = torch.cat(self._actions, dim=1)
        return self._actions