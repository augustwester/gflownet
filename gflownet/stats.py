import torch

class Stats:
    def __init__(self, s0, backward_policy, total_flow, env):
        """
        Initializes a Stats object to record sampling statistics from a
        GFlowNet (e.g. trajectories, forward and backward probabilities,
        actions, etc.)
        
        Args:
            s0: The initial state of collection of samples
            
            backward_policy: The backward policy used to estimate the backward
            probabilities associated with each sample's trajectory
            
            total_flow: The estimated total flow used by the GFlowNet during
            sampling
            
            env: The environment (i.e. state space and reward function) from
            which samples are drawn
        """
        self.backward_policy = backward_policy
        self.total_flow = total_flow
        self.env = env
        self._traj = [s0.view(len(s0), 1, -1)]
        self._fwd_probs = []
        self._back_probs = None
        self._actions = []
        self.rewards = torch.zeros(len(s0))
        self.num_samples = s0.shape[0]
    
    def log(self, s, probs, actions, done):
        """
        Logs relevant information about each sampling step
        
        Args:
            s: An NxD matrix containing he current state of complete and
            incomplete samples
            
            probs: An NxA matrix containing the forward probabilities output by the
            GFlowNet for the given states
            
            actions: A Nx1 vector containing the actions taken by the GFlowNet
            in the given states
            
            done: An Nx1 Boolean vector indicating which samples are complete
            (True) and which are incomplete (False)
        """
        had_terminating_action = actions == probs.shape[-1] - 1
        active, just_finished = ~done, ~done
        active[active == True] = ~had_terminating_action
        just_finished[just_finished == True] = had_terminating_action
    
        states = torch.zeros(self.num_samples, self.env.state_dim)
        states[active] = s[active]
        self._traj.append(states.view(self.num_samples, 1, -1))
        
        fwd_probs = torch.ones(self.num_samples, 1)
        fwd_probs[~done] = probs.gather(1, actions.unsqueeze(1))
        self._fwd_probs.append(fwd_probs)
        
        _actions = torch.full((self.num_samples, 1), self.env.num_actions - 1)
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
    def actions(self):
        if type(self._actions) is list:
            self._actions = torch.cat(self._actions, dim=1)
        return self._actions
    
    @property
    def back_probs(self):
        if self._back_probs is not None:
            return self._back_probs
        
        s = self.traj[:, 1:, :].reshape(-1, self.env.state_dim)
        prev_s = self.traj[:, :-1, :].reshape(-1, self.env.state_dim)
        actions = self.actions[:, :-1].reshape(-1, 1)
        
        back_probs = self.backward_policy(s) * self.env.mask(prev_s)
        back_probs = back_probs.gather(1, actions)
        
        actions = self.actions[:, :-1].flatten()
        back_probs[actions == self.env.num_actions - 1] = 1
        self._back_probs = back_probs.reshape(self.num_samples, -1)
        
        return self._back_probs