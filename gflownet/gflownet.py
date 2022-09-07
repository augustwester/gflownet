import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Categorical
from .stats import Stats

class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env):
        """
        Initializes a GFlowNet using the specified forward and backward policies
        acting over an environment, i.e. a state space and a reward function.
        
        Args:
            forward_policy: A policy network taking as input a state and
            outputting a vector of probabilities over actions
            
            backward_policy: A policy network (or fixed function) taking as
            input a state and outputting a vector of probabilities over the
            actions which led to that state
            
            env: An environment defining a state space and an associated reward
            function
        """
        super().__init__()
        self.total_flow = Parameter(torch.ones(1))
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.env = env
    
    def mask_and_normalize(self, s, probs):
        """
        Masks a vector of action probabilities to avoid illegal actions (i.e.
        actions that lead outside the state space).
        
        Args:
            s: An NxD matrix representing N states
            
            probs: An NxA matrix of action probabilities
        """
        probs = self.env.mask(s) * probs
        return probs / probs.sum(1).unsqueeze(1)
    
    def forward_probs(self, s):
        """
        Returns a vector of probabilities over actions in a given state.
        
        Args:
            s: An NxD matrix representing N states
        """
        probs = self.forward_policy(s)
        return self.mask_and_normalize(s, probs)
    
    def sample_states(self, s0, return_stats=False):
        """
        Samples and returns a collection of final states from the GFlowNet.
        
        Args:
            s0: An NxD matrix of initial states
            
            return_stats: Return an object containing information about the
            sampling process (e.g. the trajectory of each sample, the forward
            and backward probabilities, the actions taken, etc.)
        """
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
        
        return (s, stats) if return_stats else s
    
    def evaluate_trajectories(self, s, traj, actions):
        """
        Returns the forward probabilities, backward probabilities, and rewards
        of a collection of final states (as well as the trajectories and actions
        which led to them) according to the GFlowNet. This is useful in an
        offline learning context where samples drawn according to another policy
        (e.g. a random one) are used to train the model.
        
        Args:
            s: The set of final states for which rewards should be calculated
            
            traj: The trajectory of each samples in s
            
            actions: The actions that produced the trajectories in traj
        """
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