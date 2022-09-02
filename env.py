import torch
from abc import ABC, abstractmethod
from torch.nn.functional import one_hot

class Env(ABC):
    @abstractmethod
    def update(self, s, actions):
        pass
    
    @abstractmethod
    def mask(self, s):
        pass
    
    @abstractmethod
    def reward(self, s):
        pass

class Hypergrid(Env):
    def __init__(self, dim, size, num_actions):
        self.dim = dim
        self.size = size
        self.state_dim = size**dim
        self.num_actions = num_actions
        
    def update(self, s, actions):
        idx = s.argmax(1)
        down, right = actions == 0, actions == 1
        idx[down] = idx[down] + self.size
        idx[right] = idx[right] + 1
        return one_hot(idx, self.state_dim).float()
    
    def mask(self, s):
        mask = torch.ones(len(s), self.num_actions)
        idx = s.argmax(1) + 1
        right_edge = (idx > 0) & (idx % (self.size) == 0)
        bottom_edge = idx > self.size*(self.size-1)
        mask[right_edge, 1] = 0
        mask[bottom_edge, 0] = 0
        return mask
        
    def reward(self, s):
        grid = s.view(len(s), self.size, self.size)
        coord = (grid == 1).nonzero()[:, 1:].view(len(s), self.dim)
        R0, R1, R2 = 1e-2, 0.5, 2
        norm = torch.abs(coord / (self.size-1) - 0.5)
        R1_term = torch.prod(0.25 < norm, dim=1)
        R2_term = torch.prod((0.3 < norm) & (norm < 0.4), dim=1)
        return (R0 + R1*R1_term + R2*R2_term)