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