from abc import ABC, abstractmethod

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