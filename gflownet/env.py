from abc import ABC, abstractmethod

class Env(ABC):
    """
    Abstract base class defining the signatures of the required functions to be
    implemented in a GFlowNet environment.
    """
    @abstractmethod
    def update(self, s, actions):
        """
        Takes as input state-action pairs and returns the resulting states.
        
        Args:
            s: An NxD matrix of state vectors
            
            actions: An Nx1 vector of actions
        """
        pass
    
    @abstractmethod
    def mask(self, s):
        """
        Defines a mask to disallow certain actions given certain states.
        
        Args:
            s: An NxD matrix of state vectors
        """
        pass
    
    @abstractmethod
    def reward(self, s):
        """
        Defines a reward function, mapping states to rewards.
        
        Args:
            s: An NxD matrix of state vectors
        """
        pass