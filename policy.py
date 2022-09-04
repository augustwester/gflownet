from torch import nn
from torch.nn.functional import softmax

class ForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, num_actions) # down, right, terminate
    
    def forward(self, s):
        x = self.dense1(s)
        x = self.dense2(x)
        return softmax(x, dim=1)