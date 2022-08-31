from torch import nn
from torch.nn.functional import softmax, sigmoid

class ForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions) -> None:
        super().__init__()
        self.dense1 = nn.Linear(state_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, num_actions) # down, right, terminate
    
    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return softmax(x, dim=1)