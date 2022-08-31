import torch
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot
from gflownet import GFlowNet
from forward_policy import ForwardPolicy
from utils import detailed_balance_loss
from torch.optim import Adam
    
size_x, size_y = 20, 20
center_x, center_y = size_x // 2, size_y // 2
var_x, var_y = 3, 3
num_actions = 3 # down, right, terminate
state_dim = size_x * size_y

def reward_fn(s):
    grid = s.view(len(s), size_y, size_x)
    coord = (grid == 1).nonzero()[:, 1:].view(len(s), 2)
    R0, R1, R2 = 1e-2, 0.5, 2
    R1_term = torch.prod(0.25 < torch.abs(coord / (size_x-1) - 0.5), dim=1)
    R2_term = torch.prod((0.3 < torch.abs(coord / (size_x-1) - 0.5)) & (torch.abs(coord / (size_x-1) - 0.5) < 0.4), dim=1)
    return (R0 + R1*R1_term + R2*R2_term).unsqueeze(1)

def mask_fn(s):
    mask = torch.ones(len(s), num_actions)
    idx = s.argmax(1) + 1
    right_edge = (idx > 0) & (idx % (size_x) == 0)
    bottom_edge = idx > size_x*(size_y-1)
    mask[right_edge, 1] = 0
    mask[bottom_edge, 0] = 0
    return mask

def update_fn(s, actions):
    idx = s.argmax(1)
    down, right = actions == 0, actions == 1
    idx[down] = idx[down] + size_x
    idx[right] = idx[right] + 1
    return one_hot(idx, state_dim).float()

def backward_policy(traj):
    probs = 0.5 * torch.ones(traj.shape[:2])
    idx = traj.argmax(-1)
    probs[(idx > 0) & (idx % size_x == 0)] = 1
    probs[idx < size_x] = 1
    return probs[:, 1:]

forward_policy = ForwardPolicy(state_dim, hidden_dim=128, num_actions=num_actions)
model = GFlowNet(forward_policy, backward_policy, update_fn, reward_fn, mask_fn)

num_epochs = 300
opt = Adam(model.parameters(), lr=5e-3)

for i in range(num_epochs):
    s0 = one_hot(torch.zeros(256).long(), state_dim).float()
    s, stats = model.sample_states(s0, explore=False)
    loss = detailed_balance_loss(stats.fwd_prob,
                                 stats.back_prob,
                                 stats.term_prob,
                                 stats.rewards)
    loss.backward()
    opt.step()
    opt.zero_grad()
    if i % 10 == 0: print(loss)

s0 = one_hot(torch.zeros(2**15).long(), state_dim).float()
s, _ = model.sample_states(s0, explore=False)

mat1 = torch.zeros(state_dim)
mat2 = torch.zeros(state_dim)

for s_ in s:
    mat1[s_.argmax()] += 1

for i in range(size_x*size_y):
    s = one_hot(torch.LongTensor([i]), size_x*size_y).float()
    s = s.view(1, 1, -1)
    mat2[i] = reward_fn(s)
    
mat1 = mat1.view(size_y, size_x)
mat2 = mat2.view(size_y, size_x)
print(mat1)

_, ax = plt.subplots(1, 2)
ax[0].matshow(mat1.numpy())
ax[1].matshow(mat2.numpy())
plt.show()