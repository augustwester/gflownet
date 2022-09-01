import torch
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot
from gflownet import GFlowNet
from forward_policy import ForwardPolicy
from utils import detailed_balance_loss
from torch.optim import Adam
from env import Hypergrid

size = 20

def plot(samples, env):
    state_dim = samples.shape[1]
    mat1 = torch.zeros(state_dim)
    mat2 = torch.zeros(state_dim)

    for s_ in samples: mat1[s_.argmax()] += 1
    for i in range(state_dim):
        s = one_hot(torch.LongTensor([i]), state_dim).float()
        s = s.view(1, 1, -1)
        mat2[i] = env.reward(s)
        
    mat1 = mat1.view(size, size)
    mat2 = mat2.view(size, size)

    _, ax = plt.subplots(1, 2)
    ax[0].matshow(mat1.numpy())
    ax[1].matshow(mat2.numpy())
    plt.show()

def backward_policy(traj):
    probs = 0.5 * torch.ones(traj.shape[:2])
    idx = traj.argmax(-1)
    probs[(idx > 0) & (idx % size == 0)] = 1
    probs[idx < size] = 1
    return probs[:, 1:]

num_epochs = 350
env = Hypergrid(dim=2, size=size, num_actions=3)
forward_policy = ForwardPolicy(env.state_dim, hidden_dim=128, num_actions=3)
model = GFlowNet(forward_policy, backward_policy, env)
opt = Adam(model.parameters(), lr=5e-3)

for i in range(num_epochs):
    s0 = one_hot(torch.zeros(256).long(), env.state_dim).float()
    s, stats = model.sample_states(s0, explore=False, return_stats=True)
    loss = detailed_balance_loss(stats.fwd_probs,
                                 stats.back_probs,
                                 stats.term_probs,
                                 stats.rewards)
    loss.backward()
    opt.step()
    opt.zero_grad()
    if i % 10 == 0: print(loss)

s0 = one_hot(torch.zeros(10**4).long(), env.state_dim).float()
s, _ = model.sample_states(s0, explore=False, return_stats=False)
plot(s, env)