import torch
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from torch.nn.functional import one_hot
from gflownet.gflownet import GFlowNet
from policy import ForwardPolicy, BackwardPolicy
from gflownet.utils import trajectory_balance_loss
from torch.optim import Adam
from grid import Grid

size = 16

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

def train(batch_size, num_epochs):
    env = Grid(size=size)
    forward_policy = ForwardPolicy(env.state_dim, hidden_dim=128, num_actions=env.num_actions)
    backward_policy = BackwardPolicy(env.state_dim, num_actions=env.num_actions)
    model = GFlowNet(forward_policy, backward_policy, env)
    opt = Adam(model.parameters(), lr=5e-3)

    for i in (p := tqdm(range(num_epochs))):
        s0 = one_hot(torch.zeros(batch_size).long(), env.state_dim).float()
        s, stats = model.sample_states(s0, return_stats=True)
        loss = trajectory_balance_loss(stats.total_flow,
                                       stats.rewards,
                                       stats.fwd_probs,
                                       stats.back_probs)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if i % 10 == 0: p.set_description(f"{loss.item():.3f}")

    s0 = one_hot(torch.zeros(10**4).long(), env.state_dim).float()
    s, _ = model.sample_states(s0, return_stats=False)
    plot(s, env)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=1000)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    
    train(batch_size, num_epochs)