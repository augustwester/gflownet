import torch
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from torch.nn.functional import one_hot
from gflownet import GFlowNet
from policy import ForwardPolicy, RandomPolicy
from utils import trajectory_balance_loss
from torch.optim import Adam
from env import Hypergrid

size = 8

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

def backward_policy(s):
    idx = s.argmax(-1)
    probs = 0.5 * torch.ones(len(s), 1)
    probs[(idx > 0) & (idx % size == 0)] = 1
    probs[idx < size] = 1
    return probs

def train(batch_size, num_epochs, learning):
    env = Hypergrid(dim=2, size=size, num_actions=3)
    forward_policy = ForwardPolicy(env.state_dim, hidden_dim=128, num_actions=3)
    model = GFlowNet(forward_policy, backward_policy, env)
    opt = Adam(model.parameters(), lr=5e-3)
    
    if learning == "offline":
        random_policy = RandomPolicy(num_actions=3)
        random_model = GFlowNet(random_policy, backward_policy, env)
        s0 = one_hot(torch.zeros(100000).long(), env.state_dim).float()
        s, replay_buffer = random_model.sample_states(s0, return_stats=True)

    for i in (p := tqdm(range(num_epochs))):
        if learning == "offline":
            sample_idxs = torch.randint(low=0, high=replay_buffer.num_samples, size=(batch_size,))
            _s = s[sample_idxs]
            traj = replay_buffer.traj[sample_idxs]
            actions = replay_buffer.actions[sample_idxs]
            fwd_probs, back_probs, rewards = model.evaluate_trajectories(_s, traj, actions)
            loss = trajectory_balance_loss(model.total_flow, rewards, fwd_probs, back_probs)
        else:
            s0 = one_hot(torch.zeros(batch_size).long(), env.state_dim).float()
            s, stats = model.sample_states(s0, return_stats=True)
            loss = trajectory_balance_loss(stats.total_flow, stats.rewards, stats.fwd_probs, stats.back_probs)
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
    parser.add_argument("--learning", choices=("online", "offline"), default="online")

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning = args.learning
    
    train(batch_size, num_epochs, learning)