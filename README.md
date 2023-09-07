# GFlowNet in PyTorch

![gflownet](images/thumbnail.png)

This repo is associated with the blog post ["Proportional Reward Sampling With GFlowNets"](https://sigmoidprime.com/post/gflownets/) over at [sigmoid prime](https://sigmoidprime.com). It contains an implementation of a Generative Flow Network (GFlowNet), proposed by Bengio et al. in the paper ["Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation"](https://arxiv.org/abs/2106.04399) (2021).

The model is trained using online learning (i.e. by continually evaluating samples drawn from the model's own policy rather than a fixed set of samples drawn from another policy) and the [trajectory balance loss](https://arxiv.org/abs/2201.13259). We evaluate the model's performance using the grid domain of the original paper. This is visualized by the end of training.

![samples](images/samples.png)

The code for training the model is simple:

1. Initialize the grid environment using a grid size
2. Define a policy network taking a state vector as input and outputting a vector of probabilities over possible actions. (In the grid domain, the number of actions is three: **Down**, **Right**, and **Terminate**.)
3. Define a backward policy. In this case, the policy is not estimated but fixed to 0.5 for all parent states (except when there is only one parent state).

With this, you initialize the GFlowNet along with the optimizer to use during training.

```python
env = Grid(size=16)
forward_policy = ForwardPolicy(env.state_dim, hidden_dim=32, num_actions=3)
model = GFlowNet(forward_policy, backward_policy, env)
opt = Adam(model.parameters(), lr=5e-3)
```

To train the model, construct an NxD matrix of initial states, where N is the desired number of samples and D is the dimensionality of the state vector (i.e. `state_dim`). Then, draw samples from the model using the `sample_states` method, giving it the initial states and setting `return_log=True`. The resulting `Log` object contains information about the trajectory of each sample, which is used to compute the trajectory balance loss.

```python
for i in range(num_epochs):
  s0 = one_hot(torch.zeros(batch_size).long(), env.state_dim).float()
  s, log = model.sample_states(s0, return_log=True)
  loss = trajectory_balance_loss(log.total_flow, log.rewards, log.fwd_probs, log.back_probs)
  loss.backward()
  opt.step()
  opt.zero_grad()
```

Finally, when the model has been trained, you can sample states using the same `sample_states(...)` method as before, this time without needing to supply the `return_log=True` argument.

```python
s0 = one_hot(torch.zeros(10**4).long(), env.state_dim).float()
s = model.sample_states(s0)
```
