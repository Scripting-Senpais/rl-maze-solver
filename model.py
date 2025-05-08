# Import necessary PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os

# Constants for log standard deviation clamping to ensure numerical stability
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6  # Small constant to prevent log(0)

# Initialize weights of linear layers using Xavier initialization
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# ==============================
# CRITIC NETWORK (Twin Q-networks)
# ==============================
class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, checkpoint_dir='checkpoints', name='critic_network'):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture (used to reduce overestimation bias)
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        # Saving and loading setup
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        # Apply weight initialization
        self.apply(weights_init_)

    def forward(self, state, action):
        # Concatenate state and action for input
        xu = torch.cat([state, action], 1)

        # Q1 forward pass
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # Q2 forward pass
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2  # Return both Q-values

    def save_checkpoint(self):
        # Save model parameters
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        # Load model parameters
        self.load_state_dict(torch.load(self.checkpoint_file))


# ============================
# ACTOR NETWORK (Stochastic)
# ============================
class Actor(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, checkpoint_dir='checkpoints', name="actor_network"):
        super(Actor, self).__init__()

        # Actor network layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers for mean and log standard deviation of Gaussian
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        # Saving and loading setup
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        # Apply weight initialization
        self.apply(weights_init_)

        # Set action scale and bias based on action space bounds
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        # Pass through two hidden layers with ReLU
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        # Output the mean and log standard deviation of the Gaussian distribution
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        # Clamp log_std to avoid numerical instability
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        """
        Sample an action from the policy distribution using the reparameterization trick.
        Returns:
            - action: sampled action (squashed with tanh)
            - log_prob: log probability of the action
            - mean: deterministic action (useful for evaluation)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()                     # Convert log std to std
        normal = Normal(mean, std)              # Create normal distribution
        x_t = normal.rsample()                  # Reparameterization trick (sample + differentiate)
        y_t = torch.tanh(x_t)                   # Apply tanh to bound the action in [-1, 1]
        action = y_t * self.action_scale + self.action_bias  # Rescale to actual action space

        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)  # Sum over action dimensions

        # Also return deterministic action (for evaluation)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        # Move model and tensors to specified device
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)

    def save_checkpoint(self):
        # Save model parameters
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        # Load model parameters
        self.load_state_dict(torch.load(self.checkpoint_file))