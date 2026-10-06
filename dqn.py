import torch
import torch.nn as nn


class DQN(nn.Module):

    # input_dim = state size
    # output_dim = number of actions
    # hidden_dim = neurons in hidden layer
    def __init__(self, state_dim=12, action_dim=2, hidden_dim=256):

        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.model(x)