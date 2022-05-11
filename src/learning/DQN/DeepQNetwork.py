import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.fc1 = nn.Linear(in_features=state_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=action_size)

    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t
