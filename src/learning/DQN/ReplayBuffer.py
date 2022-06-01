import random
from collections import namedtuple, deque

import numpy as np
""""
Datenstruktur für die Erfahrungen
"""
Experience = namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'next_state'])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, next_state = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64), \
               np.array(rewards, dtype=np.float32), np.array(next_state, dtype=np.float32),

