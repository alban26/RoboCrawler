from collections import namedtuple, deque
import random

import numpy as np

""""
Datenstruktur für die Erfahrungen
"""
Experience = namedtuple(
    'Experience', field_names=['state', 'action', 'next_state', 'reward'])


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
        states, actions, next_states, rewards = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(next_states), np.array(rewards, dtype=np.float32)

