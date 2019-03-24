import collections
import numpy as np
import random

class replay_buffer:
    def __init__(self, max_length=1e6):
        self.key = ['state', 'next_state', 'reward', 'done', 'action']
        self.memory = collections.deque(maxlen=int(max_length))

    def append(self, state, next_state, reward, done, action):
        self.memory.append((state, next_state, reward, done, action))

    def get_sample(self, sample_size=32):
        batch = random.sample(self.memory, sample_size)
        state = np.stack([e[0] for e in batch])
        next_state = np.stack([e[1] for e in batch])
        reward = np.stack([e[2] for e in batch])
        done = np.stack([e[3] for e in batch])
        action = np.stack([e[4] for e in batch])
        batch_memory = [state, next_state, reward, done, action]

        return {k:v for k, v in zip(self.key, batch_memory)}