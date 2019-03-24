import numpy as np

class OU_noise:
    def __init__(self,action_size,worker_size,mu=0,theta=0.1,sigma=0.1):
        self.action_size = action_size
        self.worker_size = worker_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.X = np.ones(self.action_size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X