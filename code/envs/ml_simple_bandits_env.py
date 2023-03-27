from envs.ml_wrapper_env import MetaLearningEnv
from gym import spaces
import numpy as np


class SimpleBandits(MetaLearningEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(1)

        self.seed()

        # Need to give small reward or the grad is 0
        self.env_rewards = np.array([
            [1, 0.4, -0.1],
            [-0.1, 0.4, 1]
        ])
        self.curr_task = 0
        self.n_envs = self.env_rewards.shape[0]

    def step(self, action):
        done = 1
        reward = np.asscalar(self.env_rewards[self.curr_task, action])
        return np.zeros(1), reward, done, {}

    def reset(self):
        return np.ones(1)

    def render(self, mode='human'):
        pass

    def close(self):
        super().close()

    def sample_task(self):
        return np.random.randint(low=0, high=self.env_rewards.shape[0], size=(1,))

    def set_task(self, task):
        self.curr_task = task

    def get_task(self):
        pass

    def log_diagnostics(self, prefix):
        pass

    def plot_params(self, *args, **kwargs):
        pass



