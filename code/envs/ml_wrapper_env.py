from gym.core import Env
from abc import ABC, abstractmethod
import numpy as np


class MetaLearningEnv(ABC, Env):
    """
    Wrapper around OpenAI gym environments, interface for meta learning
    """

    def __init__(self, envs_to_sample):
        super(Env, self).__init__()
        self.task = None
        self.envs_to_sample = envs_to_sample
        self.n_envs = len(envs_to_sample)

    def sample_task(self):
        """
        Samples a task of the meta-environment

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        return self.envs_to_sample[np.random.randint(0, self.n_envs)]

    @abstractmethod
    def set_task(self, task):
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        return self.task





