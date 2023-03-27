import numpy as np
import os
import warnings
from gym import utils
from gym.envs.mujoco import mujoco_env
from envs.ml_wrapper_env import MetaLearningEnv

# TASKS
# 0 - up
# 1 - down
# 2 - right up
# 3 - right down
# 4 - right right
# 5 - right right up
# 6 - right right down
# 7 - up right
# 8 - down right


class AntObstaclesGenEnv(mujoco_env.MujocoEnv, utils.EzPickle, MetaLearningEnv):
    max_tasks = 9

    def __init__(self, exclude_envs=(), enable_resets=True):
        self.count = 0
        self.enable_resets = enable_resets
        envs_to_sample = tuple([i for i in range(AntObstaclesGenEnv.max_tasks) if i not in exclude_envs])
        MetaLearningEnv.__init__(self, envs_to_sample=envs_to_sample)
        mujoco_env.MujocoEnv.__init__(self, os.path.dirname(__file__) + '/assets/ant_obstacles_gen.xml', 5)
        utils.EzPickle.__init__(self)
        self.set_task(self.sample_task())

    def step(self, a):
        # DISABLED resets
        self.count += 1
        if self.enable_resets and self.count % 200 == 0:
            n_qpos = self.init_qpos + np.random.uniform(size=self.model.nq, low=-.1, high=.1)
            n_qvel = self.init_qvel + np.random.randn(self.model.nv) * .1
            n_qpos[:2] = self.data.qpos[:2]
            n_qpos[-11:] = self.data.qpos[-11:]
            self.set_state(n_qpos, n_qvel)

        goal = np.array([8, 24])
        if self.task == 0:
            goal = np.array([8, 24])
        elif self.task == 1:
            goal = np.array([8, -24])
        elif self.task == 2:
            goal = np.array([24, 24])
        elif self.task == 3:
            goal = np.array([24, -24])
        elif self.task == 4:
            goal = np.array([48, 0])

        elif self.task == 5:
            goal = np.array([40, 24])
        elif self.task == 6:
            goal = np.array([40, -24])
        elif self.task == 7:
            goal = np.array([32, 16])
        elif self.task == 8:
            goal = np.array([32, -16])
        elif self.task == 9:
            goal = np.array([40, 24])
        elif self.task == 10:
            goal = np.array([40, -24])
        elif self.task == 11:
            goal = np.array([40, 24])
        elif self.task == 12:
            goal = np.array([40, -24])



        # reward = -np.sum(np.square(self.data.qpos[:2,0] - goal)) / 100000

        xposbefore = self.data.qpos[0]
        yposbefore = self.data.qpos[1]

        self.do_simulation(a, self.frame_skip)

        xposafter = self.data.qpos[0]
        yposafter = self.data.qpos[1]

        if xposbefore < goal[0]:
            forward_reward = (xposafter - xposbefore)/self.dt
        else:
            forward_reward = -1*(xposafter - xposbefore)/self.dt
        if yposbefore < goal[1]:
            forward_reward += (yposafter - yposbefore)/self.dt
        else:
            forward_reward += -1*(yposafter - yposbefore)/self.dt

        ctrl_cost = .1 * np.square(a).sum()
        reward = forward_reward - ctrl_cost

        # print(reward)
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[:-11],
            self.data.qvel.flat[:-11],
            # self.data.qpos.flat,
            # self.data.qvel.flat,
        ])

    def set_task(self, task):
        # if task >= self.n_envs:
        #     raise NotImplementedError
        self.task = task

    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + np.random.randn(self.model.nv) * .1

        # self.realgoal = 4
        if self.task == 0:
            qpos[-11:] = np.array([80, 0, 0, 80, 0, 0, 0, 0, 0, 8, 24])
        elif self.task == 1:
            qpos[-11:] = np.array([0, 0, 0, 80, 0, 0, 80, 0, 0, 8, -24])
        elif self.task == 2:
            qpos[-11:] = np.array([0, 80, 0, 80, 80, 0, 0, 0, 0, 24, 24])
        elif self.task == 3:
            qpos[-11:] = np.array([0, 0, 0, 80, 80, 0, 0, 80, 0, 24, -24])
        elif self.task == 4:
            qpos[-11:] = np.array([0, 0, 0, 80, 80, 80, 0, 0, 0, 48, 0])
        elif self.task == 5:
            qpos[-11:] = np.array([0, 0, 80, 80, 80, 80, 0, 0, 0, 40, 24])
        elif self.task == 6:
            qpos[-11:] = np.array([0, 0, 0, 80, 80, 80, 0, 0, 80, 40, -24])
        elif self.task == 7:
            qpos[-11:] = np.array([80, 80, 0, 80, 0, 0, 0, 0, 0, 32, 16])
        elif self.task == 8:
            qpos[-11:] = np.array([0, 0, 0, 80, 0, 0, 80, 80, 0, 32, -16])
        elif self.task == 9:
            qpos[-11:] = np.array([80, 80, 80, 80, 0, 0, 0, 0, 0, 40, 24])
        elif self.task == 10:
            qpos[-11:] = np.array([0, 0, 0, 80, 0, 0, 80, 80, 80, 40, -24])
        elif self.task == 11:
            qpos[-11:] = np.array([0, 80, 80, 80, 80, 0, 0, 0, 0, 40, 24])
        elif self.task == 12:
            qpos[-11:] = np.array([0, 0, 0, 80, 80, 0, 0, 80, 80, 40, -24])
        else:
            raise NotImplementedError(f"Env id {self.task} not available")

        if self.task > 8:
            warnings.warn(f"Running extra env {self.task} that was not in MLSH paper")

        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.6
