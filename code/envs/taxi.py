# %%
import gym
import torch
from matplotlib import pyplot as plt
from gym import error, spaces, utils
from utils.plotting import plot_options, plot_terminations, plot_policy

import numpy as np
from PIL import Image
from envs.ml_wrapper_env import MetaLearningEnv
# Env from Maximilian Igl
# First dimension: x, second dimension: y
# Layouts are defined by 'WALKABLE' (np array) and a map with constraints

MOVE = {
    0: np.array([0, 1]),  # No-op
    1: np.array([0, -1]),  # Turn
    2: np.array([1, 0]),
    3: np.array([-1, 0]),
}

ACTIONS = {
    0: np.array([0,0]), # No-op
    1: np.array([0,-1]),
    2: np.array([0,1]),
    3: np.array([1,0]),
    4: np.array([-1,0]),
    5: np.array([0,0]), # Pickup
}

shape = (8, 8)

TAXI_ROOMS_LAYOUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
])

LOCATIONS = {
    'R': np.array([1, 1]),
    'Y': np.array([6, 1]),
    'B': np.array([6, 5]),
    'G': np.array([1, 6])
}

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]


class Taxi(MetaLearningEnv):
    metadata = {'render.modes': ['human'],
                'video.frames_per_second': 20}
    max_tasks = 60

    def __init__(self,
                 reward_per_timestep=-0.1,
                 discount=0.95,
                 exclude_envs=(),
                 random_move_prob=0.0):
        envs_to_sample = tuple([i for i in range(Taxi.max_tasks) if i not in exclude_envs])
        super().__init__(envs_to_sample)

        self.viewer = None
        self.walkable = TAXI_ROOMS_LAYOUT
        self.grid = self.map = self.walkable
        self.timestep_limit = 1500
        self.timesteps = 0

        self.random_move_prob = random_move_prob

        self.actions = ACTIONS

        self.one_hot = True
        self.observation_space = spaces.Box(low=0., high=1., shape=(72,))
        self.action_space = spaces.Discrete(6)
        self.state_map, self.state_inverse_map = self._create_coord_mapping()

        self.state = {
            'loc': (0, 0),
            'pas': False
        }

        self.task_data = {
            'pic': np.array([0, 0]),
            'gol': np.array([0, 0]),
            'start': np.array([0, 0]),
            'pas': False
        }

        self.done = True

        self.task = None
        self.set_task(self.sample_task())

        self.reward_goal_found = 2
        self.reward_per_timestep = reward_per_timestep
        self.discount = discount

    def set_task(self, task):
        choices = ['R', 'Y', 'B', 'G']
        self.task = task
        self.timesteps = 0
        if task < 48:
            goal = task % 4
            pic = (task // 4) % 3
            start = ((task // 4) // 3) % 4

            self.task_data['gol'] = LOCATIONS[choices[goal]]
            self.task_data['start'] = LOCATIONS[choices[start]]
            self.task_data['pas'] = False
            self.task_data['pic'] = LOCATIONS[choices[(goal + pic + 1) % 4]]
        else:
            task -= 48
            goal = task % 4
            start = (task // 4) % 3
            self.task_data['gol'] = LOCATIONS[choices[goal]]
            self.task_data['start'] = LOCATIONS[choices[(goal + start + 1) % 4]]
            self.task_data['pas'] = True
            self.task_data['pic'] = np.array([0, 0])

    def _create_coord_mapping(self):
        """
        Create mapping from coordinates to state id and inverse mapping
        :return: Mapping from coords to id, mapping from id to coords
        """
        state_map = {}
        state_inverse_map = {}
        for i in range(1,7):
            for j in range(1,7):
                x = i - 1
                y = j - 1
                idx = y * 6 + x
                state_map[(i, j)] = idx
                state_inverse_map[idx] = (i, j)
                state_inverse_map[idx + 36] = (i, j)
        return state_map, state_inverse_map

    def reset(self):
        self.done = False
        self.timesteps = 0
        self.state['pas'] = self.task_data['pas']
        self.state['loc'] = self.task_data['start']
        return self.create_observation()

    def step(self, action):
        if self.done:
            raise RuntimeError("Environment must be reset")

        action_to_perform = float(action)
        if np.random.rand() < self.random_move_prob:
            action_to_perform = np.random.randint(0, 6)

        # No-op (0) or movement (1-4)
        if action_to_perform < 5:
            mov = self.actions[action_to_perform]
            loc = self.state['loc']
            new_loc = loc + mov
            reward = self.reward_per_timestep
            if not (self.check_inside_area(new_loc) and self.check_walkable(new_loc)):
                new_loc = loc
            self.state['loc'] = new_loc

        # Pickup
        elif action_to_perform == 5:
            if self.check_pickup_possible():
                self.state['pas'] = True
                reward = self.reward_per_timestep
            elif self.check_dropoff_possible():
                self.done = True
                reward = self.reward_goal_found
            else:
                reward = self.reward_per_timestep

        self.timesteps += 1
        if self.timesteps >= self.timestep_limit:
            if not self.done:
                self.done = True
                reward = self.reward_per_timestep * 1 / (1 - self.discount)

        obs = self.create_observation()
        return obs, reward, self.done, {'state': self.state, 'task': self.task_data}

    def create_observation(self, loc=None, pas=None):
        if loc is None:
            loc = self.state['loc']
        if pas is None:
            pas = self.state['pas']
        obs = np.zeros(72)
        # Remove wall from state
        x = loc[0] - 1
        y = loc[1] - 1
        idx = pas * 36 + y * 6 + x
        obs[idx] = 1
        return obs

    def check_pickup_possible(self):
        return np.all(self.state['loc'] == self.task_data['pic']) and not self.state['pas']

    def check_dropoff_possible(self):
        return np.all(self.state['loc'] == self.task_data['gol']) and self.state['pas']

    def check_walkable(self, loc):
        return self.walkable[tuple(loc)] == 1

    def check_inside_area(self, loc):
        return (0 <= loc[0] < self.walkable.shape[0] and
                0 <= loc[1] < self.walkable.shape[1])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            max_x, max_y = self.walkable.shape
            square_size = 75

            screen_height = square_size * max_x
            screen_width = square_size * max_y
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.square_map = {}
            for i in range(max_x):
                for j in range(max_y):
                    l = j * square_size
                    r = l + square_size
                    t = max_x * square_size - i * square_size
                    b = t - square_size
                    square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    self.viewer.add_geom(square)
                    self.viewer.square_map[(i, j)] = square

        for square_coords in self.viewer.square_map:
            color = None
            i, j = square_coords
            square = self.viewer.square_map[square_coords]

            # Agent
            if square_coords == tuple(self.state["loc"]):
                if self.state["pas"]:
                    color = [1, 1, 0]
                else:
                    color = [1, 0, 1]
            # Goal
            elif square_coords == tuple(self.task_data["gol"]):
                color = [0, 0, 1]

            # Passenger
            elif (not self.state["pas"]) and square_coords == tuple(self.task_data["pic"]):
                color = [0, 1, 1]

            # elif square_coords == tuple(self.default_start):
            #     color = [0, 1, 0]
            elif self.walkable[i, j] == 0:
                color = [0, 0, 0]
            elif self.walkable[i, j] == 1:
                color = [1, 1, 1]
            square.set_color(*color)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_mdp_description(self):
        states = self.observation_space.shape[0]
        actions = self.action_space.n

        # +1 for terminal state
        transition_probs = np.zeros((states, actions, states))
        reward_distribution = np.zeros((states, actions))

        for state in range(states):
            y_coord, x_coord = self.state_inverse_map[state]
            pas = state >= 36

            # If it's a wall continue
            if not self.check_walkable((y_coord, x_coord)):
                continue

            curr_coords = np.array((y_coord, x_coord))
            for action in range(actions):
                # If it is a movement action reward is per_step reward and next_state might change
                if action < 5:
                    mov = self.actions[action]
                    new_coords = curr_coords + mov
                    new_pas = pas
                    reward = self.reward_per_timestep
                    if not (self.check_inside_area(new_coords) and self.check_walkable(new_coords)):
                        new_coords = curr_coords

                    next_state = self.state_map[tuple(new_coords)] + new_pas * 36

                # If it is a pickup action nothing happens unless it is passenger cell and it is picked up
                #  or it is a goal state and passenger is on board
                elif action == 5:
                    new_coords = curr_coords
                    reward = self.reward_per_timestep
                    if np.all(curr_coords == self.task_data['pic']) and not pas:
                        new_pas = True
                    else:
                        new_pas = pas

                    next_state = self.state_map[tuple(new_coords)] + new_pas * 36

                transition_probs[state, action, next_state] = 1
                reward_distribution[state, action] = reward

            if np.all(curr_coords == self.task_data['gol']) and pas:
                reward_distribution[state, 5] = self.reward_goal_found
                transition_probs[state, 5, state] = 0

        return transition_probs, reward_distribution

    def get_plot_arrow_params(self, max_val, max_index):
        entries = len(max_val)
        x_pos = np.zeros(entries)
        y_pos = np.zeros(entries)
        x_dir = np.zeros(entries)
        y_dir = np.zeros(entries)
        color = np.zeros(entries)
        # down, up , right, left

        # none left right down up none
        for i in range(entries):
            y = (i // 6) + 1
            x = (i % 6) + 1
            coords = (x, y)
            if self.map[coords] == 0:
                continue
            if max_index[i] == 3:
                y_dir[i] = 1
            elif max_index[i] == 4:
                y_dir[i] = -1
            elif max_index[i] == 2:
                x_dir[i] = 1
            elif max_index[i] == 1:
                x_dir[i] = -1

            x_pos[i] = coords[1] + 0.5
            y_pos[i] = coords[0] + 0.5
            color[i] = max_val[i]
        return x_pos, y_pos, x_dir, y_dir, color

    def create_coords(self):
        coords = []
        for i in range(36):
            y = (i // 6) + 1
            x = (i % 6) + 1
            coords.append((x, y))
        return coords

    def plot_params(self, option_probs, termination_probs, policy_probs, name_prefix="", no_blocks=False):
        grid = self.map if no_blocks else self.grid
        options = option_probs.shape[1]
        coords = self.create_coords()
        # TODO Refactor taxi and four envs a bit
        #  Mental note: Plot probabilities is probably good to have in env (since plots are different or call it plot
        #  values and use it for baseline as well)
        #  Add ploting for baseline too
        for i in range(options):
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 9))

            for j in range(2):
                if j == 0:
                    passanger_status = "NoPassenger"
                    selected_policy_probs = policy_probs[:36, i, :]
                    selected_termination_probs = termination_probs[:36, i]
                    seleceted_option_probs = option_probs[:36, i]
                else:
                    passanger_status = "Passenger"
                    selected_policy_probs = policy_probs[36:, i, :]
                    selected_termination_probs = termination_probs[36:, i]
                    seleceted_option_probs = option_probs[36:, i]
                max_val, max_index = torch.max(selected_policy_probs, dim=1)
                max_val = max_val.numpy()
                max_index = max_index.numpy()

                plot_policy(ax=axes[0][j], arrow_data=self.get_plot_arrow_params(max_val, max_index), grid=grid,
                            title_suffix=passanger_status)
                plot_terminations(ax=axes[1][j], probs=selected_termination_probs, coords=coords, grid=grid,
                                  title_suffix=passanger_status)
                plot_options(ax=axes[2][j], probs=seleceted_option_probs, coords=coords, grid=grid,
                             title_suffix=passanger_status)

            plt.tight_layout()
            plt.savefig(name_prefix + "_option{}".format(i) + ".png", bbox_inches='tight', dpi=150)

# register(
#     id='Taxi-v1',
#     entry_point='environments.taxi:Taxi',
#     max_episode_steps=50,
#     kwargs={'add_action_in_obs': False,
#             'image_obs': False}
# )


def test_env_set_task():
    env = Taxi()
    test = []
    for e in range(env.n_envs):
        env.set_task(e)
        print(f"Env {e} Goal {env.task_data['gol']} Pic {env.task_data['pic']} "
              f"Start {env.task_data['start']} Pass {env.task_data['pas']}")
        t = (list(env.task_data['gol']), list(env.task_data['pic']), list(env.task_data['start']), env.task_data['pas'])
        assert t not in test
        test.append(t)

        if e < 48:
            assert not env.task_data['pas']
            assert not np.all(env.task_data['pas'] == env.task_data['gol'])
        else:
            assert env.task_data['pas']
            assert not np.all(env.task_data['start'] == env.task_data['gol'])
    print("Test env set_task successful")


def test_envs_to_sample():
    from collections import Counter
    np.random.seed(42)
    envs_to_exclude = (1,3,4,20,57,47,30)
    env = Taxi(exclude_envs=envs_to_exclude)
    samples = 50000
    counter = Counter()

    for sample in range(samples):
        env.set_task(env.sample_task())
        assert env.get_task() not in envs_to_exclude, f"{env.get_task()} in envs_to_exclude"
        counter[env.get_task()] += 1
    percentages = (np.array(list(counter.values())) - 1) / samples
    assert np.allclose(percentages, 1.0/env.n_envs, atol=5e-2)
    assert len(percentages) == env.n_envs
    print("Test envs_to_sample successful")


if __name__ == '__main__':
    test_env_set_task()
    test_envs_to_sample()

    env = Taxi()
    b = np.copy(TAXI_ROOMS_LAYOUT)
    b[LOCATIONS['R'][0], LOCATIONS['R'][1]] = 2
    b[LOCATIONS['Y'][0], LOCATIONS['Y'][1]] = 3
    b[LOCATIONS['B'][0], LOCATIONS['B'][1]] = 4
    b[LOCATIONS['G'][0], LOCATIONS['G'][1]] = 5

    env_runs = 100
    env.seed(42)
    np.random.seed(42)
    for e in env.envs_to_sample:

        env.set_task(e)
        if (not env.task_data['pas']) and np.all(env.task_data['gol'] == env.task_data['start']):
            print(f"Task {e}\nStart/Goal: {env.task_data['gol']}\nPic: {env.task_data['pic']}")
        env.reset()
        # init_pass = env.state["pas"]
        # print("Goal {}, Pass {} Start {}".format(env.task_data["gol"], env.task_data["pic"], env.state))
        # print("Picked Up {}".format(init_pass))
        # env.render()
        # while True:
        #     state, reward, done, info = env.step(env.action_space.sample())
        #     env.render()
        #     if done:
        #         break