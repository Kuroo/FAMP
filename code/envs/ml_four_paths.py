import numpy as np
import torch
from gym import spaces
from envs.ml_wrapper_env import MetaLearningEnv
from matplotlib import pyplot as plt
from utils.plotting import plot_options, plot_terminations, plot_policy


class FourPaths(MetaLearningEnv):
    """
    Class representing environment defined in
    Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning.
    Described on page 14.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    large_map = dict(
        map=np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        envs=8,
        start_coords=8 * (np.array([2, 1]),),
        end_coords=8 * (np.array([2, 11]),),
        block_coords=(
            ((1, 3), (1, 7), (1, 11)),
            ((3, 3), (1, 7), (1, 11)),
            ((1, 3), (3, 7), (1, 11)),
            ((3, 3), (3, 7), (1, 11)),
            ((1, 3), (1, 7), (3, 11)),
            ((3, 3), (1, 7), (3, 11)),
            ((1, 3), (3, 7), (3, 11)),
            ((3, 3), (3, 7), (3, 11))
        ),
        steplimit=1000
    )

    medium_map = dict(
        map=np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        envs=4,
        start_coords=4 * (np.array([2, 1]),),
        end_coords= 4 * (np.array([2, 7]),),
        block_coords=(
            ((1, 3), (1, 7)),
            ((1, 3), (3, 7)),
            ((3, 3), (1, 7)),
            ((3, 3), (3, 7))
        ),
        # steplimit=1000
        steplimit=200
    )

    small_map = dict(
        map=np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]),
        envs=2,
        start_coords=2 * (np.array([3, 1]),),
        end_coords=2 * (np.array([3, 3],),),
        block_coords=(
            ((4, 3),),
            ((2, 3),)
        ),
        steplimit=200
    )

    def __init__(self, map_name="grid_small", one_hot=True, exclude_envs=(), random_move_prob=0.0):
        """
        Initializes environment
        """
        self.map_data = string_to_map(map_name)
        self.max_tasks = self.map_data["envs"]
        envs_to_sample = tuple([i for i in range(self.max_tasks) if i not in exclude_envs])
        super().__init__(envs_to_sample)

        self.viewer = None
        self.random_move_prob = random_move_prob

        self.map = self.map_data["map"]
        self.grid = self.map.copy()
        self.timestep_threshold = self.map_data["steplimit"]
        self.timestep = 0

        self.action_space = spaces.Discrete(4)

        self.one_hot = one_hot
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(2 if not self.one_hot else np.sum(self.grid),))
        self.state_map, self.state_inverse_map = self._create_coord_mapping()

        self.done = True

        self.task = None
        self.default_start = None
        self.goal_state = None
        self.set_task(0)

        self.current_state = self.default_start.copy()  # set start state to default

    def set_task(self, task):
        self.grid = self.map.copy()
        for i in self.map_data["block_coords"][task]:
            self.grid[i] = 0
        self.default_start = self.map_data["start_coords"][task]
        self.goal_state = self.map_data["end_coords"][task]
        self.task = task
        self.timesteps = 0

    def _create_coord_mapping(self):
        """
        Create mapping from coordinates to state id and inverse mapping
        :return: Mapping from coords to id, mapping from id to coords
        """
        state_map = {}
        state_inverse_map = {}
        index = 0
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] != 0:
                    state_map[(i, j)] = index
                    state_inverse_map[index] = (i, j)
                    index += 1
        return state_map, state_inverse_map

    def reset(self):
        """Resets goal position to one of the default positions, if none is specified goal is set to fixed pos"""
        self.current_state = self.default_start.copy()
        self.done = False
        self.timestep = 0
        return self.create_observation() if self.one_hot else self.current_state

    def step(self, action):
        """
        Perform action and get new state, reward and done flag from environment
        :param action: int[0-3] each action corresponds to one direction
        :return: new state, reward, done flag
        """
        assert not self.done, "Environment is done"
        action_to_perform = float(action)
        if np.random.rand() < self.random_move_prob:
            action_to_perform = np.random.randint(0, 4)

        self.update_state(action_to_perform)
        if np.all(self.current_state == self.goal_state):
            reward = 1
            self.done = True
        else:
            reward = 0

        self.timestep += 1
        if self.timestep >= self.timestep_threshold:
            self.done = True

        return self.create_observation() if self.one_hot else self.current_state, reward, self.done, None

    def create_observation(self):
        index = self.state_map[tuple(self.current_state)]
        onehot = np.zeros(self.observation_space.shape[0])
        onehot[index] = 1
        return onehot

    def update_state(self, action):
        """
        Check if the next state is a wall and update accordingly
        :param action: int[0-3] each action corresponds to one direction
        """
        y_coord = self.current_state[0]
        x_coord = self.current_state[1]
        # 0 up
        # 1 down
        # 2 right
        # 3 left
        if action == 0:
            y_coord += 1
        elif action == 1:
            y_coord -= 1
        elif action == 2:
            x_coord += 1
        elif action == 3:
            x_coord -= 1

        if self.grid[y_coord][x_coord] == 1:
            self.current_state[:] = [y_coord, x_coord]

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            max_x, max_y = self.grid.shape
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
            if square_coords == tuple(self.current_state):
                color = [1, 0, 1]
            elif square_coords == tuple(self.goal_state):
                color = [0, 0, 1]
            elif square_coords == tuple(self.default_start):
                color = [0, 1, 0]
            elif self.grid[i, j] == 0:
                color = [0, 0, 0]
            elif self.grid[i, j] == 1:
                color = [1, 1, 1]
            square.set_color(*color)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_mdp_description(self):
        states = self.observation_space.shape[0]
        actions = self.action_space.n

        transition_probs = np.zeros((states, actions, states))
        reward_distribution = np.zeros((states, actions))

        for state in range(states):
            y_coord, x_coord = self.state_inverse_map[state]
            if y_coord == self.goal_state[0] and x_coord == self.goal_state[1]:
                continue
            if self.grid[y_coord][x_coord] == 0:  # A state could be a wall
                continue
            for action in range(actions):
                if action == 0:  # 0 up
                    move_x_coord = x_coord
                    move_y_coord = y_coord + 1
                elif action == 1:  # 1 down
                    move_x_coord = x_coord
                    move_y_coord = y_coord - 1
                elif action == 2:  # 2 right
                    move_x_coord = x_coord + 1
                    move_y_coord = y_coord
                elif action == 3:  # 3 left
                    move_x_coord = x_coord - 1
                    move_y_coord = y_coord

                if self.grid[move_y_coord][move_x_coord] == 1:  # If a valid move
                    next_state = self.state_map[(move_y_coord, move_x_coord)]
                    if move_y_coord == self.goal_state[0] and move_x_coord == self.goal_state[1]:
                        reward_distribution[state, action] = 1
                    else:
                        transition_probs[state, action, next_state] = 1
                else:
                    transition_probs[state, action, state] = 1

        return transition_probs, reward_distribution

    def get_plot_arrow_params(self, max_val, max_index):
        entries = len(max_val)
        x_pos = np.zeros(entries)
        y_pos = np.zeros(entries)
        x_dir = np.zeros(entries)
        y_dir = np.zeros(entries)
        color = np.zeros(entries)

        for i in range(entries):
            coords = self.state_inverse_map[i]
            if max_index[i] == 0:
                y_dir[i] = 1
            elif max_index[i] == 1:
                y_dir[i] = -1
            elif max_index[i] == 2:
                x_dir[i] = 1
            elif max_index[i] == 3:
                x_dir[i] = -1

            x_pos[i] = coords[1] + 0.5
            y_pos[i] = coords[0] + 0.5
            color[i] = max_val[i]
        return x_pos, y_pos, x_dir, y_dir, color

    def create_coords(self):
        coords = []
        for i in range(len(self.state_inverse_map)):
            if self.map[self.state_inverse_map[i]] == 0:
                continue
            else:
                coords.append(self.state_inverse_map[i])
        return coords

    def plot_params(self, option_probs, termination_probs, policy_probs, name_prefix="", no_blocks=False):
        grid = self.map if no_blocks else self.grid
        options = option_probs.shape[1]

        coords = self.create_coords()
        for i in range(options):
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 9))

            probs = policy_probs[:, i, :]
            max_val, max_index = torch.max(probs, dim=1)
            max_val = max_val.numpy()
            max_index = max_index.numpy()
            plot_policy(ax=axes[0], arrow_data=self.get_plot_arrow_params(max_val, max_index), grid=grid)

            term_probs = termination_probs[:, i]
            plot_terminations(ax=axes[1], probs=term_probs, coords=coords, grid=grid)

            opt_probs = option_probs[:, i]
            plot_options(ax=axes[2], probs=opt_probs, coords=coords, grid=grid)
            plt.tight_layout()
            plt.savefig(name_prefix + "_option{}".format(i) + ".png", bbox_inches='tight', dpi=150)


def string_to_map(map_name):
    if map_name == "grid_small":
        return FourPaths.small_map
    elif map_name == "grid_medium":
        return FourPaths.medium_map
    elif map_name == "grid_large":
        return FourPaths.large_map
    else:
        return None


def env_test():
    env = FourPaths("grid_large")
    for env_n in range(env.n_envs):
        env.set_task(env.envs_to_sample[env_n])
        env.reset()
        while True:
            _, _, done, _ = env.step(env.action_space.sample())
            env.render()
            if done:
                break
    env.close()


if __name__ == '__main__':
    env_test()
