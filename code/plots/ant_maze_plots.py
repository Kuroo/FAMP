import os

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import numpy as np
import torch
import pickle
from core.initalization import create_policy
from core.sampling import generate_samples
from envs.ml_ant_obstaclesgen import AntObstaclesGenEnv
from matplotlib import cm
from matplotlib import pyplot as plt
from plot_utils import add_alg_data_test, add_alg_data_train, plot_algs, plot_all_alg_values, COMMON_NAMES
from utils.plotting import plot_terminations
from matplotlib.colors import ListedColormap

# TODO merge this with utils file in plotting
from utils.plotting import plot_policy, plot_policy_alternative

from core.policies import LearnTermOptionsPolicy

ENV_BLOCKS = (
        [0, 1, 1,
         0, 1, 1,
         1, 1, 1], # env 0
        [1, 1, 1,
         0, 1, 1,
         0, 1, 1], # env 1
        [1, 0, 1,
         0, 0, 1,
         1, 1, 1],  # env 2
        [1, 1, 1,
         0, 0, 1,
         1, 0, 1],  # env 3
        [1, 1, 1,
         0, 0, 0,
         1, 1, 1],  # env 4
        [1, 1, 0,
         0, 0, 0,
         1, 1, 1],  # env 5
        [1, 1, 1,
         0, 0, 0,
         1, 1, 0],  # env 6
        [0, 0, 1,
         0, 1, 1,
         1, 1, 1], # env 7
        [1, 1, 1,
         0, 1, 1,
         0, 0, 1], # env 8
        [0, 0, 0,
         0, 1, 1,
         1, 1, 1],  # env 9
        [1, 1, 1,
         0, 1, 1,
         0, 0, 0],  # env 10
        [1, 0, 0,
         0, 0, 1,
         1, 1, 1],  # env 11
        [1, 1, 1,
         0, 0, 1,
         1, 0, 0],  # env 12
    )

ENV_GOALS = ((8, 24), (8, -24), (24, 24), (24, -24), (48, 0), (40, 24), (40, -24), (32, 16), (32, -16),
             (40, 24), (40, -24), (40, 24), (40, -24))
ENV_OFFSET = 52

FONTSIZE = 15
ALPHA = 0.15
EXP_DIR = "../../run_data/paper_experiments"
X_LABEL = "Episodes"

# ANT
ANT_TRAIN_ENVS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
ANT_TEST_ENVS = (9, 10, 11, 12)
ANT_Y_LABEL = "Return"

ANT_FAMP_SEEDS = (42, 142, 242, 342, 442, 542, 642, 742, 842)
ANT_FAMP_TAG = "Step00Return/Avg"
ANT_FAMP_EPU = 20

ANT_MLSH_SEEDS = (42, 43, 45, 46, 47, 48, 49, 50, 51)
ANT_MLSH_TAG = "Return"
ANT_MLSH_EPU = 2

ANT_PPO_EPU = 4
ANT_RL2_EPU = 1

XLABELS = lambda x, y: [i * y for i in range(x // y + 1)]

def get_avg_performance_algs():
    EPISODES = 200
    algs = [
        {"label": COMMON_NAMES["ours"],
         "dir": f"{EXP_DIR}/ant_maze/ours/3opts/eval",
         "tag": ANT_FAMP_TAG,
         "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
         "seeds": ANT_FAMP_SEEDS,
         "timesteps": EPISODES // ANT_FAMP_EPU + 1
         },
        {"label": COMMON_NAMES['term'],
            "dir": f"{EXP_DIR}/ant_maze/ours/3opts_tt200/eval",
            "tag": ANT_FAMP_TAG,
            "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
            "seeds": ANT_FAMP_SEEDS,
            "timesteps": EPISODES // ANT_FAMP_EPU + 1
        },
        {"label": COMMON_NAMES["mlsh"],
         "dir": f"{EXP_DIR}/ant_maze/mlsh/eval",
         "tag": ANT_MLSH_TAG,
         "x_values": XLABELS(EPISODES, ANT_MLSH_EPU),
         "seeds": ANT_MLSH_SEEDS,
         "timesteps": EPISODES // ANT_MLSH_EPU + 1
         },
        {"label": COMMON_NAMES['ppo'],
            "dir": f"{EXP_DIR}/ant_maze/ppo/eval",
            "tag": "AverageEpRet",
            "x_values": XLABELS(EPISODES, ANT_PPO_EPU),
            "seeds": (42, 142, 242, 342, 442, 542, 642, 742, 842),
            "timesteps": EPISODES // ANT_PPO_EPU + 1
        },


        # {"label": COMMON_NAMES['rl2'],
        #  "dir": f"{EXP_DIR}/ant_maze/rl2/eval",
        #  "tag": "returns",
        #  "x_values": XLABELS(EPISODES, ANT_RL2_EPU),
        #  "seeds": (42, 142, 242),
        #  "timesteps": EPISODES // ANT_RL2_EPU + 1
        #  },
        {"label": f"{COMMON_NAMES['learnall']}",
            "dir": f"{EXP_DIR}/ant_maze/ours_learn_all/3opts/ilr00001/eval",
            "tag": ANT_FAMP_TAG,
            "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
            "seeds": (42, 142, 242, 342, 442, 542, 642, 742, 842),
            "timesteps": EPISODES // ANT_FAMP_EPU + 1
        },
        # {"label": "MAML",
        #  "dir": f"{EXP_DIR}/ant_maze/maml/lr0001/eval",
        #  "tag": "returns",
        #  "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
        #  "seeds": (42, 142, 242),
        #  "timesteps": EPISODES // ANT_FAMP_EPU + 1
        #  }
        {"label": COMMON_NAMES['maml'],
         "dir": f"{EXP_DIR}/ant_maze/maml_pytorch/big_lr/eval",
         "tag": "returns",
         "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
         "seeds": (42, 142, 242, 342, 442, 542, 642, 742, 842),
         "timesteps": EPISODES // ANT_FAMP_EPU + 1
         }
    ]

    # for steps in (50, 100, 200):
    #     algs.append(
    #         {"label": f"Ours + TT{steps}",
    #          "dir": f"{EXP_DIR}/ant_maze/ours/3opts_tt{steps}/eval",
    #          "tag": ANT_FAMP_TAG,
    #          "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
    #          "seeds": ANT_FAMP_SEEDS,
    #          "timesteps": EPISODES // ANT_FAMP_EPU + 1
    #          }
    #     )


    return algs


def get_hyperparams_algs():
    EPISODES = 100
    algs = []
    for opt in (4, 2, 8, 16):
        pass
        algs.append(
            {"label": f"{opt} opts L=3",
             "dir": f"{EXP_DIR}/ant_maze/ours/{opt}opts/eval",
             "tag": ANT_FAMP_TAG,
             "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
             "seeds": ANT_FAMP_SEEDS,
             "timesteps": EPISODES // ANT_FAMP_EPU + 1
             }
        )

    for lookahead in (1, 2):
        algs.append(
            {"label": f"4 opts L={lookahead}",
             "dir": f"{EXP_DIR}/ant_maze/ours/4opts_l{lookahead}/eval",
             "tag": ANT_FAMP_TAG,
             "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
             "seeds": ANT_FAMP_SEEDS,
             "timesteps": EPISODES // ANT_FAMP_EPU + 1
             }
        )
    return algs

def get_avg_performance_algs_resets():
    EPISODES = 200
    algs = [
        {"label": COMMON_NAMES["ours"],
         "dir": f"{EXP_DIR}/ant_maze_resets/ours/eval",
         "tag": ANT_FAMP_TAG,
         "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
         "seeds": ANT_FAMP_SEEDS,
         "timesteps": EPISODES // ANT_FAMP_EPU + 1
         },
        {"label": COMMON_NAMES["mlsh"],
         "dir": f"{EXP_DIR}/ant_maze_resets/mlsh/eval",
         "tag": ANT_MLSH_TAG,
         "x_values": XLABELS(EPISODES, ANT_MLSH_EPU),
         "seeds": ANT_MLSH_SEEDS,
         "timesteps": EPISODES // ANT_MLSH_EPU + 1
         },
        {"label": COMMON_NAMES['ppo'],
            "dir": f"{EXP_DIR}/ant_maze_resets/ppo/eval",
            "tag": "AverageEpRet",
            "x_values": XLABELS(EPISODES, ANT_PPO_EPU),
            "seeds": ANT_FAMP_SEEDS,
            "timesteps": EPISODES // ANT_PPO_EPU + 1
        }
    ]

    return algs


def ant_maze_avg_train():
    algs = get_avg_performance_algs()
    add_alg_data_test(algs, ANT_TRAIN_ENVS)
    plot_name = "ant_maze_performance_avg_train"
    plot_algs(algs, 1, 1, X_LABEL, ANT_Y_LABEL, plot_name, average_plots=True, spread_type="bootstrap")
    plt.show()

def ant_maze_avg_train_resets():
    algs = get_avg_performance_algs_resets()
    add_alg_data_test(algs, ANT_TRAIN_ENVS)
    plot_name = "ant_maze_performance_avg_train_resets"
    plot_algs(algs, 1, 1, X_LABEL, ANT_Y_LABEL, plot_name, average_plots=True, spread_type="bootstrap")
    plt.show()


def ant_maze_avg_test():
    algs = get_avg_performance_algs()
    add_alg_data_test(algs, ANT_TEST_ENVS)
    plot_name = "ant_maze_performance_avg_test"
    plot_algs(algs, 1, 1, X_LABEL, ANT_Y_LABEL, plot_name, average_plots=True, spread_type="bootstrap")
    plt.show()


def ant_maze_ablation():
    EPISODES = 200
    algs = [
        {"label": COMMON_NAMES["ours"],
            "dir": f"{EXP_DIR}/ant_maze/ours/3opts/eval",
            "tag": ANT_FAMP_TAG,
            "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
            "seeds": ANT_FAMP_SEEDS,
            "timesteps": EPISODES // ANT_FAMP_EPU + 1
        },
        {"label": COMMON_NAMES["term"],
            "dir": f"{EXP_DIR}/ant_maze/ours/3opts_tt200/eval",
            "tag": ANT_FAMP_TAG,
            "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
            "seeds": ANT_FAMP_SEEDS,
            "timesteps": EPISODES // ANT_FAMP_EPU + 1
        },
        {"label": COMMON_NAMES["adaptopts"],
            "dir": f"{EXP_DIR}/ant_maze/ours_adapt_options/3opts/ilr0001/eval",
            "tag": ANT_FAMP_TAG,
            "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
            "seeds": (42, 142, 242),
            "timesteps": EPISODES // ANT_FAMP_EPU + 1
        },
        {"label": f"{COMMON_NAMES['learnall']} 0001",
            "dir": f"{EXP_DIR}/ant_maze/ours_learn_all/3opts/ilr0001/eval",
            "tag": ANT_FAMP_TAG,
            "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
            "seeds": (42, 142, 242),
            "timesteps": EPISODES // ANT_FAMP_EPU + 1
        },
        {"label": f"{COMMON_NAMES['learnall']} 00005",
            "dir": f"{EXP_DIR}/ant_maze/ours_learn_all/3opts/ilr00005/eval",
            "tag": ANT_FAMP_TAG,
            "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
            "seeds": (42, 142, 242),
            "timesteps": EPISODES // ANT_FAMP_EPU + 1
        },
        {"label": f"{COMMON_NAMES['learnall']} 00001",
            "dir": f"{EXP_DIR}/ant_maze/ours_learn_all/3opts/ilr00001/eval",
            "tag": ANT_FAMP_TAG,
            "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
            "seeds": (42, 142, 242),
            "timesteps": EPISODES // ANT_FAMP_EPU + 1
        },

        {"label": COMMON_NAMES["maml"],
            "dir": f"{EXP_DIR}/ant_maze/maml_pytorch/big_lr/eval",
            "tag": "returns",
            "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
            "seeds": (42, 142, 242),
            "timesteps": EPISODES // ANT_FAMP_EPU + 1
        }
    ]
    add_alg_data_test(algs, ANT_TEST_ENVS)
    plot_name = "ant_maze_ablation_test"
    plot_algs(algs, 1, 1, X_LABEL, ANT_Y_LABEL, plot_name, average_plots=True, spread_type="std", ylim=[-2700, 2000])
    plt.show()


def ant_maze_ft_time():
    EPISODES = 200
    algs = []

    for tt in (50, 100, 200):
        algs.append(
            {"label": f"{COMMON_NAMES['term']} {tt}",
                "dir": f"{EXP_DIR}/ant_maze/ours/3opts_tt{tt}/eval",
                "tag": ANT_FAMP_TAG,
                "x_values": XLABELS(EPISODES, ANT_FAMP_EPU),
                "seeds": ANT_FAMP_SEEDS,
                "timesteps": EPISODES // ANT_FAMP_EPU + 1
            }
        )

    add_alg_data_test(algs, ANT_TRAIN_ENVS)
    plot_name = "ant_maze_ft_time_train"
    plot_algs(algs, 1, 1, X_LABEL, ANT_Y_LABEL, plot_name, average_plots=True, spread_type="std")
    plt.show()

def ant_maze_plots():
    fig, axes = plt.subplots(ncols=2, nrows=1, dpi=1000, figsize=(14, 5))
    ax = axes[0]


    algs = get_avg_performance_algs()
    add_alg_data_test(algs, ANT_TRAIN_ENVS)
    data_func = lambda x: np.mean(x, axis=1)
    plot_all_alg_values(algs=algs, data_func=data_func, ax=ax, spread_type="std")
    ax.legend(loc="lower right", prop={'size': FONTSIZE}, ncol=2)
    ax.set_ylabel("Return", fontsize=FONTSIZE + 2)
    ax.set_xlabel("Episodes", fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE + 2)
    ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA - 0.05)

    ax = axes[1]
    algs = get_avg_performance_algs()
    add_alg_data_test(algs, ANT_TEST_ENVS)

    data_func = lambda x: np.mean(x, axis=1)
    plot_all_alg_values(algs=algs, data_func=data_func, ax=ax, spread_type="std")
    # ax.legend(loc="lower right", prop={'size': FONTSIZE}, ncol=2)
    ax.set_xlabel("Episodes", fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE + 2)
    ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA - 0.05)

    plt.savefig(f"../../plots/ant_maze_plots.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')


def ant_maze_hyperparams():
    algs = get_hyperparams_algs()
    add_alg_data_test(algs, ANT_TEST_ENVS)
    plot_name = "ant_maze_hyperparams_avg"
    plot_algs(algs, 1, 1, X_LABEL, ANT_Y_LABEL, plot_name, average_plots=True, spread_type="std")


def ant_maze_metatrain_plots():
    # Ant maze
    algs = [
        {"label": "FAMP (Ours)",
         "dir": f"../../run_data/paper_experiments/ant_maze/ours/3opts/pretrain",
         "tag": "Step02Return/Avg",
         "x_values": [i * 48 * 3 * 20 for i in range(2620)],
         "seeds": (42, 142, 242),
         "timesteps": 2620
         },
        {"label": "MLSH",
         "dir": f"../../run_data/paper_experiments/ant_maze/mlsh/pretrain",
         "tag": "Return",
         "x_values": [i * 120 * 2 for i in range(1800 * 2)],
         "seeds": (42, 43, 45),
         "timesteps": 1800
         },
        {"label": "RL2",
         "dir": f"../../run_data/paper_experiments/ant_maze/rl2/pretrain",
         "tag": "Average/AverageReturn",
         "x_values": [i * 9 * 4 for i in range(2000)],
         "seeds": (42, 142, 242),
         "timesteps": 2000
         },
        {"label": "MAML",
         "dir": f"../../run_data/paper_experiments/ant_maze/maml_pytorch/big_lr/pretrain",
         "tag": "Step1/Ret/Avg",
         "x_values": [i * 16 * 3 * 20 for i in range(260)],
         "seeds": (42, 142, 242),
         "timesteps": 260
         },
    ]

    cache_path = "../../plots/cache/ant_maze_training.pickle"
    fig, ax = plt.subplots(ncols=1, nrows=1, dpi=1000)
    if os.path.isfile(cache_path):

        with open(cache_path, 'rb') as handle:
            algs = pickle.load(handle)
    else:
        add_alg_data_train(algs)
        with open(cache_path, 'wb') as handle:
            pickle.dump(algs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    data_func = lambda x: x
    plot_all_alg_values(algs=algs, data_func=data_func, ax=ax, spread_type="std")

    ax.legend(loc="lower right", prop={'size': FONTSIZE}, ncol=2)
    ax.set_ylabel("Return", fontsize=FONTSIZE)
    ax.set_xlabel("Episodes", fontsize=FONTSIZE)
    ax.set_xscale('log')
    ax.set_xlim(1e3, 1e7)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA-0.05)
    plt.savefig(f"../../plots/ant_training.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')


def ant_maze_per_env_plots():
    ENVS = [i for i in range(13)]
    ENVS.remove(ENVS[4])
    algs = get_avg_performance_algs()
    add_alg_data_test(algs, ENVS)
    nrows = 3
    ncols = 4
    plot_name = "ant_maze_per_env_performance"
    plot_algs(algs, nrows, ncols, X_LABEL, ANT_Y_LABEL, plot_name, average_plots=False, spread_type="bootstrap", fontsize=FONTSIZE)


def ant_maze_options_plot_env_map(eval_dir, env, seed, ax, env_xpos=0, env_ypos=0):
    start_goal_size = 2
    def pltcolor(stat):
        cols = []
        for i in range(len(stat)):
            if stat[i] == 0:
                cols.append('darkorchid')
            elif stat[i] == 1:
                cols.append('aqua')
            elif stat[i] == 2:
                cols.append('mediumblue')
            else:
                raise NotImplementedError
        return cols

    with open(
            f'{eval_dir}/seed{seed}/env{env}/logs/trajs.pickle',
            'rb') as handle:
        trajs = pickle.load(handle)

    trajs_start = 0
    trajs_end = 10
    coords = np.concatenate(trajs["observations"][trajs_start:trajs_end], axis=0)[:, :2]
    options = np.concatenate(trajs["options"][trajs_start:trajs_end], axis=0)
    terminations = np.concatenate(trajs["terminations"][trajs_start:trajs_end], axis=0)
    print(np.mean(terminations))
    cols = pltcolor(options)
    # cols = pltcolor(terminations)

    condition = terminations == 1

    col_dict = {0: "yellow",
                1: "red",
                2: "green"}
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
    # a = plt.scatter(np.where(condition, coords[:,0], 0), np.where(condition, coords[:,1], 0), c=cols, s=0.01)

    squares = [(8, 16), (24, 16), (40, 16), (8, 0), (24, 0), (40, 0), (8, -16), (24, -16), (40, -16)]
    lines = []
    # Blocks
    for i in range(len(squares)):
        if not ENV_BLOCKS[env][i]: continue
        square = squares[i]
        rect = patches.Rectangle((square[0] + env_xpos * ENV_OFFSET - 7.6, square[1] + env_ypos * ENV_OFFSET - 7.6),
                                 7.6 * 2, 7.6 * 2, linewidth=1, edgecolor=(1, 0.3, 0.2), facecolor=(1, 0.3, 0.2))
        lines.append([(square[0] + env_xpos * ENV_OFFSET, square[1] + env_ypos * ENV_OFFSET - 7.6),
                      (square[0] + env_xpos * ENV_OFFSET, square[1] + env_ypos * ENV_OFFSET + 7.6)])
        lines.append([(square[0] + env_xpos * ENV_OFFSET - 7.6, square[1] + env_ypos * ENV_OFFSET),
                      (square[0] + env_xpos * ENV_OFFSET + 7.6, square[1] + env_ypos * ENV_OFFSET)])
        ax.add_patch(rect)
    lc = LineCollection(lines, color=[(1.0, 0.51, 0.50)], lw=2)
    plt.gca().add_collection(lc)

    # Border
    rect = patches.Rectangle((0 + env_xpos * ENV_OFFSET, -24 + env_ypos * ENV_OFFSET), 48, 48, linewidth=2,
                             edgecolor=(0.718, 0.596, 0.329), facecolor="none")
    ax.add_patch(rect)
    # Goal
    rect = patches.Rectangle((ENV_GOALS[env][0] + env_xpos * ENV_OFFSET - start_goal_size, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET - start_goal_size),
                             2 * start_goal_size, 2 * start_goal_size, linewidth=1, edgecolor=(0.388, 0.918, 0.192), facecolor=(0.388, 0.918, 0.192))
    # rect = patches.Rectangle((40 - 0.75, -24 - 0.75), 1.5, 1.5, linewidth=1, edgecolor="lime", facecolor="lime", zorder=20)
    ax.add_patch(rect)
    lc = LineCollection([[(ENV_GOALS[env][0] + env_xpos * ENV_OFFSET, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET - start_goal_size),
                          (ENV_GOALS[env][0] + env_xpos * ENV_OFFSET, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET + start_goal_size)],
                         [(ENV_GOALS[env][0] + env_xpos * ENV_OFFSET - start_goal_size, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET),
                          (ENV_GOALS[env][0] + env_xpos * ENV_OFFSET + start_goal_size, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET)]],
                        color=[(0.463, 1.0, 0.412)], lw=2)

    # Start
    rect = patches.Rectangle(
        (2 + env_xpos * ENV_OFFSET - start_goal_size, 0 + env_ypos * ENV_OFFSET - start_goal_size),
        2 * start_goal_size, 2 * start_goal_size, linewidth=1, edgecolor='yellow', facecolor='yellow')
    ax.add_patch(rect)
    # rect = patches.Rectangle((40 - 0.75, -24 - 0.75), 1.5, 1.5, linewidth=1, edgecolor="lime", facecolor="lime", zorder=20)
    # lc = LineCollection([[(ENV_GOALS[env][0] + env_xpos * ENV_OFFSET, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET - 0.75),
    #                       (
    #                       ENV_GOALS[env][0] + env_xpos * ENV_OFFSET, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET + 0.75)],
    #                      [(ENV_GOALS[env][0] + env_xpos * ENV_OFFSET - 0.75, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET),
    #                       (ENV_GOALS[env][0] + env_xpos * ENV_OFFSET + 0.75,
    #                        ENV_GOALS[env][1] + env_ypos * ENV_OFFSET)]],
    #                     color=[(0.463, 1.0, 0.412)], lw=2)


    plt.gca().add_collection(lc)

    ax.set_aspect('equal')
    # a = ax.scatter(coords[:, 0] + env_xpos * ENV_OFFSET, coords[:, 1] + env_ypos * ENV_OFFSET, c=cols, s=0.01, zorder=50)


def ant_maze_options_map(envs=list(range(13)), seed=142,
                     eval_dir="/home/david/Projects/master_thesis/run_data/paper_experiments/ant_maze/ours/3opts/eval",
                     rows=2):
    envs.remove(4)
    fig, ax = plt.subplots()
    cols = len(envs) // rows + len(envs) % rows
    # checkpoint = torch.load("/home/david/Desktop/Papers/test/epoch0700.tar")
    # checkpoint['pytorch_rng_states'] = [checkpoint['pytorch_rng_states'][7]]
    # checkpoint['numpy_rng_states'] = [checkpoint['numpy_rng_states'][7]]
    # torch.save(checkpoint, "process7.tar")
    # for env in envs:
    for i, env in enumerate(envs):
        y_pos = (i + 1) % rows
        x_pos = i // rows
        ant_maze_options_plot_env_map(eval_dir=eval_dir, env=env, seed=seed, ax=ax, env_xpos=x_pos, env_ypos=y_pos)
    fig.patch.set_facecolor((0.235, 0.255, 0.235))
    plt.xlim(0 -2, cols * ENV_OFFSET -2)
    plt.ylim(-24-2, -24 + rows * ENV_OFFSET -2)
    plt.axis('off')
    plt.savefig(f"../../plots/ant_maze_map.pdf", format='pdf', dpi=3000, pad_inches=0, bbox_inches='tight')
    plt.show()

def ant_maze_options_plot_env(eval_dir, env, seed, ax, env_xpos=0, env_ypos=0):
    start_goal_size = 0.75
    def pltcolor(stat):
        cols = []
        for i in range(len(stat)):
            if stat[i] == 0:
                cols.append('darkorchid')
            elif stat[i] == 1:
                cols.append('aqua')
            elif stat[i] == 2:
                cols.append('mediumblue')
            else:
                raise NotImplementedError
        return cols

    with open(
            f'{eval_dir}/seed{seed}/env{env}/logs/trajs.pickle',
            'rb') as handle:
        trajs = pickle.load(handle)

    trajs_start = 0
    trajs_end = 10
    coords = np.concatenate(trajs["observations"][trajs_start:trajs_end], axis=0)[:, :2]
    options = np.concatenate(trajs["options"][trajs_start:trajs_end], axis=0)
    terminations = np.concatenate(trajs["terminations"][trajs_start:trajs_end], axis=0)
    print(np.mean(terminations))
    cols = pltcolor(options)
    # cols = pltcolor(terminations)

    condition = terminations == 1

    col_dict = {0: "yellow",
                1: "red",
                2: "green"}
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
    # a = plt.scatter(np.where(condition, coords[:,0], 0), np.where(condition, coords[:,1], 0), c=cols, s=0.01)

    squares = [(8, 16), (24, 16), (40, 16), (8, 0), (24, 0), (40, 0), (8, -16), (24, -16), (40, -16)]
    lines = []
    # Blocks
    for i in range(len(squares)):
        if not ENV_BLOCKS[env][i]: continue
        square = squares[i]
        rect = patches.Rectangle((square[0] + env_xpos * ENV_OFFSET - 7.6, square[1] + env_ypos * ENV_OFFSET - 7.6),
                                 7.6 * 2, 7.6 * 2, linewidth=1, edgecolor=(1, 0.3, 0.2), facecolor=(1, 0.3, 0.2))
        lines.append([(square[0] + env_xpos * ENV_OFFSET, square[1] + env_ypos * ENV_OFFSET - 7.6),
                      (square[0] + env_xpos * ENV_OFFSET, square[1] + env_ypos * ENV_OFFSET + 7.6)])
        lines.append([(square[0] + env_xpos * ENV_OFFSET - 7.6, square[1] + env_ypos * ENV_OFFSET),
                      (square[0] + env_xpos * ENV_OFFSET + 7.6, square[1] + env_ypos * ENV_OFFSET)])
        ax.add_patch(rect)
    lc = LineCollection(lines, color=[(1.0, 0.51, 0.50)], lw=2)
    plt.gca().add_collection(lc)

    # Border
    rect = patches.Rectangle((0 + env_xpos * ENV_OFFSET, -24 + env_ypos * ENV_OFFSET), 48, 48, linewidth=2,
                             edgecolor=(0.718, 0.596, 0.329), facecolor="none")
    ax.add_patch(rect)
    # Goal
    rect = patches.Rectangle((ENV_GOALS[env][0] + env_xpos * ENV_OFFSET - start_goal_size, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET - start_goal_size),
                             2 * start_goal_size, 2 * start_goal_size, linewidth=1, edgecolor=(0.388, 0.918, 0.192), facecolor=(0.388, 0.918, 0.192))
    # rect = patches.Rectangle((40 - 0.75, -24 - 0.75), 1.5, 1.5, linewidth=1, edgecolor="lime", facecolor="lime", zorder=20)
    ax.add_patch(rect)
    lc = LineCollection([[(ENV_GOALS[env][0] + env_xpos * ENV_OFFSET, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET - start_goal_size),
                          (ENV_GOALS[env][0] + env_xpos * ENV_OFFSET, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET + start_goal_size)],
                         [(ENV_GOALS[env][0] + env_xpos * ENV_OFFSET - start_goal_size, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET),
                          (ENV_GOALS[env][0] + env_xpos * ENV_OFFSET + start_goal_size, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET)]],
                        color=[(0.463, 1.0, 0.412)], lw=2)

    # Start
    rect = patches.Rectangle(
        (2 + env_xpos * ENV_OFFSET - start_goal_size, 0 + env_ypos * ENV_OFFSET - start_goal_size),
        2 * start_goal_size, 2 * start_goal_size, linewidth=1, edgecolor='yellow', facecolor='yellow')
    ax.add_patch(rect)
    # rect = patches.Rectangle((40 - 0.75, -24 - 0.75), 1.5, 1.5, linewidth=1, edgecolor="lime", facecolor="lime", zorder=20)
    # lc = LineCollection([[(ENV_GOALS[env][0] + env_xpos * ENV_OFFSET, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET - 0.75),
    #                       (
    #                       ENV_GOALS[env][0] + env_xpos * ENV_OFFSET, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET + 0.75)],
    #                      [(ENV_GOALS[env][0] + env_xpos * ENV_OFFSET - 0.75, ENV_GOALS[env][1] + env_ypos * ENV_OFFSET),
    #                       (ENV_GOALS[env][0] + env_xpos * ENV_OFFSET + 0.75,
    #                        ENV_GOALS[env][1] + env_ypos * ENV_OFFSET)]],
    #                     color=[(0.463, 1.0, 0.412)], lw=2)


    plt.gca().add_collection(lc)

    ax.set_aspect('equal')
    a = ax.scatter(coords[:, 0] + env_xpos * ENV_OFFSET, coords[:, 1] + env_ypos * ENV_OFFSET, c=cols, s=0.01, zorder=50)


def ant_maze_options(envs=tuple(range(13)), seed=642,
                     eval_dir="/home/david/Projects/master_thesis/run_data/paper_experiments/ant_maze/ours/3opts/eval",
                     rows=2):
    fig, ax = plt.subplots()
    cols = len(envs) // rows + len(envs) % rows
    # checkpoint = torch.load("/home/david/Desktop/Papers/test/epoch0700.tar")
    # checkpoint['pytorch_rng_states'] = [checkpoint['pytorch_rng_states'][7]]
    # checkpoint['numpy_rng_states'] = [checkpoint['numpy_rng_states'][7]]
    # torch.save(checkpoint, "process7.tar")
    # for env in envs:
    for i, env in enumerate(envs):
        y_pos = (i + 1) % rows
        x_pos = i // rows
        ant_maze_options_plot_env(eval_dir=eval_dir, env=env, seed=seed, ax=ax, env_xpos=x_pos, env_ypos=y_pos)
    fig.patch.set_facecolor((0.235, 0.255, 0.235))
    plt.xlim(0-2, cols * ENV_OFFSET -2)
    plt.ylim(-24-2, -24 + rows * ENV_OFFSET -2)
    plt.axis('off')
    plt.savefig(f"../../plots/ant_maze_options_{seed}.pdf", format='pdf', dpi=3000, pad_inches=0, bbox_inches='tight')
    plt.show()

# def ant_maze_options_paper(envs=(0, 1, 3, 4, 6, 11, 10, 12), seed=142,
def ant_maze_options_paper(envs=tuple(range(13)), seed=142,
                     eval_dir="/home/david/Projects/master_thesis/run_data/paper_experiments/ant_maze/ours/3opts/eval",
                     rows=2):
    fig, ax = plt.subplots()
    cols = len(envs) // rows + len(envs) % rows
    # checkpoint = torch.load("/home/david/Desktop/Papers/test/epoch0700.tar")
    # checkpoint['pytorch_rng_states'] = [checkpoint['pytorch_rng_states'][7]]
    # checkpoint['numpy_rng_states'] = [checkpoint['numpy_rng_states'][7]]
    # torch.save(checkpoint, "process7.tar")
    # for env in envs:
    for i, env in enumerate(envs):
        y_pos = (i + 1) % rows
        x_pos = i // rows
        ant_maze_options_plot_env(eval_dir=eval_dir, env=env, seed=seed, ax=ax, env_xpos=x_pos, env_ypos=y_pos)
    fig.patch.set_facecolor((0.235, 0.255, 0.235))
    plt.xlim(0 -2, cols * ENV_OFFSET -2)
    plt.ylim(-24-2, -24 + rows * ENV_OFFSET -2)
    plt.axis('off')
    plt.savefig(f"../../plots/ant_maze_options.pdf", format='pdf', dpi=600, pad_inches=0, bbox_inches='tight')
    plt.show()

def training_plots():
    fig, axes = plt.subplots(ncols=2, nrows=1, dpi=1000, figsize=(14, 5))
    ax = axes[1]

    # Ant maze
    algs = [
        {"label": "FAMP (Ours)",
         "dir": f"../../run_data/paper_experiments/ant_maze/ours/3opts/pretrain",
         "tag": "Step02Return/Avg",
         "x_values": [i * 48 * 3 * 20 for i in range(2540)],
         "seeds": ANT_FAMP_SEEDS,
         "timesteps": 2540
         },
        {"label": "MLSH",
         "dir": f"../../run_data/paper_experiments/ant_maze/mlsh/pretrain",
         "tag": "Return",
         "x_values": [i * 120 * 2 for i in range(1800 * 2)],
         "seeds": ANT_MLSH_SEEDS,
         "timesteps": 1800
         },
        {"label": "RL2",
         "dir": f"../../run_data/paper_experiments/ant_maze/rl2/pretrain",
         "tag": "Average/AverageReturn",
         "x_values": [i * 9 * 4 for i in range(2000)],
         "seeds": (42, 142, 242),
         "timesteps": 2000
         },
        {"label": "MAML",
         "dir": f"../../run_data/paper_experiments/ant_maze/maml_pytorch/big_lr/pretrain",
         "tag": "Step1/Ret/Avg",
         "x_values": [i * 16 * 3 * 20 for i in range(320)],
         "seeds": (42, 142, 242, 342, 442, 542, 642, 742, 842),
         "timesteps": 320
         },
    ]

    cache_path = "../../plots/cache/ant_maze_training.pickle"
    if os.path.isfile(cache_path):

        with open(cache_path, 'rb') as handle:
            algs = pickle.load(handle)
    else:
        add_alg_data_train(algs)
        with open(cache_path, 'wb') as handle:
            pickle.dump(algs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    data_func = lambda x: x
    plot_all_alg_values(algs=algs, data_func=data_func, ax=ax, spread_type="confidence_t")

    ax.legend(loc="lower right", prop={'size': FONTSIZE}, ncol=2)

    ax.set_xlabel("Episodes", fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE + 2)
    ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA - 0.05)
    ax.set_xscale('log')
    ax.set_xlim(1e3, 1e7)


    ax = axes[0]



    FAOPG_METRIC = "Step00DiscountedReturn/Avg"
    data_type = "Return"
    # Ant maze
    algs = []
    for opt in (4,):
        pass
        algs.append(
            {"label": f"FAMP (Ours)",
             "dir": f"../../run_data/paper_experiments/taxi/ours/4opts/train",
             "tag": f"Step03{data_type}/Avg",
             "x_values": [i * 64 * 10 * 4 for i in range(2000)],
             "seeds": (42, 142, 242, 342, 442, 542, 642, 742, 842),
             "timesteps": 2000
             }
        )
    algs.extend([
        {"label": "Multi-task",
            "dir": f"../../run_data/paper_experiments/taxi/multi/4opts/train",
            "tag": f"Step00{data_type}/Avg",
            "x_values": [i * 48 * 10 for i in range(1550)],
            "seeds": (42, 142, 242, 342, 442, 542, 642, 742, 842),
            "timesteps": 1550
        },
        {"label": "MLSH 2 steps",
            "dir": f"../../run_data/paper_experiments/taxi/mlsh/macro2/train",
            "tag": data_type,
            "x_values": [i * 120 * 2 for i in range(2250)],
            "seeds": (42, 45, 47, 50, 53, 56, 57, 60, 61),
            "timesteps": 2250
        }


        # {"label": "MLSH 4 steps",
        #  "dir": f"../../run_data/paper_experiments/taxi/mlsh/macro4/train",
        #  "tag": data_type,
        #  "x_values": [i * 120 * 2 for i in range(2500)],
        #  "seeds": (42, 43, 44, 45, 46),
        #  "timesteps": 2500
        #  },

        # {"label": "MLSH 10 steps",
        #  "dir": f"../../run_data/paper_experiments/taxi/mlsh/macro10/train",
        #  "tag": data_type,
        #  "x_values": [i * 120 * 2 for i in range(2500)],
        #  "seeds": (42, 43, 44, 45, 46),
        #  "timesteps": 2500
        #  }
    ])
    from plot_utils import get_tf_data
    #     else:
    #         data = np.stack([get_csv_data(f"{algs[a]['dir']}/seed{seed}/logs/", timesteps=algs[a]["timesteps"], tag=algs[a]["tag"]) for seed in algs[a]['seeds']])
    #         algs[a]["data"] = data
    cache_path = "../../plots/cache/taxi_training.pickle"
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as handle:
            algs = pickle.load(handle)
    else:
        add_alg_data_train(algs)
        with open(cache_path, 'wb') as handle:
            pickle.dump(algs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # algs.remove(algs[-2])

    algs[-1]["label"] = "MLSH"
    for a in range(len(algs)):
        # if algs[a]["label"] == "MLSH":
        #     data = np.stack(
        #         [get_tf_data(f"{algs[a]['dir']}/TaxiAgent4_{seed}/", tag=algs[a]["tag"], timesteps=algs[a]["timesteps"])
        #          for seed in algs[a]['seeds']])
        #     algs[a]["data"] = data
        if algs[a]["label"] == "MLSH":
            data = np.stack([get_tf_data(f"{algs[a]['dir']}/TaxiAgent2_{seed}/", tag=algs[a]["tag"],
                                         timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = data
    data_func = lambda x: x
    plot_all_alg_values(algs=algs, data_func=data_func, ax=ax, spread_type="confidence_t")

    # for a in range(len(algs)):
    #     data = algs[a]["data"]
    #     mean, std = np.mean(data, axis=0), np.std(data, axis=0)
    #     if "x_values" not in algs[a]:
    #         ax.plot(mean, label=algs[a]["label"])
    #         ax.fill_between(range(mean.shape[0]), mean+std, mean-std, alpha=ALPHA)
    #     else:
    #         ax.plot(algs[a]["x_values"], mean, label=algs[a]["label"])
    #         ax.fill_between(algs[a]["x_values"], mean + std, mean - std, alpha=ALPHA)
    ax.legend(loc="lower right", prop={'size': FONTSIZE}, ncol=2)
    ax.set_xlim(0, 4e5)
    ax.set_xlabel("Episodes", fontsize=FONTSIZE + 2)
    ax.set_ylabel("Return", fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE + 2)
    ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA - 0.05)
    plt.savefig(f"../../plots/training.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')

if __name__ == '__main__':
    nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": FONTSIZE,
        "font.size": FONTSIZE,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
    }
    mpl.rcParams.update(nice_fonts)
    # ant_maze_avg_train()
    # ant_maze_avg_test()
    # ant_maze_hyperparams()
    # ant_maze_ablation()
    # ant_maze_per_env_plots()
    # ant_maze_traj()
    # ant_maze_plots()
    # ant_maze_options_map()
    # ant_maze_options_paper()
    # ant_maze_options()
    # ant_maze_metatrain_plots()
    # training_plots()
    # ant_maze_ablation()
    # ant_maze_ft_time()
    ant_maze_avg_train_resets()
