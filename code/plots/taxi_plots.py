import os

import matplotlib as mpl
import matplotlib.patches as patches
import numpy as np
import torch
import pickle
from core.initalization import create_policy
from core.sampling import generate_samples
from envs.taxi import Taxi
from matplotlib import cm
from matplotlib import pyplot as plt
from plot_utils import add_alg_data_test, add_alg_data_train, plot_algs, create_grid_plot, plot_all_alg_values, get_tf_data, COMMON_NAMES, plot_policy
from utils.plotting import plot_terminations

# TODO merge this with utils file in plotting

from core.policies import LearnTermOptionsPolicy

FONTSIZE = 15
ALPHA = 0.15
EXP_DIR = "../../run_data/paper_experiments"
X_LABEL = "Episodes"

# TAXI
TAXI_TEST_ENVS = (0, 4, 8, 13, 17, 21, 26, 30, 34, 39, 43, 47)
TAXI_Y_LABEL = "Discounted Return"

TAXI_FAMP_SEEDS = (42, 142, 242, 342, 442, 542, 642, 742, 842)
TAXI_FAMP_TAG = "Step00DiscountedReturn/Avg"
TAXI_FAMP_EPU = 10
TAXI_FAMP_XLABELS = lambda x: [i * TAXI_FAMP_EPU for i in range(x // TAXI_FAMP_EPU + 1)]

TAXI_MLSH_SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50)
TAXI_MLSH_TAG = "DiscountedReturn"
TAXI_MLSH_EPU = 2
TAXI_MLSH_XLABELS = lambda x: [i * TAXI_MLSH_EPU for i in range(x // TAXI_MLSH_EPU + 1)]


def get_avg_performance_algs_no_collapsed_seed():
    EPISODES = 200
    algs = []

    for opt in (4,):
        algs.append(
            {"label": COMMON_NAMES["ours"],
             "dir": f"{EXP_DIR}/taxi/ours/{opt}opts/eval",
             "tag": TAXI_FAMP_TAG,
             "x_values": TAXI_FAMP_XLABELS(EPISODES),
             "seeds": TAXI_FAMP_SEEDS,
             "timesteps": EPISODES // TAXI_FAMP_EPU + 1
             }
        )

    for steps in (4,):
        algs.append(
            {"label": f'{COMMON_NAMES["term"]}',
             "dir": f"{EXP_DIR}/taxi/ours/4opts_tt{steps}/eval",
             "tag": TAXI_FAMP_TAG,
             "x_values": TAXI_FAMP_XLABELS(EPISODES),
             "seeds": (42, 142, 242, 342),
             "timesteps": EPISODES // TAXI_FAMP_EPU + 1
             }
        )

    for steps in (2, ):
        algs.append(
            {"label": f'{COMMON_NAMES["mlsh"]}',
             "dir": f"{EXP_DIR}/taxi/mlsh/macro{steps}/eval",
             "tag": TAXI_MLSH_TAG,
             "x_values": TAXI_MLSH_XLABELS(EPISODES),
             "seeds": (42, 45, 47, 50, 53),
             "timesteps": EPISODES // TAXI_MLSH_EPU + 1
             }
        )

    algs.extend([
        {"label": COMMON_NAMES["single"],
         "dir": f"{EXP_DIR}/taxi/single/4opts/eval",
         "tag": TAXI_FAMP_TAG,
         "x_values": TAXI_FAMP_XLABELS(EPISODES),
         "seeds": TAXI_FAMP_SEEDS,
         "timesteps": EPISODES // TAXI_FAMP_EPU + 1
         },
        {"label": f"Learn all",
            "dir": f"{EXP_DIR}/taxi/ours_learn_all/4opts/ilr20/eval",
            "tag": TAXI_FAMP_TAG,
            "x_values": TAXI_FAMP_XLABELS(EPISODES),
            "seeds": TAXI_FAMP_SEEDS,
            "timesteps": EPISODES // TAXI_FAMP_EPU + 1
        },
        {"label": COMMON_NAMES["multi"],
         "dir": f"{EXP_DIR}/taxi/multi_adapt/4opts/eval",
         "tag": TAXI_FAMP_TAG,
         "x_values": TAXI_FAMP_XLABELS(EPISODES),
         "seeds": TAXI_FAMP_SEEDS,
         "timesteps": EPISODES // TAXI_FAMP_EPU + 1
         },
    ])



    return algs

def get_avg_performance_algs():
    EPISODES = 200
    algs = []

    for opt in (4,):
        algs.append(
            {"label": COMMON_NAMES["ours"],
                "dir": f"{EXP_DIR}/taxi/ours/{opt}opts/eval",
                "tag": TAXI_FAMP_TAG,
                "x_values": TAXI_FAMP_XLABELS(EPISODES),
                "seeds": TAXI_FAMP_SEEDS,
                "timesteps": EPISODES // TAXI_FAMP_EPU + 1
            }
        )

    algs.append(
            {"label": f'{COMMON_NAMES["term"]}',
             "dir": f"{EXP_DIR}/taxi/ours/4opts_tt2/eval",
             "tag": TAXI_FAMP_TAG,
             "x_values": TAXI_FAMP_XLABELS(EPISODES),
             "seeds": TAXI_FAMP_SEEDS,
             "timesteps": EPISODES // TAXI_FAMP_EPU + 1
             }
    )

    algs.append(
            {"label": f'{COMMON_NAMES["mlsh"]}',
             "dir": f"{EXP_DIR}/taxi/mlsh/macro2/eval",
             "tag": TAXI_MLSH_TAG,
             "x_values": TAXI_MLSH_XLABELS(EPISODES),
             "seeds": (42, 45, 47, 50, 53, 56, 57, 60, 61),
             "timesteps": EPISODES // TAXI_MLSH_EPU + 1
             }
    )



    algs.extend([
        {"label": COMMON_NAMES["single"],
         "dir": f"{EXP_DIR}/taxi/single/4opts/eval",
         "tag": TAXI_FAMP_TAG,
         "x_values": TAXI_FAMP_XLABELS(EPISODES),
         "seeds": TAXI_FAMP_SEEDS,
         "timesteps": EPISODES // TAXI_FAMP_EPU + 1
         },
        {"label": f"Learn all",
            "dir": f"{EXP_DIR}/taxi/ours_learn_all/4opts/ilr20/eval",
            "tag": TAXI_FAMP_TAG,
            "x_values": TAXI_FAMP_XLABELS(EPISODES),
            "seeds": TAXI_FAMP_SEEDS,
            "timesteps": EPISODES // TAXI_FAMP_EPU + 1
        },
        {"label": COMMON_NAMES["multi"],
         "dir": f"{EXP_DIR}/taxi/multi_adapt/4opts/eval",
         "tag": TAXI_FAMP_TAG,
         "x_values": TAXI_FAMP_XLABELS(EPISODES),
         "seeds": TAXI_FAMP_SEEDS,
         "timesteps": EPISODES // TAXI_FAMP_EPU + 1
         },
    ])

    algs.append(
        {"label": COMMON_NAMES["nohierarchy"],
            "dir": f"{EXP_DIR}/taxi/ours_no_hierarchy/eval",
            "tag": TAXI_FAMP_TAG,
            "x_values": TAXI_FAMP_XLABELS(EPISODES),
            "seeds": TAXI_FAMP_SEEDS,
            "timesteps": EPISODES // TAXI_FAMP_EPU + 1
        },
    )

    algs.append(
        {"label": COMMON_NAMES["learninit"],
            "dir": f"{EXP_DIR}/taxi/ours_learn_init/4opts/eval",
            "tag": TAXI_FAMP_TAG,
            "x_values": TAXI_FAMP_XLABELS(EPISODES),
            "seeds": TAXI_FAMP_SEEDS,
            "timesteps": EPISODES // TAXI_FAMP_EPU + 1
        },
    )

    return algs


def get_hyperparams_algs():
    EPISODES = 100
    algs = []
    algs.append(
        {"label": f"4 opts L=3",
            "dir": f"{EXP_DIR}/taxi/ours/4opts/eval",
            "tag": TAXI_FAMP_TAG,
            "x_values": TAXI_FAMP_XLABELS(EPISODES),
            "seeds": TAXI_FAMP_SEEDS,
            "timesteps": EPISODES // TAXI_FAMP_EPU + 1
        }
    )

    for lookahead in (2, 1):
        algs.append(
            {"label": f"4 opts L={lookahead}",
             "dir": f"{EXP_DIR}/taxi/ours/4opts_l{lookahead}/eval",
             "tag": TAXI_FAMP_TAG,
             "x_values": TAXI_FAMP_XLABELS(EPISODES),
             "seeds": TAXI_FAMP_SEEDS,
             "timesteps": EPISODES // TAXI_FAMP_EPU + 1
             }
        )

    for opt in (2, 8, 16):
        pass
        algs.append(
            {"label": f"{opt} opts L=3",
             "dir": f"{EXP_DIR}/taxi/ours/{opt}opts/eval",
             "tag": TAXI_FAMP_TAG,
             "x_values": TAXI_FAMP_XLABELS(EPISODES),
             "seeds": TAXI_FAMP_SEEDS,
             "timesteps": EPISODES // TAXI_FAMP_EPU + 1
             }
        )
    return algs

def get_hyperparams_algs_no_collapsed_seed():
    EPISODES = 100
    algs = []
    algs.append(
        {"label": f"4 opts L=3",
            "dir": f"{EXP_DIR}/taxi/ours/4opts/eval",
            "tag": TAXI_FAMP_TAG,
            "x_values": TAXI_FAMP_XLABELS(EPISODES),
            "seeds": TAXI_FAMP_SEEDS,
            "timesteps": EPISODES // TAXI_FAMP_EPU + 1
        }
    )

    algs.append(
        {"label": f"4 opts L=2",
            "dir": f"{EXP_DIR}/taxi/ours/4opts_l2/eval",
            "tag": TAXI_FAMP_TAG,
            "x_values": TAXI_FAMP_XLABELS(EPISODES),
            "seeds": (42, 142, 242, 442),
            "timesteps": EPISODES // TAXI_FAMP_EPU + 1
        }
    )

    # for lookahead in (2, 1):
    for lookahead in (1,):
        algs.append(
            {"label": f"4 opts L={lookahead}",
             "dir": f"{EXP_DIR}/taxi/ours/4opts_l{lookahead}/eval",
             "tag": TAXI_FAMP_TAG,
             "x_values": TAXI_FAMP_XLABELS(EPISODES),
             "seeds": TAXI_FAMP_SEEDS,
             "timesteps": EPISODES // TAXI_FAMP_EPU + 1
             }
        )

    for opt in (2, 8, 16):
        algs.append(
            {"label": f"{opt} opts L=3",
             "dir": f"{EXP_DIR}/taxi/ours/{opt}opts/eval",
             "tag": TAXI_FAMP_TAG,
             "x_values": TAXI_FAMP_XLABELS(EPISODES),
             "seeds": TAXI_FAMP_SEEDS,
             "timesteps": EPISODES // TAXI_FAMP_EPU + 1
             }
        )
    return algs

def taxi_avg_no_collapsed_seed():
    algs = get_avg_performance_algs_no_collapsed_seed()
    add_alg_data_test(algs, TAXI_TEST_ENVS)
    plot_name = "taxi_performance_avg_no_collapsed_seed"
    plot_algs(algs, 1, 1, X_LABEL, TAXI_Y_LABEL, plot_name, average_plots=True, spread_type="std")
    plt.show()


def taxi_avg():
    algs = get_avg_performance_algs()
    add_alg_data_test(algs, TAXI_TEST_ENVS)
    # plot_name = "taxi_present_ours_mlsh_single_multi"
    plot_name = "taxi_performance_avg"
    plot_algs(algs, 1, 1, X_LABEL, TAXI_Y_LABEL, plot_name, average_plots=True, spread_type="bootstrap")
    plt.show()

def taxi_avg_zoom():
    algs = get_avg_performance_algs()
    add_alg_data_test(algs, TAXI_TEST_ENVS)
    plot_name = "taxi_performance_avg_zoom"
    plot_algs(algs, 1, 1, X_LABEL, TAXI_Y_LABEL, plot_name, average_plots=True, spread_type="bootstrap", ylim=[-0.62,-0.3], legend=False)
    plt.show()


def taxi_plots():
    cache_path = "../../plots/cache/taxi_avg_performance_algs.pickle"
    fig, axes = plt.subplots(ncols=2, nrows=1, dpi=1000, figsize=(14, 5))
    ax = axes[0]

    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as handle:
            algs = pickle.load(handle)
    else:
        algs = get_avg_performance_algs()
        add_alg_data_test(algs, TAXI_TEST_ENVS)
        with open(cache_path, 'wb') as handle:
            pickle.dump(algs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    data_func = lambda x: np.mean(x, axis=1)
    plot_all_alg_values(algs=algs, data_func=data_func, ax=ax, spread_type="std")
    ax.legend(loc="lower right", prop={'size': FONTSIZE}, ncol=2)
    ax.set_ylabel("Discounted Return", fontsize=FONTSIZE + 2)
    ax.set_xlabel("Episodes", fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE + 2)
    ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA - 0.05)

    cache_path = "../../plots/cache/taxi_hyperparams_algs.pickle"
    ax = axes[1]
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as handle:
            algs = pickle.load(handle)
    else:
        algs = get_hyperparams_algs()
        add_alg_data_test(algs, TAXI_TEST_ENVS)
        with open(cache_path, 'wb') as handle:
            pickle.dump(algs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    data_func = lambda x: np.mean(x, axis=1)
    plot_all_alg_values(algs=algs, data_func=data_func, ax=ax, spread_type="std")
    ax.legend(loc="lower right", prop={'size': FONTSIZE}, ncol=2)
    # ax.set_ylabel("Discounted Return", fontsize=FONTSIZE)
    ax.set_xlabel("Episodes", fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE + 2)
    ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA - 0.05)

    plt.savefig(f"../../plots/taxi_plots.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')


def taxi_hyperparams():
    algs = get_hyperparams_algs()
    add_alg_data_test(algs, TAXI_TEST_ENVS)
    plot_name = "taxi_hyperparams_avg"
    plot_algs(algs, 1, 1, X_LABEL, TAXI_Y_LABEL, plot_name, average_plots=True, spread_type="bootstrap",
        reduce_alpha=0.05)


def taxi_hyperparams_no_collapsed_seed():
    algs = get_hyperparams_algs_no_collapsed_seed()
    add_alg_data_test(algs, TAXI_TEST_ENVS)
    plot_name = "taxi_hyperparams_avg_no_collapsed_seed"
    plot_algs(algs, 1, 1, X_LABEL, TAXI_Y_LABEL, plot_name, average_plots=True, spread_type="std")



def taxi_ablation():
    EPISODES = 100
    algs = []

    for opt in (4,):
        algs.append(
            {"label": COMMON_NAMES["ours"],
             "dir": f"{EXP_DIR}/taxi/ours/{opt}opts/eval",
             "tag": TAXI_FAMP_TAG,
             "x_values": TAXI_FAMP_XLABELS(EPISODES),
             "seeds": TAXI_FAMP_SEEDS,
             "timesteps": EPISODES // TAXI_FAMP_EPU + 1
             }
        )
    for tt in (4, 7, 10):
        algs.append(
            {"label": f"{COMMON_NAMES['term']} {tt}",
             "dir": f"{EXP_DIR}/taxi/ours/4opts_tt{tt}/eval",
             "tag": TAXI_FAMP_TAG,
             "x_values": TAXI_FAMP_XLABELS(EPISODES),
             "seeds": TAXI_FAMP_SEEDS,
             "timesteps": EPISODES // TAXI_FAMP_EPU + 1
             }
        )
    algs.append(
        {"label": COMMON_NAMES["learnall"],
            "dir": f"{EXP_DIR}/taxi/ours_learn_all/4opts/ilr10/eval",
            "tag": TAXI_FAMP_TAG,
            "x_values": TAXI_FAMP_XLABELS(EPISODES),
            "seeds": TAXI_FAMP_SEEDS,
            "timesteps": EPISODES // TAXI_FAMP_EPU + 1
        }
    )
    algs.append(
        {"label": COMMON_NAMES["adaptopts"],
            "dir": f"{EXP_DIR}/taxi/ours_adapt_options/4opts/ilr10/eval",
            "tag": TAXI_FAMP_TAG,
            "x_values": TAXI_FAMP_XLABELS(EPISODES),
            "seeds": TAXI_FAMP_SEEDS,
            "timesteps": EPISODES // TAXI_FAMP_EPU + 1
        }
    )

    add_alg_data_test(algs, TAXI_TEST_ENVS)
    plot_name = "taxi_ablation"
    plot_algs(algs, 1, 1, X_LABEL, TAXI_Y_LABEL, plot_name, average_plots=True, spread_type="bootstrap")
    plt.show()

def taxi_ft_time():
    EPISODES = 100
    algs = []


    for tt in (2, 4, 10):
        algs.append(
            {"label": f"{COMMON_NAMES['term']} {tt}",
             "dir": f"{EXP_DIR}/taxi/ours/4opts_tt{tt}/eval",
             "tag": TAXI_FAMP_TAG,
             "x_values": TAXI_FAMP_XLABELS(EPISODES),
             "seeds": TAXI_FAMP_SEEDS,
             "timesteps": EPISODES // TAXI_FAMP_EPU + 1
             }
        )

    for tt in (2, 4, 10):
        algs.append(
                {"label": f'{COMMON_NAMES["mlsh"]} {tt}',
                    "dir": f"{EXP_DIR}/taxi/mlsh/macro{tt}/eval",
                    "tag": TAXI_MLSH_TAG,
                    "x_values": TAXI_MLSH_XLABELS(EPISODES),
                    "seeds": (42, 45, 47, 50, 53, 56, 57, 60, 61) if tt == 2 else TAXI_MLSH_SEEDS,
                    "timesteps": EPISODES // TAXI_MLSH_EPU + 1
                }
        )
    add_alg_data_test(algs, TAXI_TEST_ENVS)

    plot_name = "taxi_ft_time"
    plot_algs(algs, 1, 1, X_LABEL, TAXI_Y_LABEL, plot_name, average_plots=True, spread_type="bootstrap")
    plt.show()

def taxi_ablation_param_groups_hyperparams():
    EPISODES = 100
    algs = []

    algs.append(
        {"label": f"Adapt Options ILR10",
         "dir": f"{EXP_DIR}/taxi/ours_adapt_options/4opts/ilr10/eval",
         "tag": TAXI_FAMP_TAG,
         "x_values": TAXI_FAMP_XLABELS(EPISODES),
         "seeds": TAXI_FAMP_SEEDS,
         "timesteps": EPISODES // TAXI_FAMP_EPU + 1
         }
    )
    for ilr in (1, 5, 10, 20):
        algs.append(
            {"label": f"Learn all ILR {ilr}",
             "dir": f"{EXP_DIR}/taxi/ours_learn_all/4opts/ilr{ilr}/eval",
             "tag": TAXI_FAMP_TAG,
             "x_values": TAXI_FAMP_XLABELS(EPISODES),
             "seeds": TAXI_FAMP_SEEDS,
             "timesteps": EPISODES // TAXI_FAMP_EPU + 1
             }
        )

    add_alg_data_test(algs, TAXI_TEST_ENVS)
    plot_name = "taxi_ablation_param_groups_hyperparams"
    plot_algs(algs, 1, 1, X_LABEL, TAXI_Y_LABEL, plot_name, average_plots=True, spread_type="std")
    plt.show()

def taxi_per_env_plots_no_collapsed_seed():
    cache_path = "../../plots/cache/taxi_per_env_no_collapsed.pickle"
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as handle:
            algs = pickle.load(handle)
    else:
        algs = get_avg_performance_algs_no_collapsed_seed()
        add_alg_data_test(algs, TAXI_TEST_ENVS)

        with open(cache_path, 'wb') as handle:
            pickle.dump(algs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    nrows = 3
    ncols = 4
    plot_name = "taxi_per_env_performance_no_collapsed"
    plot_algs(algs, nrows, ncols, X_LABEL, TAXI_Y_LABEL, plot_name, average_plots=False, spread_type="std", fontsize=FONTSIZE + 2)

def taxi_per_env_plots():
    cache_path = "../../plots/cache/taxi_avg_performance_algs.pickle"
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as handle:
            algs = pickle.load(handle)
    else:
        algs = get_avg_performance_algs()
        add_alg_data_test(algs, TAXI_TEST_ENVS)
        with open(cache_path, 'wb') as handle:
            pickle.dump(algs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    nrows = 3
    ncols = 4
    plot_name = "taxi_per_env_performance"
    plot_algs(algs, nrows, ncols, X_LABEL, TAXI_Y_LABEL, plot_name, average_plots=False, spread_type="bootstrap", fontsize=FONTSIZE)


def taxi_traj():
    task = 4
    checkpoint_path = f"{EXP_DIR}/taxi/ours/4opts/eval/seed42/env{task}/checkpoints/epoch500.tar"
    checkpoint = torch.load(checkpoint_path)

    # option_probs = torch.softmax(checkpoint["policy_params"]["optionslayer1weight"], dim=-1)
    # termination_probs = torch.sigmoid(checkpoint["policy_params"]["terminationlayer1weight"].squeeze().transpose(0, 1))
    # policy_probs = torch.softmax(checkpoint["policy_params"]["subpolicylayer1weight"].transpose(0, 1), dim=-1)

    checkpoint["opts"].std_value = 1
    checkpoint["opts"].std_type = "diagonal"
    checkpoint["opts"].term_time = 0
    checkpoint["opts"].policy_type = "ltopt"
    policy = create_policy(checkpoint["opts"], 72, 6, "discrete", checkpoint_path=checkpoint_path)
    env = Taxi()
    state_map, state_inverse_map = env._create_coord_mapping()
    env.set_task(task)
    env.reset()
    np.random.seed(42)
    torch.manual_seed(42)

    traj_data = generate_samples(env=env, policy=policy, episodes=1, params=None)
    observation_indices = torch.argmax(torch.cat(traj_data["observations"], dim=0), dim=1)
    observation_indices = [observation_indices[i].item() for i in range(observation_indices.shape[0])]
    options = [traj_data["options"][0][i].long().item() for i in range(traj_data["options"][0].shape[0])]
    terminations = [traj_data["terminations"][0][i].long().item() for i in range(traj_data["terminations"][0].shape[0])]
    actions = [traj_data["actions"][0][i].long().item() for i in range(traj_data["terminations"][0].shape[0])]

    no_blocks = True
    grid = env.map if no_blocks else env.grid
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    cmap = cm.get_cmap('Paired')
    for j in range(2):
        create_grid_plot(ax=axes[j], grid=grid)
    for i in range(len(actions)):
        obs = observation_indices[i]
        opt = options[i]
        a = actions[i]
        y_pos, x_pos = state_inverse_map[obs]
        ax = axes[1-int(obs < 36)]
        if opt == 0:
            c = cmap(5)
        elif opt == 1:
            c = cmap(7)
        elif opt == 2:
            c = cmap(2)
        elif opt == 3:
            c = cmap(9)
        else:
            raise RuntimeError("Option index out of range for coloring")
        if 0 < a < 5:
            x_dir, y_dir = 0, 0
            if a == 3:
                y_dir = 1
            elif a == 4:
                y_dir = -1
            elif a == 2:
                x_dir = 1
            elif a == 1:
                x_dir = -1
            headwidth = 9
            headlength = 20
            ax.quiver(x_pos + 0.5, y_pos + 0.5, x_dir, y_dir, color=c, angles='xy',
                      scale_units='xy', scale=1, pivot='middle', headwidth=headwidth, headaxislength=headlength,
                      headlength=headlength)  # width=0.1)
        else:
            rect = patches.Rectangle((x_pos + 0.25, y_pos + 0.25), 0.5, 0.5, linewidth=2, edgecolor=c, facecolor=c)
            ax.add_patch(rect)
    plt.tight_layout()
    plt.savefig("taxi_traj.pdf", bbox_inches='tight', dpi=150)




def taxi_options(checkpoint_path: str, plot_name: str):
    checkpoint = torch.load(checkpoint_path)
    option_probs = torch.softmax(checkpoint["policy_params"]["optionslayer1weight"], dim=-1)
    termination_probs = torch.sigmoid(checkpoint["policy_params"]["terminationlayer1weight"].squeeze().transpose(0, 1))
    policy_probs = torch.softmax(checkpoint["policy_params"]["subpolicylayer1weight"].transpose(0, 1), dim=-1)

    env = Taxi()
    no_blocks = True
    grid = env.map if no_blocks else env.grid
    options = option_probs.shape[1]
    coords = env.create_coords()
    # TODO Refactor taxi and four envs a bit
    #  Mental note: Plot probabilities is probably good to have in env (since plots are different or call it plot
    #  values and use it for baseline as well)
    #  Add ploting for baseline too
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
    for i in range(options):

        for j in range(2):
            if j == 0:
                # NoPassenger
                selected_policy_probs = policy_probs[:36, i, :]
                selected_termination_probs = termination_probs[:36, i]
                seleceted_option_probs = option_probs[:36, i]
            else:
                # Passenger
                selected_policy_probs = policy_probs[36:, i, :]
                selected_termination_probs = termination_probs[36:, i]
                seleceted_option_probs = option_probs[36:, i]
            max_val, max_index = torch.max(selected_policy_probs, dim=1)
            max_val = max_val.numpy()
            max_index = max_index.numpy()
            create_grid_plot(ax=axes[j * 2][i], grid=grid)
            quiv_p = plot_policy(ax=axes[j * 2][i], arrow_data=env.get_plot_arrow_params(max_val, max_index), grid=grid,
                               values=False, max_index=max_index)
            quiv_t = plot_terminations(ax=axes[j * 2 + 1][i], probs=selected_termination_probs, coords=coords, grid=grid,
                                       values=False)

    plt.tight_layout()
    fig.subplots_adjust(right=0.94)
    cax = fig.add_axes([0.96, 0.07, 0.02, 0.85]) # This is the position for the colorbar
    cax.tick_params(labelsize=FONTSIZE + 3)
    fig.colorbar(quiv_p, cax=cax)

    fig.subplots_adjust(left=0.07)
    cax2 = fig.add_axes([0.02, 0.07, 0.02, 0.85])
    fig.colorbar(quiv_t, cax=cax2)
    cax2.tick_params(labelsize=FONTSIZE + 3)
    plt.savefig(f"{plot_name}.pdf", bbox_inches='tight', dpi=150)

def taxi_2options():
    checkpoint_path = f"{EXP_DIR}/taxi/ours/2opts/train/seed42/checkpoints/epoch2000.tar"
    plot_name = "taxi_2options_test"
    checkpoint = torch.load(checkpoint_path)
    option_probs = torch.softmax(checkpoint["policy_params"]["optionslayer1weight"], dim=-1)
    termination_probs = torch.sigmoid(checkpoint["policy_params"]["terminationlayer1weight"].squeeze().transpose(0, 1))
    policy_probs = torch.softmax(checkpoint["policy_params"]["subpolicylayer1weight"].transpose(0, 1), dim=-1)

    env = Taxi()
    no_blocks = True
    grid = env.map if no_blocks else env.grid
    options = option_probs.shape[1]
    coords = env.create_coords()
    # TODO Refactor taxi and four envs a bit
    #  Mental note: Plot probabilities is probably good to have in env (since plots are different or call it plot
    #  values and use it for baseline as well)
    #  Add ploting for baseline too
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    for i in range(options):

        for j in range(1):
            if j == 0:
                # NoPassenger
                selected_policy_probs = policy_probs[:36, i, :]
                selected_termination_probs = termination_probs[:36, i]
                seleceted_option_probs = option_probs[:36, i]
            else:
                # Passenger
                selected_policy_probs = policy_probs[36:, i, :]
                selected_termination_probs = termination_probs[36:, i]
                seleceted_option_probs = option_probs[36:, i]
            max_val, max_index = torch.max(selected_policy_probs, dim=1)
            max_val = max_val.numpy()
            max_index = max_index.numpy()
            create_grid_plot(ax=axes[i], grid=grid)
            quiv_p = plot_policy(ax=axes[i], arrow_data=env.get_plot_arrow_params(max_val, max_index), grid=grid,
                               values=False, max_index=max_index)
            # quiv_t = plot_terminations(ax=axes[j][i * 2 + 1], probs=selected_termination_probs, coords=coords, grid=grid,
            #                            values=False)

    plt.tight_layout()
    fig.subplots_adjust(right=0.94)
    cax = fig.add_axes([0.96, 0.07, 0.02, 0.85]) # This is the position for the colorbar
    cax.tick_params(labelsize=FONTSIZE + 3)
    fig.colorbar(quiv_p, cax=cax)

    # fig.subplots_adjust(left=0.07)
    # cax2 = fig.add_axes([0.02, 0.07, 0.02, 0.85])
    # fig.colorbar(quiv_t, cax=cax2)
    # cax2.tick_params(labelsize=FONTSIZE + 3)
    plt.savefig(f"{plot_name}.pdf", bbox_inches='tight', dpi=150)

def taxi_options_4opts():
    taxi_options(
        checkpoint_path=f"{EXP_DIR}/taxi/ours/4opts/train/seed42/checkpoints/epoch2000.tar",
        plot_name="taxi_options_test"
    )

def taxi_options_learn_all():
    taxi_options(
        checkpoint_path=f"{EXP_DIR}/taxi/ours_learn_all/4opts/ilr20/train/seed42/checkpoints/epoch2000.tar",
        plot_name="taxi_options_learn_all"
    )

def taxi_metatrain_plots():
    FAOPG_METRIC = "Step00DiscountedReturn/Avg"
    data_type = "Return"
    algs = []
    for opt in (4,):
        pass
        algs.append(
            {"label": f"FAMP (Ours)",
             "dir": f"../../run_data/paper_experiments/taxi/ours/4opts/train",
             "tag": f"Step03{data_type}/Avg",
             "x_values": [i * 64 * 10 * 4 for i in range(2000)],
             "seeds": (42, 142, 242, 342, 442),
             "timesteps": 2000
             }
        )
    algs.extend([
        {"label": "Multi-task",
         "dir": f"../../run_data/paper_experiments/taxi/multi/4opts/train",
         "tag": f"Step00{data_type}/Avg",
         "x_values": [i * 48 * 10 for i in range(1550)],
         "seeds": (42, 142, 242, 342, 442),
         "timesteps": 1550
         },
        {"label": f"Learn all",
            "dir": f"../../run_data/paper_experiments/taxi/learn_all/4opts/train",
            "tag": f"Step03{data_type}/Avg",
            "x_values": [i * 64 * 10 * 4 for i in range(2000)],
            "seeds": (42, 142, 242, 342, 442),
            "timesteps": 2000
        },
        {"label": f"Adapt options",
            "dir": f"../../run_data/paper_experiments/taxi/learn_all/4opts/train",
            "tag": f"Step03{data_type}/Avg",
            "x_values": [i * 64 * 10 * 4 for i in range(2000)],
            "seeds": (42, 142, 242, 342, 442),
            "timesteps": 2000
        },
        {"label": "MLSH",
         "dir": f"../../run_data/paper_experiments/taxi/mlsh/macro4/train",
         "tag": data_type,
         "x_values": [i * 120 * 2 for i in range(2500)],
         "seeds": (42, 43, 44, 45, 46),
         "timesteps": 2500
         },
        # {"label": "MLSH 10 steps",
        #  "dir": f"../../run_data/paper_experiments/taxi/mlsh/macro10/train",
        #  "tag": data_type,
        #  "x_values": [i * 120 * 2 for i in range(2500)],
        #  "seeds": (42, 43, 44, 45, 46),
        #  "timesteps": 2500
        #  }
    ])


    #     else:
    #         data = np.stack([get_csv_data(f"{algs[a]['dir']}/seed{seed}/logs/", timesteps=algs[a]["timesteps"], tag=algs[a]["tag"]) for seed in algs[a]['seeds']])
    #         algs[a]["data"] = data
    cache_path = "../../plots/cache/taxi_training.pickle"
    fig, ax = plt.subplots(ncols=1, nrows=1, dpi=1000)
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as handle:
            algs_old = pickle.load(handle)
    else:
        algs_old = []
    if not ([a["label"] for a in algs] == [a["label"] for a in algs_old]):
        add_alg_data_train(algs)
        with open(cache_path, 'wb') as handle:
            pickle.dump(algs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        algs = algs_old
    for a in range(len(algs)):
        if algs[a]["label"] == "MLSH":
            data = np.stack([get_tf_data(f"{algs[a]['dir']}/TaxiAgent4_{seed}/", tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = data
        elif algs[a]["label"] == "MLSH 10 steps":
            data = np.stack([get_tf_data(f"{algs[a]['dir']}/TaxiAgent10_{seed}/", tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = data
    data_func = lambda x: x
    plot_all_alg_values(algs=algs, data_func=data_func, ax=ax, spread_type="std")

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
    ax.set_ylabel("Return", fontsize=FONTSIZE)
    ax.set_xlabel("Episodes", fontsize=FONTSIZE)
    ax.set_xlim(0, 4e5)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA - 0.05)
    plt.savefig(f"../../plots/taxi_training.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')

    # ax.legend(loc="lower right", prop={'size': FONTSIZE}, ncol=2)
    # ax.set_ylabel("Return", fontsize=FONTSIZE)
    # ax.set_xlabel("Episodes", fontsize=FONTSIZE)
    # # ax.set_xscale('log')
    # ax.set_xlim(0, 4e5)
    # ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    # ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA-0.05)
    # plt.savefig(f"../../plots/taxi_training.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')


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
    taxi_avg()
    # taxi_avg_zoom()
    # taxi_hyperparams()
    # taxi_hyperparams_no_collapsed_seed()
    # taxi_ablation()
    # taxi_ablation_param_groups_hyperparams()
    # taxi_traj()
    # taxi_options_4opts()
    # taxi_options_learn_all()
    # taxi_2options()
    # taxi_plots()
    # taxi_metatrain_plots()
    # taxi_ft_time()
    # taxi_per_env_plots()
    # taxi_per_env_plots_no_collapsed_seed()
