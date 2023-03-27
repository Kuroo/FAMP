import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pickle
import numpy as np
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import ListedColormap
import scipy.stats
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LINESTYLES = ('-', '--', '-.', ':')
COMMON_NAMES = {
    "mlsh": "MLSH",
    "ours": "FAMP (Ours)",
    "single": "Single-task",
    "multi": "Multi-task",
    "term": "Ours + FT",
    "ppo": "PPO",
    "rl2": "RL2",
    "maml": "MAML",
    "learnall": "Learn All",
    "adaptopts": "Adapt Options",
    "nohierarchy": "No Hierarchy",
    "learninit": "Learn High-level"
}
FONTSIZE = 12
ALPHA = 0.15
EXP_DIR = "../../run_data/paper_experiments"
X_LABEL = "Episodes"

# ANT
ANT_TRAIN_ENVS = tuple(list(range(9)))
ANT_FAMP_SEEDS = (42, 142, 242)


def get_tf_paths(dir="../../run_data/test", subdir_regex=None):
    import os
    import re
    if subdir_regex is not None:
        pattern = re.compile(subdir_regex)
    tf_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if "tfevent" in file:
                full_path = os.path.join(root, file)
                if subdir_regex is None or pattern.search(full_path):
                    tf_files.append(full_path)
    return tf_files


def get_tf_data(dir: str, tag: str, timesteps: int, subdir_regex=None):
    paths = get_tf_paths(dir=dir, subdir_regex=subdir_regex)
    ret_val = []

    for path in paths:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        w_times, step_nums, vals = zip(*event_acc.Scalars(tag))
        if len(paths) == 1:
            ret_val = vals[:timesteps]
        else:
            ret_val.append(vals[:timesteps])
    return np.array(ret_val)


def get_garage_maml_data(dir: str, timesteps: int):
    paths = get_tf_paths(dir=dir, subdir_regex=None)
    ret_val = []

    for path in paths:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        vals = []
        for step in range(timesteps):
            w_times, step_nums, val = zip(*event_acc.Scalars(f"Step{step}/AvgRet"))
            vals.append(val[0])
        if len(paths) == 1:
            ret_val = vals[:timesteps]
        else:
            ret_val.append(vals[:timesteps])
    return np.array(ret_val)


def get_maml_data(dir: str, timesteps: int):
    paths = get_tf_paths(dir=dir, subdir_regex=None)
    ret_val = []

    for path in paths:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        vals = []
        for step in range(timesteps):
            w_times, step_nums, val = zip(*event_acc.Scalars(f"Step{step}/Ret/Avg"))
            vals.append(val[0])
        if len(paths) == 1:
            ret_val = vals[:timesteps]
        else:
            ret_val.append(vals[:timesteps])
    return np.array(ret_val)


def get_csv_data(dir: str, timesteps: int, tag: str):
    import csv
    lines = []
    with open(f'{dir}/progress.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lines.append(row)
    tag_index = -1
    for i in range(len(lines[0])):
        if lines[0][i] == tag:
            tag_index = i
            break
    average_ep_ret = [float(lines[i+1][tag_index]) for i in range(timesteps)]
    return np.array(average_ep_ret)


def get_spinning_up_data(dir: str, timesteps: int):
    import csv
    lines = []
    with open(f'{dir}/progress.txt', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            lines.append(row)
    average_ep_ret = [float(lines[i+1][1]) for i in range(timesteps)]
    return np.array(average_ep_ret)


def get_rl2_data(dir: str, timesteps: int, tag: str):
    with open(f'{dir}/run_data.pkl', 'rb') as handle:
        data = pickle.load(handle)
    return np.array(data[tag])[:timesteps]


def slidind_window_avg(vals, window_size):
    res = np.empty_like(vals)
    s = 0
    for i in range(len(vals)):
        if i >= window_size:
            s -= vals[i - window_size] / window_size
            s += vals[i] / window_size
        else:
            s = (s * i + vals[i]) / (i + 1)
        res[i] = s
    return res


def add_alg_data_test(algs, envs):
    for a in range(len(algs)):
        if "MLSH" in algs[a]["label"]:
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_tf_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                      tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]),
                                     window_size=1) for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        elif algs[a]["label"] == "PPO":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(
                    vals=get_spinning_up_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                              timesteps=algs[a]["timesteps"]),
                    window_size=1) for j in range(len(envs))] # 5
                 for i in range(len(algs[a]["seeds"]))])
        elif algs[a]["label"] == "RL2":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_rl2_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                       timesteps=algs[a]["timesteps"], tag=algs[a]["tag"]),
                                     window_size=1) for j in range(len(envs))] # 20
                 for i in range(len(algs[a]["seeds"]))])
        elif "MAML" in algs[a]["label"]:
            algs[a]["data"] = np.array(
                [[get_maml_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}", timesteps=algs[a]["timesteps"])
                  for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        else:
            algs[a]["data"] = np.array(
                [[get_tf_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}", tag=algs[a]["tag"],
                              timesteps=algs[a]["timesteps"])
                  for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])


def add_alg_data_train(algs):
    for a in range(len(algs)):
        if algs[a]["label"] == "MLSH":
            first_part = np.stack([get_tf_data(f"{algs[a]['dir']}/AntAgentNoResetInit{seed}/",
                                                      tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']])
            second_part = np.stack([get_tf_data(f"{algs[a]['dir']}/AntAgentNoResetContinue{seed}/",
                                                      tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = np.concatenate([first_part, second_part], axis=1)
        elif algs[a]["label"] == "RL2":
            data = np.stack([get_tf_data(f"{algs[a]['dir']}/antobstacles_{seed}/",
                                                       timesteps=algs[a]["timesteps"], tag=algs[a]["tag"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = data
        else:
            data = [get_tf_data(f"{algs[a]['dir']}/seed{seed}/", tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']]
            data = np.stack(data)
            algs[a]["data"] = data
    # for a in range(len(algs)):
    #     if "MLSH" in algs[a]["label"]:
    #         algs[a]["data"] = np.array(
    #             [[slidind_window_avg(vals=get_tf_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
    #                                                   tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]),
    #                                  window_size=1) for j in range(len(envs))] # 10
    #              for i in range(len(algs[a]["seeds"]))])
    #     elif algs[a]["label"] == "PPO":
    #         algs[a]["data"] = np.array(
    #             [[slidind_window_avg(
    #                 vals=get_spinning_up_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
    #                                           timesteps=algs[a]["timesteps"]),
    #                 window_size=1) for j in range(len(envs))] # 5
    #              for i in range(len(algs[a]["seeds"]))])
    #     elif algs[a]["label"] == "RL2":
    #         algs[a]["data"] = np.array(
    #             [[slidind_window_avg(vals=get_rl2_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
    #                                                    timesteps=algs[a]["timesteps"], tag=algs[a]["tag"]),
    #                                  window_size=1) for j in range(len(envs))] # 20
    #              for i in range(len(algs[a]["seeds"]))])
    #     elif "MAML" in algs[a]["label"]:
    #         algs[a]["data"] = np.array(
    #             [[get_maml_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}", timesteps=algs[a]["timesteps"])
    #               for j in range(len(envs))]
    #              for i in range(len(algs[a]["seeds"]))])
    #     else:
    #         algs[a]["data"] = np.array(
    #             [[get_tf_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}", tag=algs[a]["tag"],
    #                           timesteps=algs[a]["timesteps"])
    #               for j in range(len(envs))]
    #              for i in range(len(algs[a]["seeds"]))])




def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), scipy.stats.sem(a, axis=0)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def plot_all_alg_values(algs, data_func, ax, spread_type, reduce_alpha=0):
    for a in range(len(algs)):
        ls = LINESTYLES[a % len(LINESTYLES)]
        data = data_func(algs[a]["data"])
        if "linestyle" in algs[a]:
            ls = algs[a]["linestyle"]
        if spread_type == "median":
            mid = np.median(data, axis=0)
            top, bot = np.percentile(data, [75, 25], axis=0)
        elif spread_type == "std":
            mean, std = np.mean(data, axis=0), np.std(data, axis=0)
            mid, top, bot = mean, mean + std, mean - std
        elif spread_type == "confidence_t":
            mean, std = np.mean(data, axis=0), np.std(data, ddof=1, axis=0)
            bot, top = scipy.stats.t.interval(0.95, len(data)-1, loc=np.mean(data, axis=0), scale=scipy.stats.sem(data, axis=0))
            mid = mean
        elif spread_type == "confidence_normal":
            mid = np.mean(data, axis=0)
            sem = scipy.stats.sem(data, axis=0)
            bot, top = mid - 1.96 * sem, mid + 1.96 * sem
        elif spread_type == "bootstrap":
            mid = np.mean(data, axis=0)
            res = scipy.stats.bootstrap(data=(data, ), statistic=np.mean, confidence_level=0.95, method="percentile")
            top = res.confidence_interval.high
            bot = res.confidence_interval.low
        else:
            raise NotImplementedError
        x_values = algs[a]["x_values"] if "x_values" in algs[a] else None
        plot_values(ax=ax, x_values=x_values, mid=mid, top=top, bot=bot, label=algs[a]["label"], ls=ls,
            reduce_alpha=reduce_alpha)


def plot_values(ax, x_values, mid, top, bot, label, ls, reduce_alpha=0):
    if x_values is None:
        ax.plot(mid, label=label, linestyle=ls)
        ax.fill_between(range(mid.shape[0]), top, bot, alpha=ALPHA - reduce_alpha)
    else:
        ax.plot(x_values, mid, label=label, linestyle=ls)
        ax.fill_between(x_values, top, bot, alpha=ALPHA - reduce_alpha)


def plot_algs(algs, nrows, ncols, x_label, y_label, plot_name, average_plots, spread_type, fontsize=FONTSIZE, ylim=None,
    reduce_alpha=0, legend=True):
    if average_plots:
        fig, ax = plt.subplots(ncols=1, nrows=1, dpi=100)
        data_func = lambda x: np.mean(x, axis=1)
        plot_all_alg_values(algs=algs, data_func=data_func, ax=ax, spread_type=spread_type, reduce_alpha=reduce_alpha)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA - 0.05)
        if legend:
            ax.legend(loc="lower right", prop={'size': fontsize}, ncol=2)
        ax.set_ylabel(y_label, fontsize=fontsize)
        ax.set_xlabel(x_label, fontsize=fontsize)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
    else:
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, dpi=100, figsize=(20, 12))
        for i in range(algs[0]["data"].shape[1]):
            ax = axes[i // ncols][i % ncols]
            data_func = lambda x: x[:, i, :]
            plot_all_alg_values(algs=algs, data_func=data_func, ax=ax, spread_type=spread_type, reduce_alpha=reduce_alpha)
            if i == 0 and legend:
                ax.legend(loc="lower right", prop={'size': fontsize - 5}, ncol=2)
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA - 0.05)
            if ylim is not None:
                ax.set_ylim(ylim[0], ylim[1])
        fig.text(0.5, 0.04, x_label, ha='center', fontsize=fontsize + 3)
        fig.text(0.04, 0.5, y_label, va='center', rotation='vertical', fontsize=fontsize + 3)

    plt.savefig(f"../../plots/{plot_name}.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')
    plt.show()


def ant_maze_plots(env_type):
    nrows = 2
    ncols = 5
    envs = list(range(9))
    assert env_type in ("original", "noreset"), "Unknown env type"
    addition = "" if env_type == "original" else "NoReset"
    # Ant maze
    algs = [
        {"label": "FAMP (Ours)",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/ours/eval",
         "tag": "Step00Return/Avg",
         "x_values": [i * 20 for i in range(11)],
         "seeds": (42, 142, 242),
         "timesteps": 11
         },
        {"label": "MLSH",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/mlsh/eval",
         "tag": "Return",
         "x_values": [i * 2 for i in range(101)],
         "seeds": (42, 43, 45),
         "timesteps": 101
         },
        {"label": "PPO",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/ppo/eval",
         "tag": "AverageEpRet",
         "x_values": [i * 4 for i in range(51)],
         "seeds": (42, 142, 242),
         "timesteps": 51
         },
        {"label": "RL2",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/rl2/eval",
         "tag": "returns",
         "x_values": [i for i in range(201)],
         "seeds": (42, 142, 242),
         "timesteps": 201
         }
    ]
    for a in range(len(algs)):
        if algs[a]["label"] == "MLSH":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_tf_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                      tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]),
                                     window_size=1) for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        elif algs[a]["label"] == "PPO":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_spinning_up_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                               timesteps=algs[a]["timesteps"]),
                                     window_size=1) for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        elif algs[a]["label"] == "RL2":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_rl2_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                       timesteps=algs[a]["timesteps"], tag=algs[a]["tag"]),
                                     window_size=1) for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        else:
            algs[a]["data"] = np.array(
                [[get_tf_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}", tag=algs[a]["tag"], timesteps=algs[a]["timesteps"])
                  for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])

    for i in range(len(envs)+1):
        fig, ax = plt.subplots(ncols=1, nrows=1, dpi=100)
        for a in range(len(algs)):
            if i == len(envs):
                # ax.set_title("Average performance over all envs", fontsize=FONTSIZE+5)
                data = np.mean(algs[a]["data"], axis=1)
            else:
                # ax.set_title(f"Average performance env{envs[i]}", fontsize=FONTSIZE+5)
                data = algs[a]["data"][:, i, :]
            mean, std = np.mean(data, axis=0), np.std(data, axis=0)
            if "x_values" not in algs[a]:
                ax.plot(mean, label=algs[a]["label"])
                ax.fill_between(range(mean.shape[0]), mean+std, mean-std, alpha=ALPHA)
            else:
                ax.plot(algs[a]["x_values"], mean, label=algs[a]["label"])
                ax.fill_between(algs[a]["x_values"], mean + std, mean - std, alpha=ALPHA)
        ax.legend(loc="lower right", prop={'size': FONTSIZE}, ncol=2)
        ax.set_ylabel("Return", fontsize=FONTSIZE)
        ax.set_xlabel("Episodes", fontsize=FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA-0.05)
        if i == len(envs):
            plt.savefig(f"ant_maze_{env_type}_performance_avg.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')
        else:
            plt.savefig(f"ant_maze_{env_type}_env{i}.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', prop={'size': FONTSIZE})
    # ax.plot(np.ones(500) * (0.95 ** 5), alpha=0.3, label="Max return")
    # ax.plot(np.ones(200) * 0.59049, alpha=0.7, label="Max return")
    plt.savefig(f"ant_maze_{env_type}_performance.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')
    plt.show()


def ant_maze_plots_test(env_type):
    envs = list(range(9,13))
    assert env_type in ("original", "noreset"), "Unknown env type"
    addition = "" if env_type == "original" else "NoReset"
    # Ant maze
    algs = [
        {"label": "FAMP (Ours)",
         "dir": f"../../run_data/paper_experiments/ant_maze/noreset_test/ours/eval",
         "tag": "Step00Return/Avg",
         "x_values": [i * 20 for i in range(11)],
         "seeds": (42, 142, 242),
         "timesteps": 11
         },
        # {"label": "MLSH",
        #  "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/mlsh/eval",
        #  "tag": "Return",
        #  "x_values": [i * 2 for i in range(101)],
        #  "seeds": (42, 43, 45),
        #  "timesteps": 101
        #  },
        # {"label": "PPO",
        #  "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/ppo/eval",
        #  "tag": "AverageEpRet",
        #  "x_values": [i * 4 for i in range(51)],
        #  "seeds": (42, 142, 242),
        #  "timesteps": 51
        #  },
        # {"label": "RL2",
        #  "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/rl2/eval",
        #  "tag": "returns",
        #  "x_values": [i for i in range(201)],
        #  "seeds": (42, 142, 242),
        #  "timesteps": 201
        #  }
    ]
    for a in range(len(algs)):
        if algs[a]["label"] == "MLSH":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_tf_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                      tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]),
                                     window_size=1) for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        elif algs[a]["label"] == "PPO":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_spinning_up_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                               timesteps=algs[a]["timesteps"]),
                                     window_size=1) for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        elif algs[a]["label"] == "RL2":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_rl2_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                       timesteps=algs[a]["timesteps"], tag=algs[a]["tag"]),
                                     window_size=1) for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        else:
            algs[a]["data"] = np.array(
                [[get_tf_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}", tag=algs[a]["tag"], timesteps=algs[a]["timesteps"])
                  for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])

    for i in range(len(envs)+1):
        fig, ax = plt.subplots(ncols=1, nrows=1, dpi=100)
        for a in range(len(algs)):
            if i == len(envs):
                # ax.set_title("Average performance over all envs", fontsize=FONTSIZE+5)
                data = np.mean(algs[a]["data"], axis=1)
            else:
                # ax.set_title(f"Average performance env{envs[i]}", fontsize=FONTSIZE+5)
                data = algs[a]["data"][:, i, :]
            mean, std = np.mean(data, axis=0), np.std(data, axis=0)
            if "x_values" not in algs[a]:
                ax.plot(mean, label=algs[a]["label"])
                ax.fill_between(range(mean.shape[0]), mean+std, mean-std, alpha=ALPHA)
            else:
                ax.plot(algs[a]["x_values"], mean, label=algs[a]["label"])
                ax.fill_between(algs[a]["x_values"], mean + std, mean - std, alpha=ALPHA)
        ax.legend(loc="lower right", prop={'size': FONTSIZE}, ncol=2)
        ax.set_ylabel("Return", fontsize=FONTSIZE)
        ax.set_xlabel("Episodes", fontsize=FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA-0.05)
        if i == len(envs):
            plt.savefig(f"ant_maze_{env_type}_performance_avg.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')
        else:
            plt.savefig(f"ant_maze_{env_type}_env{i}.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', prop={'size': FONTSIZE})
    # ax.plot(np.ones(500) * (0.95 ** 5), alpha=0.3, label="Max return")
    # ax.plot(np.ones(200) * 0.59049, alpha=0.7, label="Max return")
    plt.savefig(f"ant_maze_{env_type}_performance.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')
    plt.show()


def ant_maze_metatrain_plots(ax, env_type):
    assert env_type in ("original", "noreset"), "Unknown env type"
    # Ant maze
    algs = [
        {"label": "FAMP (Ours)",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/ours/pretrain",
         "tag": "Step02Return/Avg",
         "x_values": [i * 48 * 3 * 20 for i in range(2620)],
         "seeds": (42, 142, 242),
         "timesteps": 2540
         },
        {"label": "MLSH",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/mlsh/pretrain",
         "tag": "Return",
         "x_values": [i * 120 * 2 for i in range(1800 * 2)],
         "seeds": (42, 43, 45),
         "timesteps": 1800
         },
        {"label": "RL2",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/rl2/pretrain",
         "tag": "Average/AverageReturn",
         "x_values": [i * 9 * 4 for i in range(2000)],
         "seeds": (42, 142, 242),
         "timesteps": 2000
         }
    ]
    for a in range(len(algs)):
        if algs[a]["label"] == "MLSH":
            first_part = np.stack([get_tf_data(f"{algs[a]['dir']}/AntAgentNoResetInit{seed}/",
                                                      tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']])
            second_part = np.stack([get_tf_data(f"{algs[a]['dir']}/AntAgentNoResetContinue{seed}/",
                                                      tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = np.concatenate([first_part, second_part], axis=1)
        elif algs[a]["label"] == "RL2":
            data = np.stack([get_tf_data(f"{algs[a]['dir']}/antobstacles_{seed}/",
                                                       timesteps=algs[a]["timesteps"], tag=algs[a]["tag"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = data
        else:
            data = np.stack([get_tf_data(f"{algs[a]['dir']}/seed{seed}/", tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = data

    for a in range(len(algs)):
        data = algs[a]["data"]
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        if "x_values" not in algs[a]:
            ax.plot(mean, label=algs[a]["label"])
            ax.fill_between(range(mean.shape[0]), mean+std, mean-std, alpha=ALPHA)
        else:
            ax.plot(algs[a]["x_values"], mean, label=algs[a]["label"])
            ax.fill_between(algs[a]["x_values"], mean + std, mean - std, alpha=ALPHA)
    ax.legend(loc="lower right", prop={'size': FONTSIZE}, ncol=2)
    ax.set_ylabel("Return", fontsize=FONTSIZE)
    ax.set_xlabel("Episodes", fontsize=FONTSIZE)
    ax.set_xscale('log')
    ax.set_xlim(1e3, 1e7)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA-0.05)
    # plt.savefig(f"ant_maze_{env_type}_meta_train_performance_avg.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', prop={'size': FONTSIZE})
    # ax.plot(np.ones(500) * (0.95 ** 5), alpha=0.3, label="Max return")
    # ax.plot(np.ones(200) * 0.59049, alpha=0.7, label="Max return")
    # plt.savefig(f"ant_maze_{env_type}_performance.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')
    # plt.show()


def ant_maze_metatrain_plots(ax, env_type):
    assert env_type in ("original", "noreset"), "Unknown env type"
    # Ant maze
    algs = [
        {"label": "FAMP (Ours)",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/ours/pretrain",
         "tag": "Step02Return/Avg",
         "x_values": [i * 48 * 3 * 20 for i in range(2620)],
         "seeds": (42, 142, 242),
         "timesteps": 2620
         },
        {"label": "MLSH",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/mlsh/pretrain",
         "tag": "Return",
         "x_values": [i * 120 * 2 for i in range(1800 * 2)],
         "seeds": (42, 43, 45),
         "timesteps": 1800
         },
        {"label": "RL2",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/rl2/pretrain",
         "tag": "Average/AverageReturn",
         "x_values": [i * 9 * 4 for i in range(2000)],
         "seeds": (42, 142, 242),
         "timesteps": 2000
         }
    ]
    for a in range(len(algs)):
        if algs[a]["label"] == "MLSH":
            first_part = np.stack([get_tf_data(f"{algs[a]['dir']}/AntAgentNoResetInit{seed}/",
                                                      tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']])
            second_part = np.stack([get_tf_data(f"{algs[a]['dir']}/AntAgentNoResetContinue{seed}/",
                                                      tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = np.concatenate([first_part, second_part], axis=1)
        elif algs[a]["label"] == "RL2":
            data = np.stack([get_tf_data(f"{algs[a]['dir']}/antobstacles_{seed}/",
                                                       timesteps=algs[a]["timesteps"], tag=algs[a]["tag"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = data
        else:
            data = np.stack([get_tf_data(f"{algs[a]['dir']}/seed{seed}/", tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = data

    for a in range(len(algs)):
        data = algs[a]["data"]
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        if "x_values" not in algs[a]:
            ax.plot(mean, label=algs[a]["label"])
            ax.fill_between(range(mean.shape[0]), mean+std, mean-std, alpha=ALPHA)
        else:
            ax.plot(algs[a]["x_values"], mean, label=algs[a]["label"])
            ax.fill_between(algs[a]["x_values"], mean + std, mean - std, alpha=ALPHA)
    ax.legend(loc="lower right", prop={'size': FONTSIZE}, ncol=2)
    ax.set_ylabel("Return", fontsize=FONTSIZE)
    ax.set_xlabel("Episodes", fontsize=FONTSIZE)
    ax.set_xscale('log')
    ax.set_xlim(1e3, 1e7)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA-0.05)
    # plt.savefig(f"ant_maze_{env_type}_meta_train_performance_avg.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', prop={'size': FONTSIZE})
    # ax.plot(np.ones(500) * (0.95 ** 5), alpha=0.3, label="Max return")
    # ax.plot(np.ones(200) * 0.59049, alpha=0.7, label="Max return")
    # plt.savefig(f"ant_maze_{env_type}_performance.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')
    # plt.show()


def taxi_metatrain_plots(ax=None):
    # FAOPG_METRIC = "Step00Return/Avg"
    FAOPG_METRIC = "Step00DiscountedReturn/Avg"
    # Ant maze
    algs = []
    for opt in (4,):
        pass
        algs.append(
            {"label": f"FAMP (Ours)",
             "dir": f"../../run_data/paper_experiments/taxi_1500/ours/4opts/pretrain",
             "tag": "Step03Return/Avg",
             "x_values": [i * 64 * 10 * 4 for i in range(2000)],
             "seeds": (42, 142, 242, 342, 442),
             "timesteps": 2000
             }
        )
    algs.extend([
        {"label": "Multi-task",
         "dir": f"../../run_data/paper_experiments/taxi_1500/multi/4opts/pretrain",
         "tag": "Step00Return/Avg",
         "x_values": [i * 48 * 10 for i in range(1550)],
         "seeds": (42, 142, 242, 342, 442),
         "timesteps": 1550
         },
        {"label": "MLSH 4 steps",
         "dir": f"../../run_data/paper_experiments/taxi_1500/mlsh/macro4/pretrain",
         "tag": "Return",
         "x_values": [i * 120 * 2 for i in range(2500)],
         "seeds": (42, 43, 44, 45, 46),
         "timesteps": 2500
         },
        {"label": "MLSH 10 steps",
         "dir": f"../../run_data/paper_experiments/taxi_1500/mlsh/macro10/pretrain",
         "tag": "Return",
         "x_values": [i * 120 * 2 for i in range(2500)],
         "seeds": (42, 43, 44, 45, 46),
         "timesteps": 2500
         }
    ])

    for a in range(len(algs)):
        if algs[a]["label"] == "MLSH 4 steps":
            data = np.stack([get_tf_data(f"{algs[a]['dir']}/TaxiAgent4_{seed}/", tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = data
        elif algs[a]["label"] == "MLSH 10 steps":
            data = np.stack([get_tf_data(f"{algs[a]['dir']}/TaxiAgent10_{seed}/", tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = data
        else:
            data = np.stack([get_csv_data(f"{algs[a]['dir']}/seed{seed}/logs/", timesteps=algs[a]["timesteps"], tag=algs[a]["tag"]) for seed in algs[a]['seeds']])
            algs[a]["data"] = data


    for a in range(len(algs)):
        data = algs[a]["data"]
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        if "x_values" not in algs[a]:
            ax.plot(mean, label=algs[a]["label"])
            ax.fill_between(range(mean.shape[0]), mean+std, mean-std, alpha=ALPHA)
        else:
            ax.plot(algs[a]["x_values"], mean, label=algs[a]["label"])
            ax.fill_between(algs[a]["x_values"], mean + std, mean - std, alpha=ALPHA)
    ax.legend(loc="lower right", prop={'size': FONTSIZE}, ncol=2)
    ax.set_ylabel("Return", fontsize=FONTSIZE)
    ax.set_xlabel("Episodes", fontsize=FONTSIZE)
    ax.set_xscale('log')
    # ax.set_xlim(0, 1.1e4)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA-0.05)
    plt.savefig(f"../../plots/ant_training.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')


    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', prop={'size': FONTSIZE})
    # ax.plot(np.ones(500) * (0.95 ** 5), alpha=0.3, label="Max return")
    # ax.plot(np.ones(200) * 0.59049, alpha=0.7, label="Max return")
    # plt.show()

def ant_maze_per_env_plots(env_type):
    nrows = 3
    ncols = 3
    envs = list(range(9))
    assert env_type in ("original", "noreset"), "Unknown env type"
    addition = "" if env_type == "original" else "NoReset"
    # Ant maze
    algs = [
        {"label": "FAMP (Ours)",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/ours/eval",
         "tag": "Step00Return/Avg",
         "x_values": [i * 20 for i in range(11)],
         "seeds": (42, 142, 242),
         "timesteps": 11
         },
        {"label": "MLSH",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/mlsh/eval",
         "tag": "Return",
         "x_values": [i * 2 for i in range(101)],
         "seeds": (42, 43, 45),
         "timesteps": 101
         },
        {"label": "PPO",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/ppo/eval",
         "tag": "AverageEpRet",
         "x_values": [i * 4 for i in range(51)],
         "seeds": (42, 142, 242),
         "timesteps": 51
         },
        {"label": "RL2",
         "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/rl2/eval",
         "tag": "returns",
         "x_values": [i for i in range(201)],
         "seeds": (42, 142, 242),
         "timesteps": 201
         }
    ]
    for a in range(len(algs)):
        if algs[a]["label"] == "MLSH":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_tf_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                      tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]),
                                     window_size=10) for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        elif algs[a]["label"] == "PPO":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_spinning_up_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                               timesteps=algs[a]["timesteps"]),
                                     window_size=5) for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        elif algs[a]["label"] == "RL2":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_rl2_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                       timesteps=algs[a]["timesteps"], tag=algs[a]["tag"]),
                                     window_size=20) for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        else:
            algs[a]["data"] = np.array(
                [[get_tf_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}", tag=algs[a]["tag"], timesteps=algs[a]["timesteps"])
                  for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, dpi=100, figsize=(12, 9))
    for i in range(len(envs)):
        ax = axes[i // ncols][i % nrows]
        for a in range(len(algs)):
            if i == len(envs):
                # ax.set_title("Average performance over all envs", fontsize=FONTSIZE+5)
                data = np.mean(algs[a]["data"], axis=1)
            else:
                # ax.set_title(f"Average performance env{envs[i]}", fontsize=FONTSIZE+5)
                data = algs[a]["data"][:, i, :]
            mean, std = np.mean(data, axis=0), np.std(data, axis=0)
            if "x_values" not in algs[a]:
                ax.plot(mean, label=algs[a]["label"])
                ax.fill_between(range(mean.shape[0]), mean+std, mean-std, alpha=ALPHA)
            else:
                ax.plot(algs[a]["x_values"], mean, label=algs[a]["label"])
                ax.fill_between(algs[a]["x_values"], mean + std, mean - std, alpha=ALPHA)
        if i == 0: #len(envs)-1:
            ax.legend(loc="lower right", prop={'size': FONTSIZE-2}, ncol=2)

        ax.set_yticks(np.arange(ax.get_ylim()[0] // 250 * 250, ax.get_ylim()[1] + 50, 250.0))
        # ax.set_ylabel("Return", fontsize=FONTSIZE)
        # ax.set_xlabel("Episodes", fontsize=FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA-0.05)
    fig.text(0.5, 0.04, 'Episodes', ha='center', fontsize=FONTSIZE + 3)
    fig.text(0.04, 0.5, 'Return', va='center', rotation='vertical', fontsize=FONTSIZE + 3)
    # fig.tight_layout()
    plt.savefig(f"ant_maze_per_env_performance_{env_type}.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')
    plt.show()


def ant_maze_per_env_plots_test(env_type):
    nrows = 2
    ncols = 2
    envs = list(range(9, 13))
    assert env_type in ("original", "noreset"), "Unknown env type"
    addition = "" if env_type == "original" else "NoReset"
    # Ant maze
    algs = [
        {"label": "FAMP (Ours)",
         "dir": f"../../run_data/paper_experiments/ant_maze/noreset_test/ours/eval",
         "tag": "Step00Return/Avg",
         "x_values": [i * 20 for i in range(11)],
         "seeds": (42, 142, 242),
         "timesteps": 11
         },
        # {"label": "MLSH",
        #  "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/mlsh/eval",
        #  "tag": "Return",
        #  "x_values": [i * 2 for i in range(101)],
        #  "seeds": (42, 43, 45),
        #  "timesteps": 101
        #  },
        # {"label": "PPO",
        #  "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/ppo/eval",
        #  "tag": "AverageEpRet",
        #  "x_values": [i * 4 for i in range(51)],
        #  "seeds": (42, 142, 242),
        #  "timesteps": 51
        #  },
        # {"label": "RL2",
        #  "dir": f"../../run_data/paper_experiments/ant_maze/{env_type}/rl2/eval",
        #  "tag": "returns",
        #  "x_values": [i for i in range(201)],
        #  "seeds": (42, 142, 242),
        #  "timesteps": 201
        #  }
    ]
    for a in range(len(algs)):
        if algs[a]["label"] == "MLSH":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_tf_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                      tag=algs[a]["tag"], timesteps=algs[a]["timesteps"]),
                                     window_size=10) for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        elif algs[a]["label"] == "PPO":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_spinning_up_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                               timesteps=algs[a]["timesteps"]),
                                     window_size=5) for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        elif algs[a]["label"] == "RL2":
            algs[a]["data"] = np.array(
                [[slidind_window_avg(vals=get_rl2_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}",
                                                       timesteps=algs[a]["timesteps"], tag=algs[a]["tag"]),
                                     window_size=20) for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])
        else:
            algs[a]["data"] = np.array(
                [[get_tf_data(f"{algs[a]['dir']}/seed{algs[a]['seeds'][i]}/env{envs[j]}", tag=algs[a]["tag"], timesteps=algs[a]["timesteps"])
                  for j in range(len(envs))]
                 for i in range(len(algs[a]["seeds"]))])

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, dpi=100, figsize=(12, 9))
    for i in range(len(envs)):
        ax = axes[i // ncols][i % nrows]
        for a in range(len(algs)):
            if i == len(envs):
                # ax.set_title("Average performance over all envs", fontsize=FONTSIZE+5)
                data = np.mean(algs[a]["data"], axis=1)
            else:
                # ax.set_title(f"Average performance env{envs[i]}", fontsize=FONTSIZE+5)
                data = algs[a]["data"][:, i, :]
            mean, std = np.mean(data, axis=0), np.std(data, axis=0)
            if "x_values" not in algs[a]:
                ax.plot(mean, label=algs[a]["label"])
                ax.fill_between(range(mean.shape[0]), mean+std, mean-std, alpha=ALPHA)
            else:
                ax.plot(algs[a]["x_values"], mean, label=algs[a]["label"])
                ax.fill_between(algs[a]["x_values"], mean + std, mean - std, alpha=ALPHA)
        if i == 0: #len(envs)-1:
            ax.legend(loc="lower right", prop={'size': FONTSIZE-2}, ncol=2)

        ax.set_yticks(np.arange(ax.get_ylim()[0] // 250 * 250, ax.get_ylim()[1] + 50, 250.0))
        # ax.set_ylabel("Return", fontsize=FONTSIZE)
        # ax.set_xlabel("Episodes", fontsize=FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=ALPHA-0.05)
    fig.text(0.5, 0.04, 'Episodes', ha='center', fontsize=FONTSIZE + 3)
    fig.text(0.04, 0.5, 'Return', va='center', rotation='vertical', fontsize=FONTSIZE + 3)
    # fig.tight_layout()
    plt.savefig(f"ant_maze_per_env_performance_noreset_test.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')
    plt.show()


def ant_maze_options():
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

    # checkpoint = torch.load("/home/david/Desktop/Papers/test/epoch0700.tar")
    # checkpoint['pytorch_rng_states'] = [checkpoint['pytorch_rng_states'][7]]
    # checkpoint['numpy_rng_states'] = [checkpoint['numpy_rng_states'][7]]
    # torch.save(checkpoint, "process7.tar")
    with open(
            '/home/david/Projects/master_thesis/run_data/paper_experiments/ant_maze/noreset_test/ours/eval/seed242/env12/logs/trajs.pickle',
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
    fig, ax = plt.subplots()

    squares = [(8, 16), (24, 16), (40, 16), (8, 0), (24, 0), (40, 0), (8, -16), (24, -16), (40, -16)]
    for i in range(len(squares)):
        if i in (3, 4, 5, 8): continue
        square = squares[i]
        rect = patches.Rectangle((square[0] - 7.6, square[1] - 7.6), 7.6 * 2, 7.6 * 2, linewidth=1,
                                 edgecolor="lightgray", facecolor="lightgray")
        ax.add_patch(rect)

    rect = patches.Rectangle((40 - 0.75, -24 - 0.75), 1.5, 1.5, linewidth=1, edgecolor="lime", facecolor="lime", )
    ax.add_patch(rect)
    rect = patches.Rectangle((0, -24), 48, 48, linewidth=3, edgecolor="brown", facecolor="none")
    ax.add_patch(rect)
    ax.set_aspect('equal')
    a = ax.scatter(coords[:, 0], coords[:, 1], c=cols, s=0.1, zorder=50)
    plt.xlim(0, 48)
    plt.ylim(-24, 24)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"ant_maze_options.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')


def plot_terminations(ax, probs, coords, grid, title_suffix="", colorbar_size='10%'):
    create_grid_plot(ax, grid)
    grid = np.array(grid, float)
    mat = create_grid_plot_values(ax, grid, "OrRd", coords, probs.numpy())
    return mat


def plot_both(ax, arrow_data, max_index, coords, termination_probs, grid, values=True, headwidth=9, headlength=20, colorbar_size='10%'):
    create_grid_plot(ax, grid)
    mat = create_grid_plot_values(ax, grid, "Reds", coords, termination_probs.numpy())
    x_pos, y_pos, x_dir, y_dir, color = arrow_data
    # for i in range(len(x_pos)):
    #     if max_index[i] == 5:
    #         rect = patches.Polygon(np.array([[x_pos[i] + 0.2, y_pos[i] + 0.2],
    #                                                [x_pos[i] - 0.2, y_pos[i] + 0.2],
    #                                                [x_pos[i] - 0.2, y_pos[i] - 0.2],
    #                                                [x_pos[i] + 0.2, y_pos[i] - 0.2]]),
    #                                      edgecolor=plt.get_cmap("viridis")(color[i]), facecolor=plt.get_cmap("viridis")(color[i]))
    #         ax.add_patch(rect)
    quiv = ax.quiver(x_pos, y_pos, x_dir, y_dir, color, cmap=plt.get_cmap("viridis"),
                     norm=colors.Normalize(vmin=color.min(), vmax=color.max()), angles='xy', scale_units='xy',
                     scale=1, pivot='middle', clim=(0.3, 1), headwidth=headwidth, headaxislength=headlength, headlength=headlength)# width=0.1)
    # divider = make_axes_locatable(ax)
    # plt.colorbar(quiv, cax=cax, ax=ax, format=FuncFormatter(lambda y, _: '{:.0%}'.format(y)), ticks=np.arange(0.3, 1.1, 0.1))
    # ax.set_title(("Maximum likelihood actions in states" + title_suffix))


def plot_policy(ax, arrow_data, grid, title_suffix="", values=True, headwidth=9, headlength=20, colorbar_size='10%', max_index=None):
    x_pos, y_pos, x_dir, y_dir, color = arrow_data
    norm = colors.Normalize(vmin=color.min(), vmax=color.max())
    cmap = cm.get_cmap('viridis')
    quiv = ax.quiver(x_pos, y_pos, x_dir, y_dir, color, cmap=cmap,
                     norm=norm, angles='xy', scale_units='xy',
                     scale=1, pivot='middle', clim=(0.3, 1), headwidth=headwidth, headaxislength=headlength, headlength=headlength)# width=0.1)
    for i in range(len(max_index)):
        if max_index[i] != 5 or (x_pos[i] == 0 and y_pos[i] == 0):
            continue
        x = x_pos[i]
        y = y_pos[i]
        rect = patches.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5, linewidth=2, edgecolor=cmap(norm(color[i])), facecolor=cmap(norm(color[i])))
        ax.add_patch(rect)
    return quiv
    # divider = make_axes_locatable(ax)

    # cax = divider.append_axes("right", size=colorbar_size, pad=0.025)
    # plt.colorbar(quiv, cax=cax, ax=ax, format=FuncFormatter(lambda y, _: '{:.0%}'.format(y)), ticks=np.arange(0.3, 1.1, 0.1))


def create_grid_plot(ax, grid, color_map="binary"):
    if color_map == "binary":
        grid = 1 - grid
    size_y = grid.shape[0]
    size_x = grid.shape[1]
    vmax = max(float(np.max(grid)), 1)
    vmin = 0

    mat = ax.matshow(np.flip(grid, 0), cmap=plt.get_cmap(color_map), extent=[0, size_x, 0, size_y], vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(0, size_x))
    ax.set_xticks(np.arange(0.5, size_x + 0.5), minor=True)
    # ax.set_xticklabels(np.arange(0, size_x), minor=True)
    plt.setp(ax.get_xmajorticklabels(), visible=False)
    ax.set_yticks(np.arange(0, size_y))
    ax.set_yticks(np.arange(0.5, size_y + 0.5), minor=True)
    # ax.set_yticklabels(np.arange(0, size_y), minor=True)
    ax.invert_yaxis()
    plt.setp(ax.get_ymajorticklabels(), visible=False)

    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(color="black")
    return mat


def create_grid_plot_values(ax, grid, color_map, coords, probs):
    grid = np.ma.masked_array(grid, grid == 0, dtype=np.float)
    for i in range(len(coords)):
        if grid[coords[i][0], coords[i][1]] != 0:
            x = coords[i][1]
            y = coords[i][0]
            grid[y][x] = probs[i]
            # if probs[i] < 0.9 * max(probs.detach().numpy()):
            #     color = "blue"
            # else:
            #     color = "deepskyblue"
            # ax.text(x + 0.5, y + 0.5, "{}".format(int(probs[i] * 100)), horizontalalignment='center',
            #         verticalalignment='center', color="black", fontsize=7)
    return create_grid_plot(ax, grid, color_map=color_map)


def meta_train_plots():
    fig, axes = plt.subplots(ncols=2, nrows=1, dpi=1000, figsize=(24, 9))
    taxi_metatrain_plots(ax=axes[0])
    ant_maze_metatrain_plots(ax=axes[1], env_type="noreset")
    plt.tight_layout()
    plt.savefig(f"meta_train_performance_avg.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')




if __name__ == '__main__':
    nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 20,
        "font.size": 20,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
    mpl.rcParams.update(nice_fonts)
    # taxi_avg()
    # taxi_hyperparams()
    # taxi_ablation()
    # taxi_per_env_plots()

    # taxi_options_plot_policy()
    # taxi_options_plot_terminations()
    # taxi_options_plot()
    # ant_maze_plots_test("noreset")
    # ant_maze_plots("original")
    # ant_maze_per_env_plots("noreset")
    # ant_maze_per_env_plots_test("noreset")
    # ant_maze_options()
    # ant_maze_metatrain_plots("noreset")
    meta_train_plots()
    # taxi_plots()
