import numpy as np
import os
import re
import torch
from mpi4py import MPI
from core.baselines import ZeroBaseline, LinearFeatureBaseline
from envs.ml_simple_bandits_env import SimpleBandits
from envs.ml_ant_obstaclesgen import AntObstaclesGenEnv
from envs.taxi import Taxi
from torch.optim import SGD
from gym.spaces import Discrete, Box
from gym.wrappers.time_limit import TimeLimit
from core.policies import LearnTermOptionsPolicy, TimeTermOptionsPolicy, SingleOptPolicy
from core.algorithms import MetaGradientAlg
from utils.utils import set_seeds, get_space_io_size
from utils.mpi_adam import MpiAdam
from utils.custom_types import NNLayerSizes
from torch.nn.functional import softplus


def initialize_algorithm(opts=None):
    # Set seeds
    set_seeds(opts.seed + opts.rank)

    if opts.continue_run is not None:
        run_data_dir = opts.run_data_dir
        run_name = opts.run_name
        rank = opts.rank

        load_all_path = opts.continue_run
        checkpoint = torch.load(opts.continue_run)
        opts = checkpoint['opts']
        opts.continue_run = load_all_path
        opts.epoch_start = int(re.search(r'\d+', os.path.basename(load_all_path)).group()) + 1

        opts.run_data_dir = run_data_dir
        opts.run_name = run_name
        opts.rank = rank

    # Initialize env
    if "taxi" == opts.env:
        env = Taxi(exclude_envs=opts.exclude_envs, random_move_prob=opts.random_move_prob)
    elif "ant_maze" == opts.env:
        env = TimeLimit(env=AntObstaclesGenEnv(exclude_envs=opts.exclude_envs), max_episode_steps=1000)
    elif "ant_maze_noreset" == opts.env:
        env = TimeLimit(env=AntObstaclesGenEnv(exclude_envs=opts.exclude_envs, enable_resets=False), max_episode_steps=1000)
    else:
        raise RuntimeError(f"Unknown environment specified: {opts.env}")

    # Get S(O) dim and A dim (discrete envs have different dim an io_size)
    obs_size = get_space_io_size(env.observation_space)
    act_size = get_space_io_size(env.action_space)
    if isinstance(env.action_space, Discrete):
        action_type = "discrete"
    elif isinstance(env.action_space, Box):
        action_type = "continuous"
    else:
        raise RuntimeError("Unknown action space type")

    # Initialize policy
    if opts.continue_run is not None:
        policy = create_policy(opts=opts, obs_size=obs_size, act_size=act_size, action_type=action_type,
                               checkpoint_path=opts.continue_run)
    elif opts.load_chp is not None:
        policy = create_policy(opts=opts, obs_size=obs_size, act_size=act_size, action_type=action_type,
                               checkpoint_path=opts.load_chp)
    else:
        policy = create_policy(opts=opts, obs_size=obs_size, action_type=action_type, act_size=act_size)

    # Create outer optimizer
    if opts.continue_run is not None:
        optimizer = create_optimizer(opts=opts, policy=policy, checkpoint_path=opts.continue_run)
    elif opts.load_chp is not None and opts.load_optimizer:
        optimizer = create_optimizer(opts=opts, policy=policy, checkpoint_path=opts.load_chp)
    else:
        optimizer = create_optimizer(opts=opts, policy=policy)

    # Initialize baseline
    if opts.baseline == "none":
        baseline = ZeroBaseline()
    elif opts.baseline == "linear":
        baseline = LinearFeatureBaseline()
    else:
        raise RuntimeError("Baseline {} not supported use none/linear instead".format(opts.baseline))

    # Initialize the algorithm
    algorithm = MetaGradientAlg(
        env=env,
        policy=policy,
        baseline=baseline,
        optimizer=optimizer,
        lookaheads=opts.lookaheads,
        episodes=opts.episodes,
        envs_per_process=opts.envs_per_process,
        fixed_env=opts.fixed_env,
        mpi_rank=opts.rank,
        learn_lr_inner=opts.learn_lr_inner,
        lr_inner=opts.lr_inner,
        save_trajs=opts.save_trajs,
        visualize=opts.visualize,
        base_dir=opts.base_dir,
        env_name=opts.env,
        return_discount=opts.return_discount,
        gae_discount=opts.gae_discount,
        dice_discount=opts.dice_discount,
        entropy_reg=opts.entropy_reg,
        log_level=opts.log_level,
        grad_update_type=opts.grad_update_type,
        normalize_advs=opts.normalize_advs
    )
    # Synchronize model parameters across multiple cores
    policy.synchronize(comm=MPI.COMM_WORLD)

    # Load RNG
    if opts.continue_run is not None:
        load_rng(opts.continue_run)
    elif opts.load_chp is not None and opts.load_rng:
        load_rng(opts.load_chp)

    return algorithm, opts


def load_rng(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    pytorch_rng_states = checkpoint["pytorch_rng_states"]
    numpy_rng_states = checkpoint["numpy_rng_states"]
    if MPI is not None:
        if len(pytorch_rng_states) == MPI.COMM_WORLD.Get_size() and len(numpy_rng_states) == MPI.COMM_WORLD.Get_size():
            pytorch_rng_state = MPI.COMM_WORLD.scatter(pytorch_rng_states, root=0)
            numpy_rng_state = MPI.COMM_WORLD.scatter(numpy_rng_states, root=0)
            torch.set_rng_state(pytorch_rng_state)
            np.random.set_state(numpy_rng_state)
        else:
            raise RuntimeError(f"Len of rng_states does not match the comm size: {len(pytorch_rng_states)}, "
                               f"{len(numpy_rng_states)}, {MPI.COMM_WORLD.Get_size()}")
    else:
        torch.set_rng_state(pytorch_rng_states[0])
        np.random.set_state(numpy_rng_states[0])


def create_policy(opts, obs_size, act_size, action_type, checkpoint_path=None):
    kwargs = {
        "obs_dim": obs_size,
        "action_dim": act_size,
        "options": opts.options,
        "hidden_sizes_base": NNLayerSizes(tuple(opts.hidden_sizes_base)),
        "hidden_sizes_option": NNLayerSizes(tuple(opts.hidden_sizes_option)),
        "hidden_sizes_termination": NNLayerSizes(tuple(opts.hidden_sizes_termination)),
        "hidden_sizes_subpolicy": NNLayerSizes(tuple(opts.hidden_sizes_subpolicy)),
        "nonlinearity": torch.tanh,
        "no_bias": opts.no_bias,
        "action_type": action_type,
        "std_value": opts.std_value,
        "std_type": opts.std_type,
        "learn_lr_inner": opts.learn_lr_inner,
        "lr_inner": opts.lr_inner,
        "term_time": opts.term_time,
        "temp_options": opts.temperature_options,
        "temp_terminations": opts.temperature_terminations,
        "termination_prior": opts.termination_prior,
        "adapt_options": opts.adapt_options
    }
    if opts.policy_type == "ltopt":
        assert opts.options is not None
        policy = LearnTermOptionsPolicy(**kwargs)
    elif opts.policy_type == "ttopt":
        assert opts.options is not None
        assert opts.term_time is not None
        policy = TimeTermOptionsPolicy(**kwargs)
    elif opts.policy_type == "noopt":
        policy = SingleOptPolicy(**kwargs)
    else:
        raise NotImplementedError(f"Policy type {opts.policy_type} is not implemented")
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        if opts.env == "taxi" and "subpolicylogstd" in checkpoint["policy_params"].keys():
            del checkpoint["policy_params"]["subpolicylogstd"]
        policy.load_state_dict(checkpoint["policy_params"])
    return policy


def create_optimizer(opts, policy, checkpoint_path=None):
    if opts.learn_params == "outer":
        optimizer = MpiAdam([{"params": policy.outer_params.values(),
                              "lr": opts.lr_outer}])
    elif opts.learn_params == "all":
        optimizer = MpiAdam([{"params": policy.subpolicy_params.values(),
                              "lr": opts.lr_outer},
                             {"params": policy.base_params.values(),
                              "lr": opts.lr_outer},
                             {"params": policy.termination_params.values(),
                              "lr": opts.lr_outer},
                             {"params": policy.option_params.values(),
                              "lr": opts.lr_outer}
                             ])
    elif opts.learn_params == "inner":
        if opts.learn_lr_inner:
            param_names = policy.inner_params.keys()
            optimizer = SGD([{"params": policy.inner_params[i], "lr": softplus(policy.lr_params[i+"lr"])}
                             for i in param_names])
        else:
            optimizer = SGD([{"params": policy.inner_params.values(),
                              "lr": opts.lr_inner}])
    elif opts.learn_params == "inner_adam":
        optimizer = MpiAdam([{"params": policy.inner_params.values(),
                              "lr": opts.lr_inner}])
    elif opts.learn_params == "outer_sgd":
        optimizer = SGD([{"params": policy.outer_params.values(),
                          "lr": opts.lr_outer}])
    else:
        raise RuntimeError("learn_params \"{}\" not supported".format(opts.learn_params))
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer_params"])
    return optimizer
