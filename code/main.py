import argparse
import numpy as np
import os
import torch
from mpi4py import MPI
from utils import logger
from core.initalization import initialize_algorithm
from utils.utils import GradUpdateType
import shutil


def main(opts=None):
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        opts.base_dir = f"{opts.run_data_dir}/{run_name(opts)}"
        if os.path.exists(f"{opts.base_dir}/logs"):
            shutil.rmtree(f"{opts.base_dir}/logs")
        os.makedirs(f"{opts.base_dir}/logs", exist_ok=False)
        data = [opts.base_dir for _ in range(MPI.COMM_WORLD.Get_size())]
    else:
        data = None
    if MPI is not None:
        opts.base_dir = MPI.COMM_WORLD.scatter(data, root=0)

    algorithm, opts = initialize_algorithm(opts)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure(dir=f"{opts.base_dir}/logs")
        logger.info("\nRun parameters:")
        for key, val in vars(opts).items():
            logger.info(str(key) + ": " + str(val))
        processes = 1 if MPI is None else MPI.COMM_WORLD.Get_size()
        logger.info(f"Running {processes} processes")
    else:
        logger.configure(dir=f"{opts.base_dir}/logs", format_strs=[])

    run_algorithm(opts, algorithm)


def run_algorithm(opts, algorithm):
    epoch_digits = len(str(opts.epochs))
    for epoch in range(opts.epoch_start, opts.epoch_start + opts.epochs):
        # Plot initial epoch
        if epoch == opts.epoch_start:
            algorithm.plot_epoch(name_prefix=opts.base_dir + "/plots/"
                                             + "epoch{:0{prec}d}/".format(epoch-1, prec=epoch_digits),
                                 plot_updates=False)
        algorithm.epoch()

        # Average data across processes and log
        if MPI is not None:
            logger.merge_thread_data(comm=MPI.COMM_WORLD)

        logger.log(f"\nEpoch {epoch}:")
        logger.dumpkvs()
        logger.increment_global_step()

        # Plotting (only for taxi)
        if epoch % opts.plot_freq == 0:
            algorithm.plot_epoch(name_prefix=opts.base_dir + "/plots/"
                                             + "epoch{:0{prec}d}/".format(epoch, prec=epoch_digits),
                                                                          plot_updates=False)

        # Checkpoint
        if (epoch % opts.chp_freq == 0) or (epoch == opts.epochs):
            if MPI is not None:
                pytorch_rng_states = MPI.COMM_WORLD.gather(torch.get_rng_state(), root=0)
                numpy_rng_states = MPI.COMM_WORLD.gather(np.random.get_state(), root=0)
            else:
                pytorch_rng_states = [torch.get_rng_state()]
                numpy_rng_states = [np.random.get_state()]
            if opts.rank == 0:
                state_dicts = {
                    "epoch": epoch,
                    "policy_params": algorithm.policy.state_dict(),
                    "optimizer_params": algorithm.policy_optimizer.state_dict(),
                    "opts": opts,
                    "pytorch_rng_states": pytorch_rng_states,
                    "numpy_rng_states": numpy_rng_states
                }
                save_dir = opts.base_dir +"/checkpoints/"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    state_dicts,
                    save_dir + "epoch{:0{prec}d}.tar".format(epoch, prec=epoch_digits)
                )


def run_name(opts):
    return f"{opts.run_name}"
    # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # if opts.run_name == "":
    #     return f"{opts.grad_update_type}_{opts.env}_{opts.baseline}_{opts.seed}_{timestamp}"
    # else:
    #     return f"{opts.run_name}_{opts.grad_update_type}_{opts.env}_{opts.baseline}_{opts.seed}_{timestamp}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FAMP experiments program")

    # Algorithm
    parser.add_argument('--epochs', type=int, default=200, help='Number of algorithm epochs (outer updates)')
    parser.add_argument('--envs_per_process', type=int, default=15,
                        help='Total number of envs = num_processes x envs_per_process')
    parser.add_argument('--episodes', type=int, default=20,
                        help='How many episodes should be sampled per inner update')
    parser.add_argument('--lookaheads', type=int, default=1,
                        help='How many gradient steps to optimize for (best performance after lookaheads updates)')
    parser.add_argument('--grad_update_type', type=GradUpdateType.argparse, default=GradUpdateType.META,
                        choices=list(GradUpdateType), help='Which update to use in algorithm loop '
                                                           '(meta for training, single for test)')

    # Environment options
    parser.add_argument('--env', type=str, default="taxi",
                        choices=("taxi", "ant_maze", "ant_maze_noreset"))
    parser.add_argument('--exclude_envs', type=int, nargs='*', default=[],
        help="Environments that should be excluded from environment distribution. "
             "Used to remove test envs during training.")
    parser.add_argument('--fixed_env', type=int, default=-1,
                        help='Used to manually set environment to one concrete val.')
    parser.add_argument('--random_move_prob', type=float, default=0.0, help='Random move prob in taxi/grid environment')

    # RL settings
    parser.add_argument('--gae_discount', type=float, default=0.98, help='GAE lambda parameter for advantage estimator')
    parser.add_argument('--dice_discount', type=float, default=0, help='Past influence discount for loaded dice')
    parser.add_argument('--return_discount', type=float, default=0.95, help='Discount factor for returns')
    parser.add_argument('--baseline', type=str, default="none", choices=("none", "linear"),
                        help='Baseline to be used for loss calculation')
    parser.add_argument('--entropy_reg', type=float, default=0, help='Entropy regularization coefficient')

    # Optimizers/Learning rates
    parser.add_argument('--lr_outer', type=float, default=1e-1, help='Outer learning rate')
    parser.add_argument('--lr_inner', type=float, default=10.0, help='Inner learning rate')
    parser.add_argument('--learn_params', type=str, default="outer", choices=("outer", "all", "inner", "inner_adam",
    "outer_sgd") , help='Which parts of hierarchical policy to optimize.')
    parser.add_argument('--adapt_options', action='store_true', help='If true, adapts sub-pols and terms in inner '
                                                                     'updates')
    parser.add_argument('--learn_lr_inner', action='store_true', help='Learn inner learning rate in the outer loop')

    # Policy options
    parser.add_argument('--policy_type', type=str, default="ltopt", choices=("ltopt", "ttopt", "noopt"),
                        help='Choose options policy with time-based/learned terminations or policy without options')
    parser.add_argument('--options', type=int, help='Number of options')
    parser.add_argument('--hidden_sizes_base', type=int, nargs='*', default=[], help='Layer sizes for shared layers')
    parser.add_argument('--hidden_sizes_option', type=int, nargs='*',  default=[], help='Layer sizes for options')
    parser.add_argument('--hidden_sizes_subpolicy', type=int, nargs='*', default=[], help='Layer sizes for subpolicy')
    parser.add_argument('--hidden_sizes_termination', type=int, nargs='*',  default=[],
                        help='Layer sizes for terminations')
    parser.add_argument('--term_time', type=int, help='Termination time for fixed term policy')
    parser.add_argument('--std_type', type=str, default='diagonal', choices=('diagonal', 'single'),
                        help='Use fixed standard deviation for policy')
    parser.add_argument('--std_value', type=float, default=1.0, help='Initial value of standard deviation')
    parser.add_argument('--no_bias', action='store_true', help='Do not use bias for policy networks')
    parser.add_argument('--normalize_advs', action='store_true', help='Use advantage normalization for updates')
    parser.add_argument('--termination_prior', type=float, default=-1, help='Prior termination probability')

    # Seeding
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')

    # Logging, plotting, saving, visualization
    parser.add_argument('--run_data_dir', type=str, default='../run_data', help='Path to directory with run data')
    parser.add_argument('--run_name', type=str, default='test', help='Name to identify the run')
    parser.add_argument('--plot_freq', type=int, default=20, help='Plotting frequency (in episodes)')
    parser.add_argument('--chp_freq', type=int, default=20, help='Checkpoint frequency (in episodes)')
    parser.add_argument('--save_trajs', action='store_true', help='Use pickle to save trajectories (for heatmaps')
    parser.add_argument('--visualize', action='store_true', help='Renders episodes during sampling.')

    # Load model
    parser.add_argument('--load_chp', type=str, nargs='?', help='Path to checkpoint')
    parser.add_argument('--load_rng', action='store_true', help='Load RNG states')
    parser.add_argument('--continue_run', type=str, nargs='?', help='Continue run')
    parser.add_argument('--load_optimizer', type=str, nargs='?', help='Load optimizer state')

    # Used as default values
    parser.add_argument('--log_level', type=int, default=1)
    parser.add_argument('--temperature_options', type=float, default=1, help='Temp for softmax policy over options')
    parser.add_argument('--temperature_terminations', type=float, default=1, help='Temp for sigmoid terminations')
    parser.add_argument('--max_option_prob', type=float, default=1, help='Max probability of selecting and option')

    opts = parser.parse_args()

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        opts.rank = 0
    else:
        opts.rank = MPI.COMM_WORLD.Get_rank()
    opts.epoch_start = 1
    main(opts)


