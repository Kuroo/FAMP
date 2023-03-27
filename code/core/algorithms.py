import os
import sys
from tqdm import tqdm
import torch

from envs.ml_wrapper_env import MetaLearningEnv
from torch.nn import functional as F
from collections import OrderedDict
from utils.plotting import *
from core.policies import NamedParamsNNPolicy
from core.baselines import Baseline
from statistics import mean
from utils.logger import logkv_mean
from mpi4py import MPI
from utils.custom_types import TrajDataDict, ParamDict
from utils.utils import get_space_dim, GradUpdateType
from utils.pytorch_modules import DiscountedSumForward
from utils.mpi_adam import MpiAdam
from core.sampling import generate_samples
from core.sample_processing import discounted_cumsum, calculate_gae_advantages


class MetaGradientAlg(object):

    def __init__(
        self,
        env: MetaLearningEnv,
        policy: NamedParamsNNPolicy,
        baseline: Baseline,
        optimizer: torch.optim.Optimizer,
        lookaheads: int,
        episodes: int,
        envs_per_process: int,
        fixed_env: int,
        mpi_rank: int,
        learn_lr_inner: bool,
        lr_inner: float,
        save_trajs: bool,
        visualize: bool,
        base_dir: str,
        env_name: str,
        return_discount: float,
        gae_discount: float,
        dice_discount: float,
        entropy_reg: float,
        log_level: int,
        normalize_advs: bool,
        grad_update_type: GradUpdateType
    ):
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.normalize_advs = normalize_advs
        self.grad_update_type = grad_update_type
        self.envs_per_process = envs_per_process
        self.fixed_env = fixed_env

        self.lookaheads = lookaheads
        self.episodes = episodes
        self.mpi_rank = mpi_rank

        self.learn_lr_inner = learn_lr_inner
        self.lr_inner = lr_inner

        self.return_discount = return_discount
        self.gae_discount = gae_discount
        self.dice_discount = dice_discount
        self.entropy_reg = entropy_reg

        self.save_trajs = save_trajs
        self.visualize = visualize
        self.base_dir = base_dir
        self.env_name = env_name

        self.log_level = log_level
        self.policy_optimizer = optimizer

    def epoch(self):
        # Set policy gradients to zero
        self.policy_optimizer.zero_grad()


        if self.grad_update_type == GradUpdateType.MULTI:
            # For multi each env is sampled exactly 1 time
            n_envs_to_sample = len(self.env.envs_to_sample)
            range_start = self.mpi_rank
            range_step = 1 if MPI is None else MPI.COMM_WORLD.Get_size()
            for index in range(range_start, n_envs_to_sample, range_step):
                # Accumulate outer grads for this env
                self.env_grads(task=self.env.envs_to_sample[index], env_count_normalizer=n_envs_to_sample)
        else:
            # For each environment sample from env distribution
            for env_sample in tqdm(range(self.envs_per_process), desc=f"Process {self.mpi_rank}",
                                   position=self.mpi_rank, file=sys.stdout, disable=True):
                task = self.fixed_env if self.fixed_env >= 0 else self.env.sample_task()
                n_envs_to_sample = (1 if MPI is None else MPI.COMM_WORLD.Get_size()) * self.envs_per_process
                # Accumulate outer grads for this env
                self.env_grads(task=task, env_count_normalizer=n_envs_to_sample)

        # Update the initial policy params
        if isinstance(self.policy_optimizer, MpiAdam):
            self.policy_optimizer.step(comm=None if MPI is None else MPI.COMM_WORLD)
        else:
            self.policy_optimizer.step()

    def env_grads(self, task, env_count_normalizer):
        self.env.set_task(task)
        current_params = OrderedDict(self.policy.named_parameters())
        current_inner_params = self.policy.inner_params
        for lookahead in range(self.lookaheads):
            loss, traj_data = self.get_update_data(params=current_params)

            # Get updated inner parameters
            lr = self.policy.lr_params if self.learn_lr_inner else self.lr_inner
            new_inner_params = self.sgd_update(loss=loss, lr=lr, params=current_inner_params)

            current_params = self.policy.outer_params.copy()
            current_params.update(new_inner_params)
            current_inner_params = new_inner_params
            self._log_traj_data(traj_data=traj_data, task=task, prefix=f"Step{lookahead:02}")

        # Calculate outer loss
        loss, traj_data = self.get_update_data(params=current_params)
        loss /= env_count_normalizer

        if self.save_trajs:
            with open(self.base_dir + "/logs/trajs.pickle", 'wb') as handle:
                pickle.dump(traj_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Accumulate grads
        loss.backward()
        self._log_traj_data(traj_data=traj_data, task=task, prefix=f"Step{self.lookaheads:02}")

    @staticmethod
    def loaded_dice(log_probs, advantages, lam, entropies, beta):
        first_term = DiscountedSumForward.apply(log_probs, lam)
        second_term = first_term - log_probs
        deps_first = torch.exp(first_term - first_term.detach())
        deps_second = torch.exp(second_term - second_term.detach())
        total = deps_first * advantages + deps_second * (beta * entropies - advantages)
        return -torch.sum(total, dim=0, keepdim=True)

    def get_update_data(self, params=None):
        # Generate trajectory data
        traj_data = TrajDataDict(generate_samples(env=self.env, policy=self.policy,
                                                  episodes=self.episodes, params=params, visualize=self.visualize))

        # Get log probs
        policy_update_data = self.policy.get_update_data(traj_data, params=params)
        traj_data["action_log_probs"], traj_data["posterior_option_probs"], traj_data["entropies"] = policy_update_data

        # Calculate discounted returns
        traj_data["discounted_returns"] = [discounted_cumsum(r, discount=self.return_discount)
                                           for r in traj_data["rewards"]]

        # Fit the baseline and get V(s)
        cat_obs = torch.cat([traj_data["observations"][i] for i in range(len(traj_data["observations"]))], dim=0)
        cat_timesteps = torch.cat([traj_data["timesteps"][i] for i in range(len(traj_data["timesteps"]))], dim=0)
        self.baseline.update(obs=cat_obs, timesteps=cat_timesteps, disc_rets=torch.cat(traj_data["discounted_returns"]))
        traj_data["baselines"] = [self.baseline.predict(obs=o, timesteps=t)
                                  for o, t in zip(traj_data["observations"], traj_data["timesteps"])]

        # Calculate advantages
        calculate_gae_advantages(traj_data=traj_data, return_discount=self.return_discount,
                                 gae_discount=self.gae_discount, normalize_adv=self.normalize_advs)

        traj_data["loss"] = list(map(lambda p, q, r: self.loaded_dice(p, q, lam=self.dice_discount,
                                                                      entropies=r, beta=self.entropy_reg),
                                     traj_data["action_log_probs"],
                                     traj_data["advantages"],
                                     traj_data["entropies"]))

        loss = torch.mean(torch.cat(traj_data["loss"]))
        return loss, traj_data

    @staticmethod
    def sgd_update(loss: torch.Tensor, lr: float, params: ParamDict, create_graph: bool = True) -> ParamDict:
        """
        Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """
        grads = torch.autograd.grad(loss, params.values(), create_graph=create_graph,  allow_unused=True)
        updated_params = OrderedDict()
        if isinstance(lr, OrderedDict):
            for name, param, grad in zip(params.keys(), params.values(), grads):
                updated_params[name] = param - F.softplus(lr[f"{name}lr"]) * grad
        else:
            for name, param, grad in zip(params.keys(), params.values(), grads):
                updated_params[name] = param - lr * grad

        return ParamDict(updated_params)

    def plot_epoch(self, name_prefix="", plot_updates=True):
        if "taxi" not in self.env_name and "grid" not in self.env_name:
            return None
        if self.mpi_rank == 0:
            os.makedirs(name_prefix, exist_ok=True)

        if MPI is not None:
            MPI.COMM_WORLD.Barrier()
        if self.env.one_hot:
            inputs = torch.eye(int(get_space_dim(self.env.observation_space)))
        else:
            raise NotImplementedError("TODO implement the coordinate input gen")
        if self.mpi_rank == 0:
            with torch.no_grad():
                current_params = OrderedDict(self.policy.named_parameters())
                params = self.policy(inputs, current_params)
                opt_params = params[0]
                policy_params = params[1]
                policy_probs = F.softmax(policy_params / self.policy.temp_act, dim=-1)
                option_probs = opt_params.exp()
                term_probs = params[2] if len(params) > 2 else option_probs
            torch.save({"option_probs": option_probs, "termination_probs": term_probs,
                        "policy_probs": policy_probs}, name_prefix + "pre_update_data")
            self.env.plot_params(option_probs=option_probs, termination_probs=term_probs, policy_probs=policy_probs,
                                 name_prefix=name_prefix + "pre_update", no_blocks=True)

        if plot_updates:
            processes = 1 if MPI is None else MPI.COMM_WORLD.Get_size()
            for current_task in range(self.mpi_rank, self.env.n_envs, processes):
                self.env.set_task(current_task)

                current_params = OrderedDict(self.policy.named_parameters())
                current_inner_params = self.policy.inner_params
                for lookahead in range(self.lookaheads):
                    loss, traj_data = self.get_update_data(params=current_params)

                    # Get updated parameters
                    lr = self.policy.lr_params if self.learn_lr_inner else self.lr_inner
                    new_inner_params = self.sgd_update(loss=loss, lr=lr, params=current_inner_params, create_graph=False)
                    current_params = self.policy.outer_params.copy()
                    current_params.update(new_inner_params)
                    current_inner_params = new_inner_params

                with torch.no_grad():
                    params = self.policy(inputs, params=current_params)
                    opt_params = params[0]
                    policy_params = params[1]
                    policy_probs = F.softmax(policy_params / self.policy.temp_act, dim=-1)
                    option_probs = opt_params.exp()
                    term_probs = params[2] if len(params) > 2 else option_probs
                torch.save({"option_probs": option_probs, "termination_probs": term_probs,
                            "policy_probs": policy_probs},
                           name_prefix + "env{}_data".format(current_task))
                self.env.plot_params(option_probs=option_probs, termination_probs=term_probs, policy_probs=policy_probs,
                                     name_prefix=name_prefix + "env{}".format(current_task), no_blocks=False)
        plt.close("all")

    def _log_traj_data(self, traj_data, task, prefix=""):
            if self.log_level >= 1:
                avg_episode_length = mean(map(len, traj_data["observations"]))
                avg_discounted_return = mean(map(lambda p: p[0].item(), traj_data["discounted_returns"]))
                avg_return = mean(map(lambda p: p.sum().item(), traj_data["rewards"]))
                avg_terminations = mean(map(lambda p: p.sum().item() / len(p), traj_data["terminations"]))
                keys = ["EpisodeLength", "DiscountedReturn", "Return", "Terminations"]
                values = [avg_episode_length, avg_discounted_return, avg_return, avg_terminations]
                for option in range(self.policy.options):
                    avg_option_usage = mean(map(lambda p: (p == option).sum().float().item() / len(p), traj_data["options"]))
                    keys.append(f"Option{option}Usage")
                    values.append(avg_option_usage)
                log_dict = dict(zip(keys, values))

                for key, val in log_dict.items():
                    logkv_mean(f"{prefix}{key}/Avg", val)
                    logkv_mean(f"{prefix}{key}/Env{task}", val)
