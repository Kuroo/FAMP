import torch
import gym
from time import sleep
from collections import defaultdict
from core.policies import NamedParamsNNPolicy
from utils.custom_types import ParamDict, TrajDataDict


def generate_samples(env: gym.Env, policy: NamedParamsNNPolicy, episodes: int, params: ParamDict = None,
                     visualize: bool = False) -> TrajDataDict:
    """
    Generate samples from the environment with given policy
    """
    traj_data: dict[str, list] = defaultdict(list)
    traj: dict[str, list] = defaultdict(list)

    # Reset the environment to get the initial observation
    obs = torch.tensor([env.reset()], dtype=torch.get_default_dtype())
    context = policy.reset_context()
    context["timestep"] = 0
    episodes_done = 0

    while episodes_done < episodes:
        # Get action from policy
        with torch.no_grad():
            # Use context to keep track of options/terminations/time etc. based on policy
            action, new_context = policy.get_action(obs=obs, context=context, params=params)

        # Perform an action
        next_obs, reward, done, env_infos = env.step(action.numpy())
        if visualize:
            env.render()

        # Store data in buffers
        traj["observations"].append(obs)
        traj["actions"].append(action)
        traj["rewards"].append(reward)
        traj["dones"].append(1 if done else 0)
        # Store context variables in buffers
        for context_name, val in new_context.items():
            traj[f"{context_name}s"].append(val)

        # If done reset otherwise use the new observation
        if done:
            episodes_done += 1
            traj_data["observations"].append(torch.cat(traj["observations"], dim=0))
            traj_data["actions"].append(torch.cat(traj["actions"], dim=0))
            traj_data["rewards"].append(torch.tensor(traj["rewards"], dtype=torch.get_default_dtype()))
            traj_data["dones"].append(torch.tensor(traj["dones"], dtype=torch.uint8))

            for context_name, val in new_context.items():
                traj_data[f"{context_name}s"].append(torch.tensor(traj[f"{context_name}s"], dtype=torch.int))
            traj_data["timesteps"].append(torch.arange(context["timestep"] + 1)[:, None])

            # Reset context and env and clear buffers
            context = policy.reset_context()
            context["timestep"] = 0
            obs = torch.tensor([env.reset()], dtype=torch.get_default_dtype())
            traj: dict[str, list] = defaultdict(list)
        else:
            obs = torch.tensor([next_obs], dtype=torch.get_default_dtype())
            new_context["timestep"] = context["timestep"] + 1
            context = new_context

    return TrajDataDict(traj_data)
