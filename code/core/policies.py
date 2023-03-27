import mpi4py.MPI
import torch
import torch.autograd
from torch import nn
from torch.distributions import Independent, Normal
from torch.distributions.utils import probs_to_logits
from torch.nn import functional as F
from typing import Union
from abc import ABC, abstractmethod
from collections import OrderedDict
from utils.utils import weighted_logsumexp, inv_softplus
from utils.custom_types import ParamDict, TrajDataDict, PolicyContext, NNLayerSizes
import numpy as np
from mpi4py import MPI


class PolicyHead(ABC):

    def __init__(self, output_dim: int) -> None:
        self.output_dim = output_dim

    @abstractmethod
    def sample_action(self, policy_params: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_log_probs(self, policy_params: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DiagMVNHead(PolicyHead):

    def __init__(self, output_dim: int) -> None:
        PolicyHead.__init__(self, output_dim=output_dim)

    def sample_action(self, policy_params: torch.Tensor) -> torch.Tensor:
        means = torch.tanh(policy_params[:, :self.output_dim])
        stds = torch.exp(policy_params[:, self.output_dim:])
        # shape 1 x action dim
        action = torch.normal(mean=means, std=stds)
        return action

    def get_distributions(self, policy_params: torch.Tensor) -> torch.distributions.Distribution:
        means = torch.tanh(policy_params[..., :self.output_dim])
        stds = torch.exp(policy_params[..., self.output_dim:])
        return Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)

    def get_log_probs(self, policy_params: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        means = torch.tanh(policy_params[..., :self.output_dim])
        stds = torch.exp(policy_params[..., self.output_dim:])
        # means shape batch x options x action_dim
        diagonal_mvn = Independent(Normal(loc=means, scale=stds),
                                   reinterpreted_batch_ndims=1)
        log_probs = diagonal_mvn.log_prob(actions[:, None, :])
        return log_probs


class CategoricalHead(PolicyHead):

    def __init__(self, output_dim: int, temperature: float) -> None:
        PolicyHead.__init__(self, output_dim=output_dim)
        self.temperature = temperature

    def sample_action(self, policy_params: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(policy_params / self.temperature, dim=-1)

        # Sample an action and return it together with new option
        return torch.multinomial(probs, num_samples=1)

    def get_log_probs(self, policy_params: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(policy_params / self.temperature, dim=-1)
        selected_log_probs = torch.gather(log_probs, dim=2,
                                          index=actions[:, :, None].expand(-1, log_probs.shape[1], 1)).squeeze(dim=2)
        return selected_log_probs


class NamedParamsNNPolicy(ABC, nn.Module):
    """
    Interface for policy with named parameters. These policies are used for lookahead gradients.
    """
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 nonlinearity: Union[torch.relu, torch.tanh],
                 bias: bool = True):
        nn.Module.__init__(self)
        self.bias = bias
        self._nonlinearity = nonlinearity
        self.inner_params = ParamDict(OrderedDict())
        self.outer_params = ParamDict(OrderedDict())
        self.all_params = ParamDict(OrderedDict())
        self._obs_dim = obs_dim
        self._act_dim = act_dim

    @abstractmethod
    def reset_context(self) -> PolicyContext:
        raise NotImplementedError

    @abstractmethod
    def get_action(self, obs: torch.Tensor, context: PolicyContext, params: ParamDict):
        raise NotImplementedError

    def get_update_data(self, traj_data: TrajDataDict, params: ParamDict = None) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if params is None:
            params = self.all_params

        act_log_probs, m_s, entropies = map(list, zip(*[self._single_traj_update_data(obs=p, actions=q, params=params)
                                                        for p, q in zip(traj_data["observations"],
                                                                        traj_data["actions"])]))
        return act_log_probs, m_s, entropies

    @abstractmethod
    def _single_traj_update_data(self,
                                 obs: torch.Tensor,
                                 actions: torch.Tensor,
                                 params: ParamDict = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def create_named_layers(self, layer_sizes: NNLayerSizes, param_dicts: tuple[OrderedDict, ...],
                            extra_dims: tuple[int, ...] = (), prefix: str = ""):
        """
        Creates named layers and add them as modules

        Args:
            layer_sizes (tuple): tuple with layer sizes (inp, sizes*, out)
            extra_dims (tuple): tuple with extra dimensions (eg. for options we use number of options)
            param_dicts (tuple): parameters can belong to multiple param dictionaries (example: inner_params, subpolicy)
            prefix (str): name of the part of the model
        """
        param_dicts = (*param_dicts, self.all_params)

        for i in range(1, len(layer_sizes)):
            weight_name = '{0}layer{1}weight'.format(prefix, i)
            weight_params = nn.Parameter(torch.Tensor(*extra_dims, layer_sizes[i - 1], layer_sizes[i]))
            weight = weight_params.data
            self.register_parameter(name=weight_name, param=weight_params)

            if self.bias:
                bias_name = '{0}layer{1}bias'.format(prefix, i)
                bias_params = nn.Parameter(torch.Tensor(*extra_dims, layer_sizes[i]))
                bias = bias_params.data
                self.register_parameter(name=bias_name, param=bias_params)
            else:
                bias = None

            self._reset_parameters(weight=weight, bias=bias)  # Initialize values

            for param_dict in param_dicts:
                param_dict[weight_name] = weight_params
                if self.bias:
                    param_dict[bias_name] = bias_params

    @staticmethod
    def named_forward_pass(inp: torch.Tensor, params: OrderedDict, num_layers: int,
                           nonlinearity, prefix="") -> torch.Tensor:
        """
        Performs a forward pass through the named part of the network

        Args:
            inp (torch.Tensor): input
            params (OrderedDict): a dictionary with parameters that should be used
            num_layers (int): number of layers in the named part
            nonlinearity : non-linearity function that should be used
            prefix (str): name of the part of the model

        Returns:
            Output from the named part of the model
        """
        output = inp
        for i in range(1, num_layers + 1):
            weight = params['{0}layer{1}weight'.format(prefix, i)]
            bias = params['{0}layer{1}bias'.format(prefix, i)] if '{0}layer{1}bias'.format(prefix, i) in params.keys() else None
            if output.dim() == 2 and weight.dim() == 2:
                if bias is not None:
                    output = torch.addmm(bias, output, weight)
                else:
                    output = torch.mm(output, weight)
            else:
                if output.dim() == 2 and weight.dim() == 3:
                    output = output.repeat(weight.shape[0], 1, 1)
                if output.dim() == 3 and weight.dim() == 3:
                    if bias is not None:
                        output = torch.baddbmm(bias.unsqueeze(dim=1), output, weight)
                    else:
                        output = torch.bmm(output, weight)
                else:
                    raise RuntimeError(f"Wrong dims in named forward pass {output.dim()} {weight.dim()}")
            if i < num_layers:
                output = nonlinearity(output)

        return output

    def synchronize(self, comm: mpi4py.MPI.Intracomm) -> None:
        """
        Broadcast parameter values to all threads in the comm
        :param comm: comm to broadcast to
        """
        for p in self.parameters():
            if comm.Get_rank() == 0:
                global_theta = p.data.view(-1).numpy()
                comm.Bcast([global_theta, MPI.FLOAT], 0)
            else:
                global_theta = np.empty_like(p.data.view(-1))
                comm.Bcast([global_theta, MPI.FLOAT], 0)
                p.data = torch.from_numpy(global_theta).reshape(p.data.shape)

    @staticmethod
    def _reset_parameters(weight, bias):
        """
        Function for weight initialization (similar to pytorch)
        """
        weight_dim = weight.dim()
        if weight_dim == 3:
            for d in range(weight.shape[0]):
                torch.nn.init.xavier_uniform_(tensor=weight[d, :, :].transpose(0, 1))
        elif weight_dim == 2:
            torch.nn.init.xavier_uniform_(tensor=weight.transpose(0, 1))
        else:
            raise NotImplementedError("Too many dimensions (>3) for parameter reset")
        if bias is not None:
            torch.nn.init.zeros_(bias)


class OptionsPolicy(NamedParamsNNPolicy, ABC):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 options: int,
                 hidden_sizes_base: NNLayerSizes,
                 hidden_sizes_option: NNLayerSizes,
                 hidden_sizes_subpolicy: NNLayerSizes,
                 nonlinearity: Union[torch.relu, torch.tanh],
                 no_bias: bool,
                 action_type: str,
                 std_value: float,
                 std_type: str,
                 learn_lr_inner: bool,
                 lr_inner: float,
                 temp_options: float,
                 adapt_options: bool
    ) -> None:
        NamedParamsNNPolicy.__init__(self, obs_dim=obs_dim, act_dim=action_dim, nonlinearity=nonlinearity,
                                     bias=not no_bias)  # dont use bias to have tab policy
        assert action_type in ("discrete", "continuous")
        assert std_value > 0, "Initial std has to be positive!"
        self.temp_opt = temp_options
        self.temp_act = 1
        self._std_value = std_value
        self._std_type = std_type

        self.learn_lr_inner = learn_lr_inner
        self.initial_lr_inner = lr_inner
        self.options = options

        self.hid_sizes_base = hidden_sizes_base
        self.hid_sizes_opt = hidden_sizes_option
        self.hid_sizes_subpolicy = hidden_sizes_subpolicy

        self.action_type = action_type
        if action_type == "discrete":
            self.subpolicy_head = CategoricalHead(self._act_dim, self.temp_act)
        else:
            self.subpolicy_head = DiagMVNHead(self._act_dim)
        self._adapt_options = adapt_options

        self._init_nns()
        if self.learn_lr_inner:
            self.lr_params = ParamDict(OrderedDict())
            self._init_lr_params(lr=self.initial_lr_inner, source_param_dict=self.option_params,
                                 param_dicts=(self.lr_params, self.outer_params))
            if self._adapt_options:
                self._init_lr_params(
                    lr=self.initial_lr_inner, source_param_dict=self.subpolicy_params,
                    param_dicts=(self.lr_params, self.outer_params)
                )
                try:
                    self._init_lr_params(
                        lr=self.initial_lr_inner, source_param_dict=self.termination_params,
                        param_dicts=(self.lr_params, self.outer_params)
                    )
                except AttributeError:
                    print("No self.termination_params available (probably using time based terminations?)")

    def reset_context(self) -> PolicyContext:
        return PolicyContext({"option": None, "termination": None})

    def get_action(self,
                   obs: torch.Tensor,
                   context: PolicyContext,
                   params: ParamDict = None) -> tuple[torch.Tensor, PolicyContext]:
        # If no params are specified use default model params
        if params is None:
            params = OrderedDict(self.named_parameters())

        base = self._forward_base(obs=obs, params=params)
        new_context = self._update_context(base=base, context=context, params=params)
        action = self._sample_action(inp=base, active_option=new_context["option"], params=params)
        return action, new_context

    def _init_base_nn(self) -> None:
        self.base_params = ParamDict(OrderedDict())  # Parameters of shared layers
        if len(self.hid_sizes_base) > 0:
            base_layer_sizes = NNLayerSizes((self._obs_dim,) + self.hid_sizes_base)
            self.create_named_layers(layer_sizes=base_layer_sizes, param_dicts=(self.outer_params, self.base_params),
                                     prefix="base")

    def _init_opt_nn(self) -> None:
        self.option_params = ParamDict(OrderedDict())  # Parameters of option layers
        if len(self.hid_sizes_base) > 0:
            opt_layer_sizes = NNLayerSizes((self.hid_sizes_base[-1],) + self.hid_sizes_opt + (self.options,))
        else:
            opt_layer_sizes = NNLayerSizes((self._obs_dim,) + self.hid_sizes_opt + (self.options,))
        self.create_named_layers(layer_sizes=opt_layer_sizes, param_dicts=(self.inner_params, self.option_params),
                                 prefix="options")

    def _init_subpolicy_nn(self) -> None:
        self.subpolicy_params = ParamDict(OrderedDict())  # Parameters of subpolicy layers
        if len(self.hid_sizes_base) > 0:
            policy_layer_sizes = NNLayerSizes((self.hid_sizes_base[-1],) + self.hid_sizes_subpolicy +
                                              (self._action_dim,))
        else:
            policy_layer_sizes = NNLayerSizes((self._obs_dim,) + self.hid_sizes_subpolicy + (self._act_dim,))
        if self._adapt_options:
            param_dicts = (self.outer_params, self.inner_params, self.subpolicy_params)
        else:
            param_dicts = (self.outer_params, self.subpolicy_params)
        self.create_named_layers(layer_sizes=policy_layer_sizes, extra_dims=(self.options,),
                                 param_dicts=param_dicts, prefix="subpolicy")

        if self.action_type != "discrete":
            if self._std_type == "diagonal":
                log_std_params = nn.Parameter(torch.log(torch.ones(self.options, self._act_dim) * self._std_value))
            elif self._std_type == "single":
                log_std_params = nn.Parameter(torch.log(torch.ones(self.options, 1) * self._std_value))
            else:
                raise NotImplementedError
            log_std_name = "subpolicylogstd"
            self.register_parameter(name=log_std_name, param=log_std_params)
            param_dicts = (self.outer_params, self.subpolicy_params, self.all_params)
            for pd in param_dicts:
                pd[log_std_name] = log_std_params

    def _init_lr_params(self, lr: float, source_param_dict: ParamDict, param_dicts: tuple[ParamDict, ...]) -> None:
        for key in source_param_dict.keys():
            name = f"{key}lr"
            lr_param = nn.Parameter(inv_softplus(torch.tensor(lr)))
            self.register_parameter(name=name, param=lr_param)
            for param_dict in param_dicts:
                param_dict[name] = lr_param

    def _update_context(self, base: torch.Tensor, context: PolicyContext, params: ParamDict) -> PolicyContext:
        if context["option"] is None or self._sample_termination(base=base, context=context, params=params):
            termination = int(context["option"] is not None)
            new_option = self._sample_option(base, params)
        else:
            termination = 0
            new_option = context["option"]
        return PolicyContext({"option": new_option, "termination": termination})

    def _forward_base(self, obs: torch.Tensor, params: ParamDict) -> torch.Tensor:
        # If there are shared layers use them
        if len(self.hid_sizes_base) == 0:
            return obs
        else:
            return self.named_forward_pass(obs, params, len(self.hid_sizes_base), self._nonlinearity, prefix="base")

    def _forward_option(self, base: torch.Tensor, params: ParamDict) -> torch.Tensor:
        # Get option selection probs (observations x options)
        option_params = self.named_forward_pass(base, params, len(self.hid_sizes_opt) + 1,
                                                self._nonlinearity, prefix="options")
        return F.log_softmax(option_params / self.temp_opt, dim=1)

    def _forward_subpolicy(self, base: torch.Tensor, params: ParamDict) -> torch.Tensor:
        # Get subpolicy distribution params for all options (observations x options x dist_params)
        subpolicy_params = self.named_forward_pass(base, params, len(self.hid_sizes_subpolicy) + 1,
                                                   self._nonlinearity, prefix="subpolicy").transpose(0, 1)

        if self.action_type == "discrete":
            return subpolicy_params
        else:
            return torch.cat([subpolicy_params, params["subpolicylogstd"].expand_as(subpolicy_params)], dim=-1)

    def _sample_option(self, base: torch.Tensor, params: ParamDict) -> int:
        opt_log_probs = self._forward_option(base=base, params=params)
        option = torch.multinomial(opt_log_probs.exp(), num_samples=1).item()
        return option

    @abstractmethod
    def _init_nns(self):
        raise NotImplementedError

    @abstractmethod
    def _sample_termination(self, base: torch.Tensor, context: PolicyContext, params: ParamDict) -> int:
        raise NotImplementedError

    def _sample_action(self, inp: torch.Tensor, active_option: int, params: ParamDict) -> torch.Tensor:
        subpolicy_params = self._forward_subpolicy(base=inp, params=params)[:, active_option, :]
        return self.subpolicy_head.sample_action(policy_params=subpolicy_params)


class SingleOptPolicy(OptionsPolicy):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_sizes: NNLayerSizes,
                 nonlinearity: Union[torch.relu, torch.tanh],
                 no_bias: bool,
                 adapt_options: bool,
                 action_type: str = "continuous",
                 std_type: str = "single",
                 std_value: float = 1,
                 **kwargs) -> None:
        OptionsPolicy.__init__(
            self,
            obs_dim=obs_dim,
            action_dim=action_dim,
            options=1,
            hidden_sizes_base=NNLayerSizes(tuple()),
            hidden_sizes_option=NNLayerSizes(tuple()),
            hidden_sizes_subpolicy=hidden_sizes,
            nonlinearity=nonlinearity,
            no_bias=no_bias,
            action_type=action_type,
            std_value=std_value,
            std_type=std_type,
            learn_lr_inner=False,
            lr_inner=1,
            temp_options=1,
            adapt_options=adapt_options
        )

    def forward(self, obs: torch.Tensor, params: ParamDict) -> torch.Tensor:
        return self._forward_subpolicy(base=obs, params=params)[:, 0, :]

    def get_distributions(self, obs: torch.Tensor, params: ParamDict) -> torch.distributions.Distribution:
        policy_params = self.forward(obs=obs, params=params)
        return self.subpolicy_head.get_distributions(policy_params=policy_params)

    def get_log_probs(self, obs: torch.Tensor, acts: torch.Tensor, params: ParamDict) -> torch.Tensor:
        policy_params = self.forward(obs=obs, params=params)
        return self.subpolicy_head.get_log_probs(policy_params=policy_params, actions=acts)

    def get_action(self, obs: torch.Tensor, context: PolicyContext,
                   params: ParamDict = None) -> tuple[torch.Tensor, PolicyContext]:
        if params is None:
            params = OrderedDict(self.named_parameters())
        action = self._sample_action(inp=obs, active_option=0, params=params)
        return action, PolicyContext({})

    def get_kls(self, obs: torch.Tensor, old_params: ParamDict, new_params: ParamDict) -> torch.Tensor:
        with torch.no_grad():
            old = self.get_distributions(obs=obs, params=old_params)
        new = self.get_distributions(obs=obs, params=new_params)
        return torch.distributions.kl.kl_divergence(p=old, q=new)

    def reset_context(self) -> PolicyContext:
        return PolicyContext({})

    def _init_nns(self):
        self._init_subpolicy_nn()

    def _single_traj_update_data(self, obs: torch.Tensor, actions: torch.Tensor,
                                 params: ParamDict = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = self.get_log_probs(obs=obs, acts=actions, params=params)
        return log_probs, torch.zeros_like(log_probs), torch.zeros_like(log_probs)

    def _sample_termination(self, base: torch.Tensor, context: PolicyContext, params: ParamDict) -> int:
        raise NotImplementedError

    def _sample_option(self, base: torch.Tensor, params: ParamDict) -> int:
        raise NotImplementedError


class LearnTermOptionsPolicy(OptionsPolicy):

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 options: int,
                 hidden_sizes_base: NNLayerSizes,
                 hidden_sizes_option: NNLayerSizes,
                 hidden_sizes_termination: NNLayerSizes,
                 hidden_sizes_subpolicy: NNLayerSizes,
                 nonlinearity: Union[torch.relu, torch.tanh],
                 no_bias: bool,
                 action_type: str,
                 std_value: float,
                 learn_lr_inner: bool,
                 lr_inner: float,
                 temp_options: float,
                 temp_terminations: float,
                 termination_prior: float,
                 std_type: str,
                 adapt_options: bool,
                 **kwargs) -> None:
        self.temperature_terminations = temp_terminations
        self.hid_sizes_term = hidden_sizes_termination
        self._termination_prior = termination_prior
        OptionsPolicy.__init__(
            self,
            obs_dim=obs_dim,
            action_dim=action_dim,
            options=options,
            hidden_sizes_base=hidden_sizes_base,
            hidden_sizes_option=hidden_sizes_option,
            hidden_sizes_subpolicy=hidden_sizes_subpolicy,
            nonlinearity=nonlinearity,
            no_bias=no_bias,
            action_type=action_type,
            std_value=std_value,
            std_type=std_type,
            learn_lr_inner=learn_lr_inner,
            lr_inner=lr_inner,
            temp_options=temp_options,
            adapt_options=adapt_options
        )

    def forward(self, obs: torch.Tensor, params: ParamDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if params is None:
            params = OrderedDict(self.named_parameters())
        base = self._forward_base(obs=obs, params=params)
        term_params = self._forward_termination(base=base, params=params)
        opt_params = self._forward_option(base=base, params=params)
        subpolicy_params = self._forward_subpolicy(base=base, params=params)
        return opt_params, subpolicy_params, term_params

    def _init_nns(self):
        self._init_base_nn()
        self._init_opt_nn()
        self._init_subpolicy_nn()
        self._init_term()

    def _init_term(self) -> None:
        self.termination_params = ParamDict(OrderedDict())  # Parameters of termination layers
        if len(self.hid_sizes_base) > 0:
            term_layer_sizes = NNLayerSizes((self.hid_sizes_base[-1],) + self.hid_sizes_term + (1,))
        else:
            term_layer_sizes = NNLayerSizes((self._obs_dim,) + self.hid_sizes_term + (1,))
        if self._adapt_options:
            param_dicts = (self.outer_params, self.inner_params, self.termination_params)
        else:
            param_dicts = (self.outer_params, self.termination_params)
        self.create_named_layers(layer_sizes=term_layer_sizes, extra_dims=(self.options,),
                                 param_dicts=param_dicts, prefix="termination")
        if self._termination_prior > 0:
            self._set_terminations(self._termination_prior)


    def _set_terminations(self, termination_prob: float) -> None:
        weight_value = torch.log(torch.tensor([termination_prob / (1 - termination_prob)]))
        termination_weights = torch.ones_like(self.terminationlayer1weight.data) * weight_value

        self.terminationlayer1weight.data = termination_weights

    def _single_traj_update_data(self, obs: torch.Tensor, actions: torch.Tensor, params: ParamDict = None):
        # Get relevant observations, selected actions and create placeholder for action probabilities
        if self.action_type == "discrete":
            actions = actions.long()

        opt_log_probs, policy_params, term_params  = self.forward(obs, params)
        opt_probs = torch.exp(opt_log_probs)

        # Calculate transition probs from from i to j, sum to 1 over dim=2 (observation x options x options)
        transition_probs = torch.diag_embed(1 - term_params[:, :]) + torch.matmul(opt_probs[:, :, None],
                                                                                  term_params[:, None, :])

        # Select the actions probabilities according to actions which were performed (observations x options)

        select_act_log_probs = self.subpolicy_head.get_log_probs(policy_params=policy_params, actions=actions)
        selected_action_probs = torch.exp(select_act_log_probs)
        m_s = []
        for i in range(actions.shape[0]):
            # If the new episode starts we set m (probability of being in an option) to policy over option probabilities
            if i == 0:
                m_s.append(opt_probs[i])
            # Otherwise perform an update according to IOPG
            else:
                m_s.append(torch.matmul(transition_probs[i], c_vec / torch.sum(c_vec)))
            # Calculate c_vec which is probability of being in options * probability of performing an action in option
            c_vec = m_s[-1] * selected_action_probs[i]
            c_vec = c_vec * (1 / torch.max(c_vec).detach())

        # Calculate action probabilities with one operation
        m_s = torch.stack(m_s, dim=0)
        marg_select_act_log_probs = weighted_logsumexp(input=select_act_log_probs, dim=1, weights=m_s, keepdim=True)
        opt_entropies = -torch.sum(probs_to_logits(opt_probs) * opt_probs, dim=-1, keepdim=True)
        entropies = opt_entropies
        return marg_select_act_log_probs, m_s, entropies

    def _forward_termination(self, base: torch.Tensor, params: ParamDict) -> torch.Tensor:
        # Get termination probs (observations x options)
        term_params = self.named_forward_pass(base, params, len(self.hid_sizes_term) + 1, self._nonlinearity,
                                              prefix="termination").squeeze(dim=2).transpose(0, 1)
        return torch.sigmoid(term_params / self.temperature_terminations)

    def _sample_termination(self, base: torch.Tensor, context: PolicyContext, params: ParamDict) -> int:
        active_option = context["option"]
        term_probs = self._forward_termination(base=base, params=params)  # probabilities for all options
        return torch.bernoulli(term_probs[:, active_option]).item()


class TimeTermOptionsPolicy(OptionsPolicy):

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 options: int,
                 hidden_sizes_base: NNLayerSizes,
                 hidden_sizes_option: NNLayerSizes,
                 hidden_sizes_subpolicy: NNLayerSizes,
                 nonlinearity: Union[torch.relu, torch.tanh],
                 no_bias: bool,
                 action_type: str,
                 std_value: float,
                 learn_lr_inner: bool,
                 lr_inner: float,
                 temp_options: float,
                 term_time: int,
                 std_type: str,
                 adapt_options: bool,
                 **kwargs) -> None:
        OptionsPolicy.__init__(
            self,
            obs_dim=obs_dim,
            action_dim=action_dim,
            options=options,
            hidden_sizes_base=hidden_sizes_base,
            hidden_sizes_option=hidden_sizes_option,
            hidden_sizes_subpolicy=hidden_sizes_subpolicy,
            nonlinearity=nonlinearity,
            no_bias=no_bias,
            action_type=action_type,
            std_value=std_value,
            std_type=std_type,
            learn_lr_inner=learn_lr_inner,
            lr_inner=lr_inner,
            temp_options=temp_options,
            adapt_options=adapt_options
        )
        self.temp_time = term_time

    def forward(self, obs: torch.Tensor, params: ParamDict) -> tuple[torch.Tensor, torch.Tensor]:
        if params is None:
            params = OrderedDict(self.named_parameters())
        base = self._forward_base(obs=obs, params=params)
        opt_params = self._forward_option(base=base, params=params)
        subpolicy_params = self._forward_subpolicy(base=base, params=params)
        return opt_params, subpolicy_params

    def _single_traj_update_data(self, obs: torch.Tensor, actions: torch.Tensor, params: ParamDict = None):
        # Get relevant observations, selected actions and create placeholder for action probabilities
        if self.action_type == "discrete":
            actions = actions.long()
        n_obs = obs.shape[0]
        opt_log_probs, policy_params = self.forward(obs, params)

        # Select the actions probabilities according to actions which were performed (observations x options)
        select_act_log_probs = self.subpolicy_head.get_log_probs(policy_params=policy_params, actions=actions)

        # Reshape logprobs to N/t, t, opts to allow for quick calculation
        padding_to_add = (self.temp_time - n_obs % self.temp_time) % self.temp_time
        if padding_to_add > 0:
            select_act_log_probs = torch.cat((select_act_log_probs, torch.ones(padding_to_add, self.options)), dim=0)
        reshaped_log_probs = select_act_log_probs.view(-1, self.temp_time, self.options)

        # Remove last timestep and add logprobs of zeroes at the start
        reshaped_log_probs = torch.cat((torch.ones(reshaped_log_probs.shape[0], 1, self.options),
                                        reshaped_log_probs[:, :self.temp_time-1, :]), dim=1)

        # Get cumulative sum
        cumsum_log_probs = torch.cumsum(reshaped_log_probs, dim=1)
        unnorm_log_m_s = cumsum_log_probs + opt_log_probs[::self.temp_time, :][:, None, :]
        log_m_s = unnorm_log_m_s - torch.logsumexp(unnorm_log_m_s, dim=2, keepdim=True)
        m_s = torch.exp(log_m_s).view(-1, self.options)

        # Calculate action probabilities with one operation
        marg_select_act_log_probs = weighted_logsumexp(input=select_act_log_probs, dim=1, weights=m_s, keepdim=True)
        if padding_to_add > 0:
            marg_select_act_log_probs = marg_select_act_log_probs[:n_obs]
        opt_entropies = -torch.sum(opt_log_probs * opt_log_probs.exp(), dim=-1, keepdim=True)
        entropies = opt_entropies

        return marg_select_act_log_probs, m_s, entropies

    def _sample_termination(self, base: torch.Tensor, context: PolicyContext, params: ParamDict) -> int:
        timestep = context["timestep"]
        return int(timestep > 0 and (timestep % self.temp_time == 0))

    def _init_nns(self):
        self._init_base_nn()
        self._init_opt_nn()
        self._init_subpolicy_nn()


def net_test():
    state_dim = 8
    options = 4
    batch_size = 200
    prefix = ""
    states = torch.rand(batch_size, state_dim)
    use_bias = True
    net = nn.Sequential(
            nn.Linear(state_dim, 64, bias=use_bias),
            nn.ReLU(),
            nn.Linear(64, 64, bias=use_bias),
            nn.ReLU(),
            nn.Linear(64, options, bias=use_bias)
    )

    outputs1 = net.forward(states)

    policy = NamedParamsNNPolicy(bias=use_bias)

    module_indices = [0, 2, 4]
    for index in range(len(module_indices)):
        weight = net._modules[f'{module_indices[index]}'].weight.data.clone().transpose(0, 1)
        weight_name = '{0}layer{1}weight'.format(prefix, index+1)
        weight_params = nn.Parameter(weight)
        policy.register_parameter(name=weight_name, param=weight_params)
        if use_bias:
            bias = net._modules[f'{module_indices[index]}'].bias.data.clone()
            bias_name = '{0}layer{1}bias'.format(prefix, index+1)
            bias_params = nn.Parameter(bias)
            policy.register_parameter(name=bias_name, param=bias_params)
    outputs2 = policy.named_forward_pass(states, params=OrderedDict(policy.named_parameters()), num_layers=3,
                                         nonlinearity=torch.relu, prefix=prefix)
    assert torch.all(outputs1 == outputs2), torch.max(torch.abs(outputs1-outputs2))


def multinet_test():
    state_dim = 8
    action_dim = 5
    options = 4
    batch_size = 200
    prefix = ""
    states = torch.rand(batch_size, state_dim)
    use_bias = True
    nets = [
        nn.Sequential(
            nn.Linear(state_dim, 64, bias=use_bias),
            nn.ReLU(),
            nn.Linear(64, 64, bias=use_bias),
            nn.ReLU(),
            nn.Linear(64, action_dim, bias=use_bias)
        ) for _ in range(options)
    ]
    outputs1 = torch.cat([nets[i].forward(states)[None, :, :] for i in range(options)], dim=0)

    policy = NamedParamsNNPolicy(bias=use_bias)

    module_indices = [0, 2, 4]
    for index in range(len(module_indices)):
        weight = torch.cat([nets[i]._modules[f'{module_indices[index]}'].weight.data.clone()[None, :, :]
                            for i in range(options)], dim=0).transpose(1, 2)
        weight_name = '{0}layer{1}weight'.format(prefix, index+1)
        weight_params = nn.Parameter(weight)
        policy.register_parameter(name=weight_name, param=weight_params)
        if use_bias:
            bias = torch.cat([nets[i]._modules[f'{module_indices[index]}'].bias.data.clone()[None, :]
                              for i in range(options)], dim=0)
            bias_name = '{0}layer{1}bias'.format(prefix, index+1)
            bias_params = nn.Parameter(bias)
            policy.register_parameter(name=bias_name, param=bias_params)
    outputs2 = policy.named_forward_pass(states, params=OrderedDict(policy.named_parameters()), num_layers=3,
                                         nonlinearity=torch.relu, prefix=prefix)
    assert torch.all(outputs1 == outputs2), torch.max(torch.abs(outputs1-outputs2))


if __name__ == '__main__':
    multinet_test()
    net_test()
