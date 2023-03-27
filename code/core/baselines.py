import torch
import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import Union


class Baseline(ABC):
    @abstractmethod
    def predict(self, obs: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def update(self, obs: torch.Tensor, timesteps: torch.Tensor, disc_rets: torch.Tensor) -> None:
        raise NotImplementedError


class TimeIndependentBaseline(Baseline):
    def predict(self, obs: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self._predict(obs=obs)

    @abstractmethod
    def _predict(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def update(self, obs: torch.Tensor, timesteps: torch.Tensor, disc_rets: torch.Tensor) -> None:
        self._update(obs=obs, disc_rets=disc_rets)

    @abstractmethod
    def _update(self, obs: torch.Tensor, disc_rets: torch.Tensor) -> None:
        raise NotImplementedError


class LinearFeatureBaseline(Baseline):
    """
    Linear (polynomial) time-state dependent return baseline model
    (see. Duan et al. 2016, "Benchmarking Deep Reinforcement Learning for Continuous Control", ICML)
    https://github.com/rll/rllab/
    """

    def __init__(self, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def predict(self, obs: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Predicts the linear reward baselines estimates for a provided trajectory / path.
        If the baseline is not fitted - returns zero baseline
        """

        if self._coeffs is None:
            baseline = torch.zeros(obs.shape[0], 1)
        else:
            baseline = torch.tensor(self._features(obs, timesteps).dot(self._coeffs), dtype=torch.get_default_dtype())
        return baseline

    def update(self, obs: torch.Tensor, timesteps: torch.Tensor, disc_rets: torch.Tensor) -> None:
        """
        Fits the linear baseline model with the provided paths via damped least squares
        """

        featmat = self._features(obs, timesteps)
        target = disc_rets.numpy()
        reg_coeff = self._reg_coeff
        for i in range(10):
            try:
                self._coeffs = np.linalg.lstsq(
                    featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                    featmat.T.dot(target),
                    rcond=-1
                )[0]
                if not np.any(np.isnan(self._coeffs)):
                    break
            except Exception as exc:
                if i < 9:
                    warnings.warn(f'LSTSQ did not converge with reg coef {self._reg_coeff}', RuntimeWarning)
                    reg_coeff *= 10
                else:
                    raise exc

    @staticmethod
    def _features(obs, timesteps):
        obs = np.clip(obs, -10, 10)
        path_length = len(obs)
        time_step = timesteps.numpy() / 100.0
        return np.concatenate([obs, obs ** 2, time_step, time_step ** 2, time_step ** 3, np.ones((path_length, 1))],
                              axis=1)

    def set_param_values(self, value):
        self._coeffs = value

    def get_param_values(self):
        return self._coeffs


class ZeroBaseline(TimeIndependentBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def _predict(obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(obs.shape[0], 1)

    def _update(self, obs: torch.Tensor, disc_rets: torch.Tensor) -> None:
        pass

