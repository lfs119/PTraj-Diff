import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
from pytorch_lightning.utilities import rank_zero_only

def inflate_batch_array(array: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Inflate the batch array (`array`) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e., shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape = (array.shape[0],) + (1,) * (len(target.shape) - 1)
    return array.view(target_shape)

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
log = get_pylogger(__name__)

def cosine_beta_schedule(
    num_timesteps: int,
    s: float = 0.008,
    raise_to_power: float = 1
) -> np.ndarray:
    """
    A cosine variance schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """
    steps = num_timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

def clip_noise_schedule(
    alphas2: np.ndarray,
    clip_value: float = 0.001
) -> np.ndarray:
    """
    For a noise schedule given by (alpha ^ 2), this clips alpha_t / (alpha_t - 1).
    This may help improve stability during sampling.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(
    num_timesteps: int,
    s: float = 1e-4,
    power: float = 3.0
) -> np.ndarray:
    """
    A noise schedule based on a simple polynomial equation: 1 - (x ^ power).
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """
    steps = num_timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


class PredefinedNoiseSchedule(nn.Module):
    """
    A predefined noise schedule. Essentially, creates a lookup array for predefined (non-learned) noise schedules.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """

    def __init__(
        self,
        noise_schedule: str,
        num_timesteps: int,
        noise_precision: float,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__()

        self.timesteps = num_timesteps

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(num_timesteps)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(num_timesteps, s=noise_precision, power=power)
        else:
            raise ValueError(noise_schedule)

        if verbose:
            log.info(f"alphas2: {alphas2}")

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        if verbose:
            log.info(f"gamma: {-log_alphas2_to_sigmas2}")

        self.gamma = nn.Parameter(
            torch.tensor(-log_alphas2_to_sigmas2).float(),
            requires_grad=False
        )


    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]