import torch
import torch.optim as optim
import numpy as np
# from models import utils as mutils
from utils.discrete_schedulers import MixtureDiscreteProbPath
import torch.nn.functional as F
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from torch.nn.modules.loss import _Loss
from typing import Union


from torch import Tensor


class MixturePathGeneralizedKL(_Loss):
    r"""A generalized KL loss for discrete flow matching.
    A class that measures the generalized KL of a discrete flow model :math:`p_{1|t}` w.r.t. a probability path given by ``path``. Note: this class is assuming that the model is trained on the same path.

    For a model trained on a space :math:`\mathcal{S} = \mathcal{T}^d`, :math:`\mathcal{T} = [K] = \set{1,2,\ldots,K}`, the loss is given by

    .. math::
            \ell_i(x_1, x_t, t) = -\frac{\dot{\kappa}_t}{1-\kappa_t} \biggr[  p_{1|t}(x_t^i|x_t) -\delta_{x^i_1}(x_t^i) + (1-\delta_{x^i_1}(x_t^i))\left(\log p_{1|t}(x_1^i|x_t)\right)\biggr],

    where :math:`\kappa_t` is the scheduler associated with ``path``.

    Args:
        path (MixtureDiscreteProbPath): Probability path (x-prediction training).
        reduction (str, optional): Specify the reduction to apply to the output ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction is applied to the output, ``'mean'``: the output is reduced by mean over sequence elements, ``'sum'``: the output is reduced by sum over sequence elements. Defaults to 'mean'.
    """

    def __init__(self, path: MixtureDiscreteProbPath, reduction: str = "mean") -> None:
        super().__init__(None, None, reduction)
        self.path = path

    def forward(self, logits: Tensor, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Evaluates the generalized KL loss.

        Args:
            logits (Tensor): posterior model output (i.e., softmax(``logits``) :math:`=p_{1|t}(x|x_t)`), shape (batch, d, K).
            x_1 (Tensor): target data point :math:`x_1 \sim q`, shape (batch, d).
            x_t (Tensor): conditional sample at :math:`x_t \sim p_t(\cdot|x_1)`, shape (batch, d).
            t (Tensor): times in :math:`[0,1]`, shape (batch).

        Raises:
            ValueError: reduction value must be one of ``'none'`` | ``'mean'`` | ``'sum'``.

        Returns:
            Tensor: Generalized KL loss.
        """
        # x_1 = x_1.permute((0, 2, 3, 1))
        # x_0 = x_0.permute((0, 2, 3, 1))
        x_1_shape = x_1.shape
        # print("x_1", x_1_shape)

        # extract x_1 value of log(p_{1|t}(x|x_t)).
        # print('l',logits,logits.shape)
        log_p_1t = torch.log_softmax(logits, dim=1)
        # x_1_01 = ((x_1 + 1) // 2).long()
        # x_t_01 = ((x_t + 1) // 2).long()
        # print("log_p_1t", log_p_1t.shape)
        log_p_1t_x1 = torch.gather(log_p_1t, dim=1, index=x_1.unsqueeze(1).long())
        # log_p_1t_x1 = torch.gather(log_p_1t, dim=1, index=x_1_01.unsqueeze(1).long())
        log_p_1t_x1 = log_p_1t_x1.view(*x_1_shape)

        # extract x_t value of p_{1|t}(x|x_t).
        p_1t = torch.exp(log_p_1t)
        p_1t_xt = torch.gather(p_1t, dim=1, index=x_t.unsqueeze(1).long())
        # p_1t_xt = torch.gather(p_1t, dim=1, index=x_t_01.unsqueeze(1).long())
        p_1t_xt = p_1t_xt.view(*x_1_shape)

        scheduler_output = self.path.scheduler(t)

        jump_coefficient = (
            scheduler_output.d_alpha_t / (1 - scheduler_output.alpha_t+ 1e-8)
        )[(...,) + (None,) * (x_1.dim() - 1)]
        jump_coefficient = jump_coefficient.repeat(1, *x_1_shape[1:])
        delta_x1_xt = (x_t == x_1).to(log_p_1t.dtype)

        loss = -jump_coefficient * (
            p_1t_xt - delta_x1_xt + (1 - delta_x1_xt) * log_p_1t_x1
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")


class MixturePathGeneralizedKLKO(_Loss):
    """Kinetic-optimal generalized KL loss."""

    def __init__(self, path: MixtureDiscreteProbPath, reduction: str = "mean"):
        super().__init__(None, None, reduction)
        self.path = path

    def forward(self, logits: Tensor, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        x_1_shape = x_1.shape
        log_p_1t = torch.log_softmax(logits, dim=1)
        log_p_1t_x1 = torch.gather(log_p_1t, dim=1, index=x_1.unsqueeze(1).long())
        log_p_1t_x1 = log_p_1t_x1.view(*x_1_shape)

        p_1t = torch.exp(log_p_1t)
        p_1t_xt = torch.gather(p_1t, dim=1, index=x_t.unsqueeze(1).long())
        p_1t_xt = p_1t_xt.view(*x_1_shape)

        scheduler_output = self.path.scheduler(t)
        lambda_t = (scheduler_output.d_alpha_t / (1 - scheduler_output.alpha_t + 1e-8))[(...,) + (None,) * (x_1.dim() - 1)]
        lambda_t = lambda_t.repeat(1, *x_1_shape[1:])
        delta_x1_xt = (x_t == x_1).to(logits.dtype)

        # ELBO for the kinetic-optimal objective
        loss = -lambda_t * (
                p_1t_xt - delta_x1_xt + (1 - delta_x1_xt) * log_p_1t_x1
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")