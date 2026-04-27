import torch
import torch.optim as optim
import numpy as np
# from models import utils as mutils

import math
import torch.nn.functional as F
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from torch.nn.modules.loss import _Loss
from typing import Union


from torch import Tensor


@dataclass
class SchedulerOutput:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        alpha_t (Tensor): :math:`\alpha_t`, shape (...).
        sigma_t (Tensor): :math:`\sigma_t`, shape (...).
        d_alpha_t (Tensor): :math:`\frac{\partial}{\partial t}\alpha_t`, shape (...).
        d_sigma_t (Tensor): :math:`\frac{\partial}{\partial t}\sigma_t`, shape (...).

    """

    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})

@dataclass
class PathSample:

    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        x_1 (Tensor): the target sample :math:`X_1`.
        x_0 (Tensor): the source sample :math:`X_0`.
        t (Tensor): the time sample :math:`t`.
        x_t (Tensor): samples :math:`X_t \sim p_t(X_t)`, shape (batch_size, ...).
        dx_t (Tensor): conditional target :math:`\frac{\partial X}{\partial t}`, shape: (batch_size, ...).

    """

    x_1: Tensor = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: Tensor = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: Tensor = field(metadata={"help": "time samples t (batch_size, ...)."})
    x_t: Tensor = field(
        metadata={"help": "samples x_t ~ p_t(X_t), shape (batch_size, ...)."}
    )
    dx_t: Tensor = field(
        metadata={"help": "conditional target dX_t, shape: (batch_size, ...)."}
    )

class ProbPath(ABC):
    r"""Abstract class, representing a probability path.

    A probability path transforms the distribution :math:`p(X_0)` into :math:`p(X_1)` over :math:`t=0\rightarrow 1`.

    The ``ProbPath`` class is designed to support model training in the flow matching framework. It supports two key functionalities: (1) sampling the conditional probability path and (2) conversion between various training objectives.
    Here is a high-level example

    .. code-block:: python

        # Instantiate a probability path
        my_path = ProbPath(...)

        for x_0, x_1 in dataset:
            # Sets t to a random value in [0,1]
            t = torch.rand()

            # Samples the conditional path X_t ~ p_t(X_t|X_0,X_1)
            path_sample = my_path.sample(x_0=x_0, x_1=x_1, t=t)

            # Optimizes the model. The loss function varies, depending on model and path.
            loss(path_sample, my_model(x_t, t)).backward()

    """

    @abstractmethod
    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        r"""Sample from an abstract probability path:

        | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)`.
        | returns :math:`X_0, X_1, X_t \sim p_t(X_t)`, and a conditional target :math:`Y`, all objects are under ``PathSample``.

        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            PathSample: a conditional sample.
        """

    def assert_sample_shape(self, x_0: Tensor, x_1: Tensor, t: Tensor):
        assert (
            t.ndim == 1
        ), f"The time vector t must have shape [batch_size]. Got {t.shape}."
        assert (
            t.shape[0] == x_0.shape[0] == x_1.shape[0]
        ), f"Time t dimension must match the batch size [{x_1.shape[0]}]. Got {t.shape}"



@dataclass
class DiscretePathSample:
    """
    Represents a sample of a conditional-flow generated discrete probability path.

    Attributes:
        x_1 (Tensor): the target sample :math:`X_1`.
        x_0 (Tensor): the source sample :math:`X_0`.
        t (Tensor): the time sample  :math:`t`.
        x_t (Tensor): the sample along the path  :math:`X_t \sim p_t`.
    """

    x_1: Tensor = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: Tensor = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: Tensor = field(metadata={"help": "time samples t (batch_size, ...)."})
    x_t: Tensor = field(
        metadata={"help": "samples X_t ~ p_t(X_t), shape (batch_size, ...)."}
    )

class Scheduler(ABC):
    """Base Scheduler class."""

    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        r"""
        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...

class ConvexScheduler(Scheduler):
    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        """Scheduler for convex paths.

        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        """
        Computes :math:`t` from :math:`\kappa_t`.

        Args:
            kappa (Tensor): :math:`\kappa`, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...

    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        kappa_t = snr / (1.0 + snr)

        return self.kappa_inverse(kappa=kappa_t)



class PolynomialConvexScheduler(ConvexScheduler):
    """Polynomial Scheduler."""

    def __init__(self, n: Union[float, int]) -> None:
        assert isinstance(
            n, (float, int)
        ), f"`n` must be a float or int. Got type(n) = {type(n)}."
        assert n > 0, f"`n` must be positive. Got n = {n}."

        self.n = n

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t**self.n,
            sigma_t=1 - t**self.n,
            d_alpha_t=self.n * (t ** (self.n - 1)),
            d_sigma_t=-self.n * (t ** (self.n - 1)),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return torch.pow(kappa, 1.0 / self.n)

class MixtureDiscreteProbPath(ProbPath):
    r"""The ``MixtureDiscreteProbPath`` class defines a factorized discrete probability path.

    This path remains constant at the source data point :math:`X_0` until a random time, determined by the scheduler, when it flips to the target data point :math:`X_1`.
    The scheduler determines the flip probability using the parameter :math:`\sigma_t`, which is a function of time `t`. Specifically, :math:`\sigma_t` represents the probability of remaining at :math:`X_0`, while :math:`1 - \sigma_t` is the probability of flipping to :math:`X_1`:

    .. math::

        P(X_t = X_0) = \sigma_t \quad \text{and} \quad  P(X_t = X_1) = 1 - \sigma_t,

    where :math:`\sigma_t` is provided by the scheduler.

    Example:

    .. code-block:: python
        #
        # >>> x_0 = torch.zeros((1, 3, 3))
        # >>> x_1 = torch.ones((1, 3, 3))
        #
        # >>> path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))
        # >>> result = path.sample(x_0, x_1, t=torch.tensor([0.1])).x_t
        # >>> result
        # tensor([[[0.0, 0.0, 0.0],
        #          [0.0, 0.0, 1.0],
        #          [0.0, 0.0, 0.0]]])
        #


    Args:
        scheduler (ConvexScheduler): The scheduler that provides :math:`\sigma_t`.
    """

    def __init__(self, scheduler: ConvexScheduler):
        assert isinstance(
            scheduler, ConvexScheduler
        ), "Scheduler for ConvexProbPath must be a ConvexScheduler."

        self.scheduler = scheduler

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> DiscretePathSample:
        r"""Sample from the affine probability path:
            | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)` and a scheduler :math:`(\alpha_t,\sigma_t)`.
            | return :math:`X_0, X_1, t`, and :math:`X_t \sim p_t`.
        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            DiscretePathSample: a conditional sample at :math:`X_t ~ p_t`.
        """
        # print(x_0.shape,x_1.shape)
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        sigma_t = self.scheduler(t).sigma_t

        sigma_t = expand_tensor_like(input_tensor=sigma_t, expand_to=x_1)

        source_indices = torch.rand(size=x_1.shape, device=x_1.device) < sigma_t
        x_0 = x_0.long().to(source_indices.device)
        x_1 = x_1.long().to(source_indices.device)
        x_t = torch.where(condition=source_indices, input=x_0, other=x_1)

        # # 🔴 [Added #1]: keep the upper triangle of x_t (drop lower triangle to avoid asymmetry)
        # x_t = torch.triu(x_t, diagonal=1)
        #
        # # 🔴 [Added #2]: symmetrize by adding the upper triangle and its transpose
        # x_t = x_t + x_t.transpose(-1, -2)

        return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t)

    def posterior_to_velocity(
        self, posterior_logits: Tensor, x_t: Tensor, t: Tensor
    ) -> Tensor:
        r"""Convert the factorized posterior to velocity.

        | given :math:`p(X_1|X_t)`. In the factorized case: :math:`\prod_i p(X_1^i | X_t)`.
        | return :math:`u_t`.

        Args:
            posterior_logits (Tensor): logits of the x_1 posterior conditional on x_t, shape (..., vocab size).
            x_t (Tensor): path sample at time t, shape (...).
            t (Tensor): time in [0,1].

        Returns:
            Tensor: velocity.
        """
        posterior = torch.softmax(posterior_logits, dim=-1)
        vocabulary_size = posterior.shape[-1]
        x_t = F.one_hot(x_t, num_classes=vocabulary_size)
        t = unsqueeze_to_match(source=t, target=x_t)

        scheduler_output = self.scheduler(t)

        kappa_t = scheduler_output.alpha_t
        d_kappa_t = scheduler_output.d_alpha_t

        return (d_kappa_t / (1 - kappa_t)) * (posterior - x_t)

    def prob(self, x0, t, x1):
        # Compute p_t(x) = σ_t·δ(x,x0) + (1-σ_t)·δ(x,x1) elementwise on the matrix
        # s = self.scheduler(t)['sigma'].view(-1,1,1)
        s = self.scheduler(t).sigma_t.view(-1, 1, 1)
        return s * (x0==1).float() + (1-s) * (x1==1).float()

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
        # print("log_p_1t", log_p_1t.shape)
        log_p_1t_x1 = torch.gather(log_p_1t, dim=1, index=x_1.unsqueeze(1).long())
        log_p_1t_x1 = log_p_1t_x1.view(*x_1_shape)

        # extract x_t value of p_{1|t}(x|x_t).
        p_1t = torch.exp(log_p_1t)
        p_1t_xt = torch.gather(p_1t, dim=1, index=x_t.unsqueeze(1).long())
        p_1t_xt = p_1t_xt.view(*x_1_shape)

        scheduler_output = self.path.scheduler(t)

        jump_coefficient = (
            scheduler_output.d_alpha_t / (1 - scheduler_output.alpha_t)
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




def unsqueeze_to_match(source: Tensor, target: Tensor, how: str = "suffix") -> Tensor:
    """
    Unsqueeze the source tensor to match the dimensionality of the target tensor.

    Args:
        source (Tensor): The source tensor to be unsqueezed.
        target (Tensor): The target tensor to match the dimensionality of.
        how (str, optional): Whether to unsqueeze the source tensor at the beginning
            ("prefix") or end ("suffix"). Defaults to "suffix".

    Returns:
        Tensor: The unsqueezed source tensor.
    """
    assert (
        how == "prefix" or how == "suffix"
    ), f"{how} is not supported, only 'prefix' and 'suffix' are supported."

    dim_diff = target.dim() - source.dim()

    for _ in range(dim_diff):
        if how == "prefix":
            source = source.unsqueeze(0)
        elif how == "suffix":
            source = source.unsqueeze(-1)

    return source


def expand_tensor_like(input_tensor: Tensor, expand_to: Tensor) -> Tensor:
    """`input_tensor` is a 1d vector of length equal to the batch size of `expand_to`,
    expand `input_tensor` to have the same shape as `expand_to` along all remaining dimensions.

    Args:
        input_tensor (Tensor): (batch_size,).
        expand_to (Tensor): (batch_size, ...).

    Returns:
        Tensor: (batch_size, ...).
    """
    assert input_tensor.ndim == 1, "Input tensor must be a 1d vector."
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

    dim_diff = expand_to.ndim - input_tensor.ndim

    t_expanded = input_tensor.clone()
    t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))

    return t_expanded.expand_as(expand_to)