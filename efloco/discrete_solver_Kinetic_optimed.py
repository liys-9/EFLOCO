# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from math import ceil
from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch.nn import functional as F


from utils.discrete_schedulers import PolynomialConvexScheduler,MixtureDiscreteProbPath
from utils.step_optim import StepOptim



try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


from abc import ABC, abstractmethod

from torch import nn, Tensor


class Solver(ABC, nn.Module):
    """Abstract base class for solvers."""

    @abstractmethod
    def sample(self, x_0: Tensor = None) -> Tensor:
        ...




class MixtureDiscreteEulerSolver(Solver):
    r"""Solver that simulates the CTMC process :math:`(X_t)_{t_{\text{init}}\leq t\leq t_{\text{final}}}` defined by :math:`p_t` the marginal probability path of ``path``.
    Given :math:`X_t \sim p_t`, the algorithm of solver step from :math:`t` to :math:`t+h` for the i-th coordinate is:

    .. math::

        \begin{align*}
            & X_1^i \sim p_{1|t}^i(\cdot|X_t)\\
            & \lambda^i \gets \sum_{x^i\ne X_t^i} u_t^i(x^i, X_t^i|X_1^i)\\
            & Z^i_{\text{change}} \sim U[0,1]\\
            & X_{t+h}^i \sim \begin{cases}
                \frac{u_t^i(\cdot, X_t^i|X_1^i)}{\lambda^i}(1-\delta_{X_t^i}(\cdot)) \text{ if $Z^i_{\text{change}}\le 1-e^{-h\lambda^i}$}\\
                \delta_{X_t^i}(\cdot) \text{ else }
            \end{cases}
        \end{align*}

    Where :math:`p_{1|t}(\cdot|X_t)` is the output of ``model``, and the conditional probability velocity is of the mixture probability path is:

    .. math::

        u_t^i(x^i, y^i|x_1^i) = \hat{u}_t^i(x^i, y^i|x_1^i) + c_{\text{div\_free}}\left[\hat{u}_t^i(x^i, y^i|x_1^i) - \check{u}_t^i(x^i, y^i|x_1^i) \right],

    where

    .. math::
        \hat{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{1-\kappa_t} \left[ \delta_{x_1^i}(x^i) - \delta_{y^i}(x^i) \right],

    and

    .. math::

        \check{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{\kappa_t}\left[ \delta_{y^i}(x^i) - p(x^i) \right].

    The source distribution :math:`p(x^i)` is given by ``p``.

    Args:
        model (ModelWrapper): trained with x-prediction, outputting posterior probabilities (in the range :math:`[0,1]`), output must be [..., vocabulary_size].
        path (MixtureDiscreteProbPath): Probability path used for x-prediction training.
        vocabulary_size (int): size of the discrete vocabulary.
        source_distribution_p (Optional[Tensor], optional): Source distribution, must be of shape [vocabulary_size]. Required only when divergence-free term for the probability velocity is non-zero. Defaults to None.
    """

    def __init__(
        self,
        model: nn.Module,
        path: MixtureDiscreteProbPath,
        vocabulary_size: int,
        sparse: int,
        source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size
        self.sparse = sparse

        if source_distribution_p is not None:
            assert source_distribution_p.shape == torch.Size(
                [vocabulary_size]
            ), f"Source distribution p dimension must match the vocabulary size {vocabulary_size}. Got {source_distribution_p.shape}."

        self.source_distribution_p = source_distribution_p

        # self.cached_t_discretization = None

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        div_free: Union[float, Callable[[float], float]] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        eval_batch = None,
        use_step_optim: bool = False,           # Enable step optimization
        nfe: Optional[int] = None,              # Number of optimized steps
        step_optim_init_type: str = "unif",     # Initialization strategy
    ) -> Tensor:

        if not div_free == 0.0:
            assert (
                self.source_distribution_p is not None
            ), "Source distribution p must be specified in order to add a divergence-free term to the probability velocity."

        # Initialize the current state `x_t` with the initial state `X_0`.
        # time_grid = torch.tensor(time_grid).to(device=x_init.device)
        time_grid = time_grid.clone().detach()  # Make a detached copy
        t_init = float(time_grid[0].item())
        t_final = float(time_grid[-1].item())

        if use_step_optim:
            assert nfe is not None, "Must specify nfe if use_step_optim=True"
            # from step_optim import StepOptim  # Ensure this import path matches your project layout



            step_optim = StepOptim(self.path.scheduler)
            T = time_grid[0].item()
            eps = time_grid[-1].item()
            t_discretization = get_or_load_t_grid(
                scheduler=self.path.scheduler,
                nfe=nfe,
                eps=eps,
                initType=step_optim_init_type,
                device=x_init.device,
                verbose=verbose
            )
            # print(t_discretization)
            n_steps = len(t_discretization) - 1
            # if self._cached_t_discretization is None:
            # t_discretization, _ = step_optim.get_ts_lambdas(N=nfe, eps=0.0, initType='unif_t')
            # t_discretization = t_discretization.to(x_init.device)
            # n_steps = nfe

        else:
            if nfe is not None:
                # Uniform discretization with exactly `nfe` steps over [t_init, t_final].
                n_steps = int(nfe)
                t_discretization = torch.linspace(
                    t_init, t_final, n_steps + 1, device=x_init.device
                )
            else:
                # Use the provided time_grid directly.
                t_discretization = time_grid.to(x_init.device)
                n_steps = len(t_discretization) - 1

            if return_intermediates:
                # get order of intermediate steps:
                order = torch.argsort(time_grid)
                # Compute intermediate steps to return via nearest points in t_discretization to time_grid.
                time_grid = time_grid.to(x_init.device)
                time_grid = get_nearest_times(
                    time_grid=time_grid, t_discretization=t_discretization
                )

        x_t = x_init.clone()
        if not self.sparse:
            _, points, adj_matrix, _ = eval_batch
            edge_index = None
        else:
            _, graph_data, point_indicator, edge_indicator, _ = eval_batch
            route_edge_flags = graph_data.edge_attr
            points = graph_data.x
            edge_index = graph_data.edge_index
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
        points = points.to(x_init.device)
        batch_size = adj_matrix.shape[0]
        steps_counter = 0
        res = []

        if return_intermediates:
            res = [x_init.clone().to(x_init.device)]

        if verbose:
            if not TQDM_AVAILABLE:
                raise ImportError(
                    "tqdm is required for verbose mode. Please install it."
                )
            ctx = tqdm(total=t_final, desc=f"NFE: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx:
            for i in range(n_steps):
                # print(t_discretization)

                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]
                h = h.to(x_init.device)

                # Sample x_1 ~ p_1|t( \cdot |x_t)
                # p_1t = self.model(x=x_t, t=t.repeat(x_t.shape[0]), **model_extras)
                x_t = x_t.to(x_init.device)
                t = t.to(x_init.device)

                p_1t = self.model(
                    points,
                    x_t,
                    t.repeat(x_t.shape[0]),
                    edge_index,
                )
                # print('p1t',p_1t,p_1t.shape)
                p_1t = torch.softmax(p_1t, dim=1)
                # print('p1tsoft', p_1t, p_1t.shape)
                if not self.sparse:
                    p_1t_prob = p_1t.permute((0, 2, 3, 1)).contiguous()
                else:
                    p_1t_prob = p_1t.contiguous()
                # print('p_1t_prob', p_1t_prob, p_1t_prob.shape)
                x_1 = categorical(p_1t_prob.to(dtype=dtype_categorical))

                # Checks if final step
                if i == n_steps - 1:
                    # x_t = x_1
                    # x_t = p_1t[:, 1, :, :].contiguous() #for not sprase
                    x_t = p_1t[:, 1].contiguous()
                else:
                    # Compute u_t(x|x_t,x_1)
                    scheduler_output = self.path.scheduler(t=t)

                    k_t = scheduler_output.alpha_t
                    d_k_t = scheduler_output.d_alpha_t

                    # delta_1 = F.one_hot(x_1, num_classes=self.vocabulary_size).to(
                    #     k_t.dtype
                    # )
                    # u = d_k_t / (1 - k_t) * delta_1

                    delta_x1 = F.one_hot(x_1, self.vocabulary_size).to(k_t.dtype)
                    u_optimal = (d_k_t / (1 - k_t + 1e-8)).unsqueeze(-1) * delta_x1

                    # Add divergence-free part
                    div_free_t = div_free(t) if callable(div_free) else div_free

                    if div_free_t > 0:
                        p_0 = self.source_distribution_p[(None,) * x_t.dim()]
                        u_optimal = u_optimal + div_free_t * d_k_t / (k_t * (1 - k_t)) * (
                            (1 - k_t) * p_0 + k_t * delta_x1
                        )

                    # Set u_t(x_t|x_t,x_1) = 0
                    delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                    u_optimal = torch.where(
                        delta_t.to(dtype=torch.bool), torch.zeros_like(u_optimal), u_optimal
                    )

                    # Sample x_t ~ u_t( \cdot |x_t,x_1)
                    intensity = u_optimal.sum(dim=-1).to(x_init.device)  # Assuming u_t(xt|xt,x1) := 0
                    mask_jump = torch.rand(
                        size=x_t.shape, device=x_t.device
                    ) < 1 - torch.exp(-h * intensity)

                    # if mask_jump.sum() > 0:
                    #     x_t[mask_jump] = categorical(
                    #         u[mask_jump].to(dtype=dtype_categorical)
                    #     )
                    if mask_jump.any():
                        x_t[mask_jump] = categorical(u_optimal[mask_jump].to(dtype_categorical))

                steps_counter += 1
                t = t + h
                time_grid = time_grid.to(x_init.device)

                if return_intermediates and (t in time_grid):
                    res.append(x_t.clone())

                if verbose:
                    ctx.n = t.item()
                    ctx.refresh()
                    ctx.set_description(f"NFE: {steps_counter}")

        if return_intermediates:
            return torch.stack(res, dim=0)
        else:
            return x_t

    def jump_state_to(
        self,
        # points: graph structure / coordinates passed to the model (if needed)
        x_ta: torch.Tensor,  # Current state, shape e.g. [B, H, W] or [B, D]
        ta: torch.Tensor,  # Current time scalar, shape [1]
        tb: torch.Tensor,  # Target time scalar, shape [1]
        logits: torch.Tensor,  # Model output at time ta, shape [B, V, ...]
        div_free: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Jump from state x_ta at time ta to state x_tb at time tb.
        logits is the model output at time ta (before softmax).
        """
        # 1. Compute time step h
        # x_ta = x_ta.view(ta.shape[0], -1)
        # logits = logits.view(tb.shape[0], -1, 2)
        



        # print(x_ta.shape,ta.shape,ta)

        # print(logits,logits.shape)
        h = (tb - ta).to(x_ta.device)  # shape [1]

        # 2. Softmax logits to obtain p(x1 | x_ta, t=ta)
        # print('logit',logits.shape)
        p_1t = torch.softmax(logits, dim=1)  # shape [B, V, ...]
        # print('p_1t',p_1t.shape)

        # 3. Sample a candidate state from p_1t
        # p_perm = p_1t.permute((0, *range(2, p_1t.ndim), 1)).contiguous()
        p_1t_prob = p_1t.permute((0, 2, 3, 1)).contiguous() #for no sprase
        # p_1t_prob = p_1t.contiguous()

        # x_tb_candidate = categorical(p_1t_prob)
        x_tb_candidate = categorical(p_1t_prob.to(dtype=torch.float32))

        # 4. Fetch scheduler parameters
        sched = self.path.scheduler(t=ta)
        k_t = sched.alpha_t
        d_k_t = sched.d_alpha_t

        # 5. Construct u_opt
        delta = F.one_hot(x_tb_candidate, self.vocabulary_size).to(d_k_t.dtype)

        # print('d',delta,delta.shape)
        # print('kt',k_t.shape,'dkt',d_k_t.shape)
        # d_k_t = d_k_t.view(-1, *[1] * (delta.ndim - 1))
        # k_t = k_t.view(-1, *[1] * (delta.ndim - 1))
        d_k_t = d_k_t.view(-1, 1, 1, 1)
        k_t = k_t.view(-1, 1, 1, 1)
        # d_k_t = d_k_t.view(-1, 1, 1)
        # k_t = k_t.view(-1, 1, 1)
        # print(d_k_t.shape,k_t.shape)

        u_opt = (d_k_t / (1 - k_t + 1e-8)) * delta

        # if div_free:
        #     df = div_free(ta)
        #     if df > 0:
        #         p_0 = self.source_distribution_p[(None,) * x_ta.dim()]
        #         u_opt = u_opt + df * d_k_t / (k_t * (1 - k_t)) * ((1 - k_t) * p_0 + k_t * delta)

        delta_t = F.one_hot(x_ta.long(), num_classes=self.vocabulary_size)
        u_opt = torch.where(delta_t.bool(), torch.zeros_like(u_opt), u_opt)
        # print('u',u_opt.shape)

        # 6. Sample final state based on jump intensities
        intensity = u_opt.sum(dim=-1)
        h = h.view(-1, 1, 1) #no sp
        # h = h.view(-1, 1)

        # print(f"h shape: {h.shape}, intensity shape: {intensity.shape}")
        p_jump = 1 - torch.exp(-h * intensity)
        mask_jump = torch.rand_like(intensity) < p_jump
        # print('mask',mask_jump,mask_jump.shape)


        x_tb = x_ta.clone().to(dtype=torch.float32)
        if mask_jump.any():
            x_tb[mask_jump] = categorical(u_opt[mask_jump]).to(dtype=torch.float32)
        # print('xb',x_tb.shape)

        return x_tb


def categorical(probs: Tensor) -> Tensor:
    r"""Categorical sampler according to weights in the last dimension of ``probs`` using :func:`torch.multinomial`.

    Args:
        probs (Tensor): probabilities.

    Returns:
        Tensor: Samples.
    """

    return torch.multinomial(probs.flatten(0, -2), 1, replacement=True).view(
        *probs.shape[:-1]
    )


def get_nearest_times(time_grid: Tensor, t_discretization: Tensor) -> Tensor:


    distances = torch.cdist(
        time_grid.unsqueeze(1),
        t_discretization.unsqueeze(1),
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    nearest_indices = distances.argmin(dim=1)

    return t_discretization[nearest_indices]


class KineticOptimalDiscreteEulerSolver(Solver):
    def __init__(
            self,
            model: nn.Module,
            path: MixtureDiscreteProbPath,
            vocabulary_size: int,
            source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size
        self.source_distribution_p = source_distribution_p

        if source_distribution_p is not None:
            assert source_distribution_p.shape == torch.Size(
                [vocabulary_size]
            ), f"Source distribution p dimension must match vocabulary size {vocabulary_size}."
        if source_distribution_p is None:
            self.source_distribution_p = torch.ones(vocabulary_size) / vocabulary_size


    @torch.no_grad()
    def sample(
            self,
            x_init: Tensor,
            div_free: Union[float, Callable[[float], float]] = 0.0,
            dtype_categorical: torch.dtype = torch.float32,
            time_grid: Tensor = torch.tensor([0.0, 1.0]),
            return_intermediates: bool = False,
            verbose: bool = False,
            eval_batch=None,
    ) -> Tensor:
        """Kinetic-Optimal Discrete Flow Matching Sampler.

        Key Changes:
        1. Replaced mixture path velocity with kinetic-optimal velocity.
        2. Added support for metric-induced paths via `self.path`.
        3. Simplified divergence-free term handling.
        """
        # Initialize time grid and state

        time_grid = time_grid.to(device=x_init.device)
        t_discretization = time_grid
        n_steps = len(time_grid) - 1

        x_t = x_init.clone()
        points = eval_batch[1].to(x_init.device) if eval_batch else None
        res = [x_init.clone()] if return_intermediates else []
        source_p = self.source_distribution_p.to(device=x_init.device)
        # Kinetic-optimal sampling loop
        with tqdm(total=n_steps, desc="Sampling") if verbose else nullcontext() as pbar:
            for i in range(n_steps):
                t = t_discretization[i]
                h = t_discretization[i + 1] - t_discretization[i] if i < n_steps - 1 else 0

                # 1. Get conditional probabilities
                p_1t = self.model(points, x_t, t.expand(x_t.shape[0]), None)
                p_1t = torch.softmax(p_1t, dim=1)

                # Adjust shape to match spatial layout
                if p_1t.dim() == 4:  # [B, C, H, W]
                    p_1t = p_1t.permute(0, 2, 3, 1).contiguous()  # -> [B, H, W, C]

                x_1 = categorical(p_1t.to(dtype_categorical))

                if i == n_steps - 1:
                    x_t = p_1t
                    # x_t = x_1
                else:
                    # 2. Compute kinetic-optimal velocity
                    scheduler_output = self.path.scheduler(t=t)
                    k_t, d_k_t = scheduler_output.alpha_t, scheduler_output.d_alpha_t
                    source_p = self.source_distribution_p.to(x_init.device)

                    # Ensure shape compatibility: [B, D] or [B, H, W, D]
                    if x_1.dim() == 2:  # [B, D]
                        delta_1 = F.one_hot(x_1, self.vocabulary_size).float()
                    else:  # [B, H, W]
                        delta_1 = F.one_hot(x_1, self.vocabulary_size).float()
                        delta_1 = delta_1.view(-1, self.vocabulary_size)  # [B*H*W, D]
                        source_p = source_p.unsqueeze(0).expand(delta_1.size(0), -1)

                    p_t = (1 - k_t) * source_p + k_t * delta_1
                    p_t_dot = d_k_t * (delta_1 - source_p)

                    # Compute flux j_t [..., D, D]
                    j_t = (p_t.unsqueeze(-1) * p_t_dot.unsqueeze(-2) -
                           p_t_dot.unsqueeze(-1) * p_t.unsqueeze(-2)).clamp(min=0)

                    # Compute velocity u_t [..., D, D]
                    u_t = j_t / (p_t.unsqueeze(-2) + 1e-8)

                    # Enforce rate constraints (shape-safe)
                    diag_terms = u_t.sum(dim=-1)  # [..., D]
                    u_t = u_t - torch.diag_embed(diag_terms)

                    # 3. Sample next step
                    if x_t.dim() == 2:  # [B, D]
                        delta_t = F.one_hot(x_t, self.vocabulary_size).float()
                        intensity = (u_t * delta_t.unsqueeze(-1)).sum(dim=-1).sum(dim=-1)
                    else:  # [B, H, W]
                        delta_t = F.one_hot(x_t, self.vocabulary_size).float()
                        delta_t = delta_t.view(-1, self.vocabulary_size)  # [B*H*W, D]
                        intensity = (u_t * delta_t.unsqueeze(-1)).sum(dim=-1).sum(dim=-1)
                        intensity = intensity.view(x_t.shape)  # Restore original shape

                    mask_jump = torch.rand_like(intensity) < 1 - torch.exp(-h * intensity)

                    if mask_jump.any():
                        if x_t.dim() == 2:
                            x_t[mask_jump] = categorical(u_t[mask_jump].to(dtype_categorical))
                        else:
                            flat_x = x_t.view(-1)
                            flat_mask = mask_jump.view(-1)
                            flat_x[flat_mask] = categorical(u_t.view(-1, self.vocabulary_size)[flat_mask])
                            x_t = flat_x.view(x_t.shape)

                if return_intermediates and (i in [0, n_steps // 2, n_steps - 1]):
                    res.append(x_t.clone())

                if verbose:
                    pbar.update(1)

        # return torch.stack(res, dim=0) if return_intermediates else x_t

        return x_t

import os
def get_or_load_t_grid(
    scheduler,
    nfe: int,
    eps: float,
    initType: str = "unif_t",
    cache_dir: str = "./t_grid_cache",
    device: str = "cpu",
    verbose: bool = True
):
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, f"t_grid_{initType}_N{nfe}_eps{eps}_T.pt")

    if os.path.exists(file_path):
        if verbose:
            print(f"[✓] Loaded cached t_grid from {file_path}")
        t_grid = torch.load(file_path).to(device)
    else:
        if verbose:
            print(f"[⏳] Optimizing t_grid for N={nfe}, initType={initType} ...")
        step_optim = StepOptim(scheduler)
        t_grid, _ = step_optim.get_ts_lambdas(N=nfe, eps=1e-3, initType=initType)
        torch.save(t_grid.cpu(), file_path)
        if verbose:
            print(f"[✓] Saved optimized t_grid to {file_path}")
        t_grid = t_grid.to(device)

    return t_grid