# initialize a GP model from botorch
import random

import numpy as np
from gpytorch.constraints import Interval
from botorch.models.transforms.outcome import Standardize
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.priors import GammaPrior
import torch
import gpytorch
import botorch
from typing import List, Optional, Dict, Any
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from gpytorch.likelihoods import GaussianLikelihood
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
)
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from adapterhub.bo.search_space import AdapterSearchSpace
from adapterhub.bo.utils import filter_invalid, apply_normal_copula_transform
from adapterhub.bo.botorch_replace.fit import fit_fully_bayesian_model_nuts
import time


def get_kernel(use_ard: bool = False,
               use_ard_binary: bool = True,
               non_binary_indices: List[int] = None,
               binary_indices: List[int] = None):
    """Helper method to initialize the kernels"""
    kernels = []
    if not use_ard:
        kernels.append(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=None
            )
        )
    else:
        if use_ard_binary:
            kernels.append(
                gpytorch.kernels.MaternKernel(
                    nu=2.5,
                )
            )
        else:
            assert non_binary_indices is not None
            assert binary_indices is not None
            if len(binary_indices):
                kernels.append(
                    gpytorch.kernels.MaternKernel(
                        nu=2.5,
                        active_dims=binary_indices,
                        ard_num_dims=None
                    )
                )
            kernels.append(
                gpytorch.kernels.MaternKernel(
                    nu=2.5,
                    active_dims=non_binary_indices,
                    ard_num_dims=len(non_binary_indices)
                )
            )
    kernel = kernels[0]
    if len(kernels) > 1:
        for k in kernels[1:]:
            kernel *= k
    return gpytorch.kernels.ScaleKernel(kernel)


def initialize_model(
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        base_model_class=None,
        fit_model: bool = True,
        verbose: bool = False,
        apply_copula: bool = False,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        covar_module_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
):
    if train_Y.ndim == 1:
        train_Y = train_Y.reshape(-1, 1)
    model_kwargs = []
    # define the default values
    optimizer_kwargs_ = {
        # default settings for fitting the SAAS-GP model
        "warmup_steps": 256,
        "num_samples": 128,
        "thinning": 16
    }
    optimizer_kwargs_.update(optimizer_kwargs or {})
    covar_module_kwargs = covar_module_kwargs or {}

    # initialize the model classes -- note that each dimension of output can have a different model class
    if base_model_class is None:
        base_model_class = SingleTaskGP

    # initialize the covariance module
    if base_model_class != SaasFullyBayesianSingleTaskGP:
        covar_module = covar_module or get_kernel(**covar_module_kwargs)
    else:
        covar_module = None     # for Saasmodel we don't need to specify the covariance module

    if apply_copula:
        train_Y, ecdfs = apply_normal_copula_transform(train_Y)
    else:
        ecdfs = None

    for i in range(train_Y.shape[-1]):
        model_kwargs.append(
            {
                "train_X": train_X,
                "train_Y": train_Y[..., i: i + 1],
                "outcome_transform": None if apply_copula else Standardize(m=1),
            }
        )
        if base_model_class != SaasFullyBayesianSingleTaskGP:
            model_kwargs[i]["covar_module"] = covar_module
            model_kwargs[i]["likelihood"] = GaussianLikelihood(
                noise_prior=GammaPrior(0.9, 10.0),
                noise_constraint=Interval(1e-7, 1e-3)
            )
    models = [base_model_class(**model_kwargs[i])
              for i in range(train_Y.shape[-1])]
    if len(models) > 1:
        model = ModelListGP(*models).to(device=train_X.device)
    else:
        model = models[0].to(device=train_X.device)

    if verbose:
        print(model)

    # fit the model
    if fit_model:
        if len(models) == 1:
            if base_model_class == SaasFullyBayesianSingleTaskGP:
                n_attempt = 0
                while n_attempt < 3:
                    try:
                        fit_fully_bayesian_model_nuts(model,
                                                      warmup_steps=optimizer_kwargs_[
                                                          "warmup_steps"],
                                                      num_samples=optimizer_kwargs_[
                                                          "num_samples"],
                                                      thinning=optimizer_kwargs_["thinning"],
                                                      disable_progbar=True)
                        break
                    except Exception as e:
                        n_attempt += 1
                if n_attempt >= 3:
                    raise ValueError(
                        f"Fitting SAASGP Failed with error message {e}")
            else:
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                    model.likelihood, model).to(device=train_X.device)
                fit_gpytorch_torch(mll)
        else:
            # if there is at least one model that contains a SAAS-GP model -- note that SAAS-GP model needs to
            # be fitted differently.
            if base_model_class == SaasFullyBayesianSingleTaskGP:
                for i in range(train_Y.shape[-1]):
                    if base_model_class == SaasFullyBayesianSingleTaskGP:
                        fit_fully_bayesian_model_nuts(model.models[i],
                                                      warmup_steps=optimizer_kwargs_[
                                                          "warmup_steps"],
                                                      num_samples=optimizer_kwargs_[
                                                          "num_samples"],
                                                      thinning=optimizer_kwargs_["thinning"],
                                                      disable_progbar=True)
                    else:
                        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.models[i].likelihood, model.models[i]).to(
                            device=train_X.device)
                        fit_gpytorch_torch(mll)
            else:
                mll = gpytorch.mlls.SumMarginalLogLikelihood(
                    model.likelihood, model).to(device=train_X.device)
                n_attempt = 0
                while n_attempt < 3:
                    try:
                        fit_gpytorch_torch(mll, options={"disp": False})
                        break
                    except Exception as e:
                        n_attempt += 1
                    if n_attempt >= 3:
                        print(
                            f"Fitting model failed after {n_attempt} number of attempts with error {e}!")

    return model, ecdfs


def get_EI(
    model,
    train_Y: torch.Tensor,
) -> botorch.acquisition.AcquisitionFunction:
    acq = qExpectedImprovement(
        model,
        train_Y.max(),
    )
    return acq


def get_qEHVI(
        model,
        train_Y: torch.Tensor,
        ref_point: torch.Tensor,
        X_baseline: torch.Tensor,
        noisy: bool = True
) -> qExpectedHypervolumeImprovement:
    bd = FastNondominatedPartitioning(ref_point=ref_point, Y=train_Y)
    if noisy:
        acq = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            prune_baseline=True,
        )
    else:
        acq = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=bd,
        )
    return acq


def get_acqf(model,
             X_baseline: torch.Tensor,
             train_Y: torch.Tensor,
             label: str = None,
             ref_point: Optional[torch.Tensor] = None,
             ):
    if train_Y.shape[-1] == 1:  # one-dimensional case
        label = label or "ei"
        if label == "ei":
            acq_func = get_EI(model, train_Y,)
        else:
            raise NotImplementedError(
                f"acquisition function {label} is not implemented!")
    else:
        label = label or "nehvi"
        if label == "nehvi":
            if ref_point is None:
                non_dominated_points = train_Y[botorch.utils.multi_objective.is_non_dominated(
                    train_Y)]
                ref_point = botorch.utils.multi_objective.infer_reference_point(
                    non_dominated_points)
            acq_func = get_qEHVI(
                model=model,
                train_Y=train_Y,
                ref_point=ref_point,
                X_baseline=X_baseline
            )
        else:
            raise NotImplementedError(
                f"acquisition function {label} is not implemented!")
    return acq_func


def optimize_acqf(acqf: botorch.acquisition.AcquisitionFunction,
                  search_space: AdapterSearchSpace,
                  optim_method: str = "local",
                  optim_kwargs: Optional[Dict[str, Any]] = None,
                  unique: bool = True,
                  q: int = 4,
                  device: str = "cpu",
                  dtype: torch.dtype = torch.float,
                  X_baseline=None,
                  fraction: float = 0.0
                  ):
    """
    Optimize the acquisition function

    """
    tkwargs = {"dtype": dtype, "device": device}
    base_X_pending = acqf.X_pending if q > 1 else None
    base_Z_avoid = None
    Z_avoid = None      # we don't know the input dim yet...

    options = {
        "max_eval": 1000,
        "n_restart": 10,
        "sample_around_best": True,  # whether to initialize around best points seen so far
        "fraction_random": fraction,
        # generate starting points by perturbing one of the top 'top_k' points in terms of obj func value
        # "n_neighbors": 10,        # when unspecified, this will be set to be equal to search dimension.
    }
    print("optimizing acquisition function using options", options)
    options.update(optim_kwargs or {})
    candidate_list = []
    candidate_list_z = []
    for i in range(q):
        if optim_method == "random":
            n_rand = options["max_eval"]
        elif optim_method == "local_search":
            # this specifies how many starting points are generated from perturbing the best
            n_rand = int(options["n_restart"] * (1. - options["fraction_random"]))

        if optim_method == "local_search" and options["sample_around_best"]:
            assert X_baseline is not None

            X0, Z0 = [], []
            patience = 100
            while len(X0) < n_rand and patience > 0:
                # perturb X0_ to get a neighbour -- locally unset seed, otherwise
                #   same neighbour is chosen repeatedly
                x0_ = np.random.RandomState(int(time.time())).choice(X_baseline)
                z0, x0 = search_space.get_neighbours(x0_,
                                                     stochastic=True,
                                                     n_neighbors=1,
                                                     return_dict_repr=True,
                                                     # seed=int(time.time())
                                                     )
                z0 = torch.from_numpy(np.array(z0).reshape(1, -1)).to(**tkwargs)
                if Z_avoid is None:
                    Z_avoid = torch.zeros(0, z0[0].reshape(1, -1).shape[1], **tkwargs)
                z0 = filter_invalid(z0, Z_avoid)
                if not z0.shape[0]:
                    patience -= 1
                    continue
                X0 += x0
                Z0.append(z0)
            # patience exhausted and n_rand is still unsatisfied OR some points need to be generated randomly
            # -- fill with random designs
            if len(X0) < options["n_restart"]:
                while len(X0) < options["n_restart"]:
                    z0, x0 = search_space.sample_configuration(return_dict_repr=True)
                    z0 = torch.from_numpy(np.array(z0).reshape(1, -1)).to(**tkwargs)
                    z0 = filter_invalid(z0, Z_avoid)
                    if z0.shape[0] == 0:
                        continue
                    X0.append(x0)
                    Z0.append(z0)
            Z0 = torch.cat(Z0)
        else:
            X0, Z0 = [], []
            while len(X0) < n_rand:
                z0, x0 = search_space.sample_configuration(return_dict_repr=True)
                z0 = torch.from_numpy(np.array(z0).reshape(1, -1)).to(**tkwargs)
                if Z_avoid is None:
                    Z_avoid = torch.zeros(0, z0.shape[-1], **tkwargs)
                z0 = filter_invalid(z0.to(Z_avoid), Z_avoid)
                if z0.shape[0] == 0:
                    continue
                X0.append(x0)
                Z0.append(z0)
            Z0 = torch.cat(Z0)

        best_xs = list(X0)
        best_zs = Z0
        with torch.no_grad():
            best_acqvals = acqf(Z0.unsqueeze(1))

        if optim_method == "local_search":
            for j, z0 in enumerate(Z0):
                curr_z, curr_f = z0.clone(), best_acqvals[j]
                curr_x = X0[j]
                n_evals_left = options["max_eval"] // len(Z0)

                while n_evals_left > 0:
                    with torch.no_grad():
                        neighbors_z, neighbors_x = search_space.get_neighbours(curr_x,
                                                                               # seed=int(time.time()),
                                                                               stochastic=True,
                                                                               n_neighbors=options.get(
                                                                                   "n_neighbors", None),
                                                                               return_dict_repr=True)

                        neighbors_z = torch.stack(
                            [torch.from_numpy(z).to(**tkwargs) for z in neighbors_z])
                        neighbors_z, valid_idx = filter_invalid(
                            neighbors_z, Z_avoid, return_index=True)
                        neighbors_x = [neighbors_x[i] for i in valid_idx]
                        if len(neighbors_x) == 0:
                            break
                        acq_val_neighbors = acqf(neighbors_z.unsqueeze(1))
                        n_evals_left -= neighbors_z.shape[0]
                        if acq_val_neighbors.max() > curr_f:
                            indbest = acq_val_neighbors.argmax()
                            curr_z = neighbors_z[indbest]
                            curr_x = neighbors_x[indbest]
                            curr_f = acq_val_neighbors[indbest]

                best_zs[j, :], best_acqvals[j] = curr_z, curr_f
                print("best_zs[j, :], best_acqvals[j]", curr_z, curr_f)
                # print(best_xs)
                best_xs.append(curr_x)

        best_idx = best_acqvals.argmax()
        candidate_list.append(best_xs[best_idx])
        candidate_list_z.append(best_zs[best_idx].unsqueeze(0))

        # set pending points
        candidates_z = torch.cat(candidate_list_z, dim=-2)
        if q > 1:
            acqf.set_X_pending(
                torch.cat([base_X_pending, candidates_z], dim=-2)
                if base_X_pending is not None
                else candidates_z
            )
            # Update points to avoid if unique is True
            if unique:
                Z_avoid = (
                    torch.cat([base_Z_avoid, candidates_z], dim=-2)
                    if base_Z_avoid is not None
                    else candidates_z
                )
    # Reset acq_func to original X_pending state
    if q > 1:
        if hasattr(acqf, "set_X_pending"):
            acqf.set_X_pending(base_X_pending)
    with torch.no_grad():
        # compute joint acquisition value
        acq_value = acqf(candidates_z.unsqueeze(1))
    return candidate_list, candidates_z, acq_value
