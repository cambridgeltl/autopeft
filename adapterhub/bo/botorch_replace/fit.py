#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Drop-in replacement of the botorch version which could fail due to torchscript errors.

r"""
Utilities for model fitting.
"""

from __future__ import annotations
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from pyro.infer.mcmc import MCMC, NUTS


FAILED_CONVERSION_MSG = (
    "Failed to convert ModelList to batched model. "
    "Performing joint instead of sequential fitting."
)


def fit_fully_bayesian_model_nuts(
    model: SaasFullyBayesianSingleTaskGP,
    max_tree_depth: int = 6,
    warmup_steps: int = 512,
    num_samples: int = 256,
    thinning: int = 16,
    disable_progbar: bool = False,
) -> None:
    r"""Fit a fully Bayesian model using the No-U-Turn-Sampler (NUTS)


    Args:
        model: SaasFullyBayesianSingleTaskGP to be fitted.
        max_tree_depth: Maximum tree depth for NUTS
        warmup_steps: The number of burn-in steps for NUTS.
        num_samples:  The number of MCMC samples. Note that with thinning,
            num_samples / thinning samples are retained.
        thinning: The amount of thinning. Every nth sample is retained.
        disable_progbar: A boolean indicating whether to print the progress
            bar and diagnostics during MCMC.

    Example:
        >>> gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(gp)
    """
    model.train()

    # Do inference with NUTS
    try:
        nuts = NUTS(
            model.pyro_model.sample,
            jit_compile=True,
            full_mass=True,
            ignore_jit_warnings=True,
            max_tree_depth=max_tree_depth,
        )
        mcmc = MCMC(
            nuts,
            warmup_steps=warmup_steps,
            num_samples=num_samples,
            disable_progbar=disable_progbar,
        )
        mcmc.run()
    except:
        nuts = NUTS(
            model.pyro_model.sample,
            jit_compile=False,
            full_mass=True,
            ignore_jit_warnings=True,
            max_tree_depth=max_tree_depth,
        )
        mcmc = MCMC(
            nuts,
            warmup_steps=warmup_steps,
            num_samples=num_samples,
            disable_progbar=disable_progbar,
        )
        mcmc.run()

    # Get final MCMC samples from the Pyro model
    mcmc_samples = model.pyro_model.postprocess_mcmc_samples(
        mcmc_samples=mcmc.get_samples()
    )
    for k, v in mcmc_samples.items():
        mcmc_samples[k] = v[::thinning]

    # Load the MCMC samples back into the BoTorch model
    model.load_mcmc_samples(mcmc_samples)
    model.eval()
