from time import time

import torch
from typing import Optional, Dict, Any
import numpy as np
from adapterhub.bo.models import initialize_model, get_acqf, optimize_acqf
from settings import DEFAULT_BO_SETTINGS, TASK_SETTINGS
import random
import os
from adapterhub.bo.search_space import AdapterSearchSpace
import argparse
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective import is_non_dominated, infer_reference_point
from adapterhub.bo.base_function import Problem
import botorch
import logger as logging_setup
import logging
from definition import ROOT_DIR
from adapterhub.bo.utils import apply_normal_copula_transform

parser = argparse.ArgumentParser()
parser.add_argument('-m', "--method", type=str,
                    default="bo", choices=["bo", "random"])
parser.add_argument('-sp', "--save_path", type=str,
                    default=f"{ROOT_DIR}/output/")
parser.add_argument('-dp', "--data_path", type=str,
                    default=f"{ROOT_DIR}/datasets/glue/")
parser.add_argument('-mp', "--model_path", type=str,
                    default=f"{ROOT_DIR}/models/bert-base-uncased/")
parser.add_argument('-t', "--task", type=str, default="mrpc")
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument("--no_copula", action="store_true",
                    help="Whether to apply copula standardization")
parser.add_argument('-mi', '--max_iter', type=int, default=200)
parser.add_argument('-bs', '--batch_size', type=int, default=4)
parser.add_argument('-ni', "--n_init", type=int, default=20)
parser.add_argument('-o', "--objectives", nargs="+", default=["param", "acc"])
parser.add_argument('--acq_optim_method', type=str,
                    default="local_search", choices=["local_search", "random"])
parser.add_argument('--base_model', type=str,
                    default="saasgp", choices=["saasgp", "gp"])
parser.add_argument('--overwrite', action="store_true")
parser.add_argument('--mock_run', action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument('-an', "--adapter_name", type=str, default="ours")
parser.add_argument('-rd', "--resplit_dataset", type=bool, default=False)
args, _ = parser.parse_known_args()
logger = logging_setup.get_logger(__name__)
logger.setLevel(logging.DEBUG)


# If the save_path or the model path is a relative path
if args.save_path.startswith("./"):
    args.save_path = args.save_path.split("./")[1]
    args.save_path = os.path.join(ROOT_DIR, args.save_path)
if args.model_path.startswith("./"):
    args.model_path = args.save_path.split("./")[1]
    args.model_path = os.path.join(ROOT_DIR, args.model_path)

save_path = os.path.join(
    args.save_path, f"NAS_{args.task}_{args.acq_optim_method}_seed_{args.seed}_bs_{args.batch_size}_{args.method}"
)
data_path = os.path.join(
    args.data_path, args.task
)
model_path = args.model_path


resume = False
if not os.path.exists(save_path):
    os.makedirs(save_path)
elif args.resume:
    resume = True
    logger.info(f"Resuming from {save_path}")
elif not args.overwrite:
    raise FileExistsError(
        f"{save_path} is not empty. Change to another save_path, or enable the overwrite flag.")
logging_path = os.path.join(save_path, "train_logs.log")
logging_setup.setup_logging(logging_path, 'w')
logger.info(f"Save dir = {save_path}")
logger.info(vars(args))
if args.mock_run:
    logger.warning(
        "This run is a mock run. No actual training will be performed.")

# Fix the seeds
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def run_one_replication(
        acqf_optim_kwargs: Optional[dict] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        dtype: torch.dtype = torch.float,
        device: str = None,
        save_frequency: int = 1,
):

    iterations = max(1, args.max_iter // args.batch_size)
    batch_size = args.batch_size
    tkwargs = {"dtype": dtype, "device": device or (
        "cuda" if torch.cuda.is_available() else "cpu")}
    n_initial_points = args.n_init
    # Specify default args
    acqf_optim_kwargs = acqf_optim_kwargs or DEFAULT_BO_SETTINGS["ACQF_OPTIM_KWARGS"]
    model_kwargs = model_kwargs or DEFAULT_BO_SETTINGS["MODEL_KWARGS"]
    is_large = '-large' in model_path
    ss = AdapterSearchSpace(seed=seed, is_large=is_large)
    f = Problem(
        adapter_name=args.adapter_name,
        task_name=args.task,
        search_space=ss,
        save_path=save_path,
        data_path=data_path,
        model_path=model_path,
        objectives=args.objectives,
        logger=logger,
        seed=args.seed,
        mock_run=args.mock_run,
        resplit_dataset=args.resplit_dataset,
        is_large=is_large,
    )
    # generate initial data or load from a previous run.
    n_initial_points = n_initial_points or 20
    if resume:
        with open(os.path.join(save_path, "result_stats.pt"), "rb") as fp:
            load_dict = torch.load(fp, map_location="cpu")
        Z = load_dict["Z"].to(**tkwargs)
        X = load_dict["X"]
        Y = load_dict["Y"].to(**tkwargs)
        wall_time_prev = load_dict["wall_time"]
        best_objs = load_dict["best_obj"]
        if len(X) < n_initial_points:
            remaining_init = n_initial_points - len(X)
            new_Z, new_X = list(zip(*[ss.sample_configuration(return_dict_repr=True)
                                      for _ in range(remaining_init)]))
            new_Z = torch.stack(
                [torch.from_numpy(z).to(**tkwargs) for z in new_Z])
            Z = torch.cat([Z, new_Z], dim=0)
            X += list(new_X)
            new_Y = f(new_X).to(**tkwargs)
            Y = torch.cat([Y, new_Y])
    else:
        Z, X = list(zip(*[ss.sample_configuration(return_dict_repr=True)
                    for _ in range(n_initial_points)]))
        X = list(X)
        Z = torch.stack([torch.from_numpy(z).to(**tkwargs) for z in Z])
        Y = f(X).to(**tkwargs)
        wall_time_prev = None
        best_objs = None

    is_moo = f.num_objectives > 1
    default_fraction = 0.0
    if not is_moo:
        default_fraction = 0.0
    if is_moo:
        if f.ref_point is not None:
            ref_point = f.ref_point.to(**tkwargs)
            # inferred_ref_point = False
        else:
            non_dominated_points = Y[is_non_dominated(Y)]
            ref_point = infer_reference_point(non_dominated_points)
            # inferred_ref_point = True
        logger.info(f"Inferred reference point = {ref_point}")

    # Set some counters to keep track of things.
    start_time = time()
    existing_iterations = len(X) // batch_size
    wall_time = torch.zeros(iterations, dtype=dtype)
    # if wall_time_prev is not None:
    #     wall_time[:existing_iterations] = wall_time_prev.view(-1)
    if is_moo:
        bd = DominatedPartitioning(Y=Y, ref_point=ref_point)
        if best_objs is None:
            best_objs = bd.compute_hypervolume().view(-1).cpu()
        else:
            best_objs = torch.cat(
                [best_objs, bd.compute_hypervolume().view(-1).cpu()], dim=0)
    else:
        obj = Y
        if best_objs is None:
            best_objs = obj.max().view(-1).cpu()
        else:
            best_objs = torch.cat([best_objs, obj.max().view(-1).cpu()], dim=0)

    for i in range(existing_iterations, iterations):
        logger.info(
            f"Starting seed {seed}, iteration {i}, "
            f"time: {time() - start_time}, "
            f"Last obj: {Y[-batch_size:]}"
            f"current best obj: {best_objs[-1]}."
        )
        if args.method == "random":
            z_temp, candidates_x = list(
                zip(*[ss.sample_configuration(return_dict_repr=True) for _ in range(batch_size)]))
            candidates_z = torch.stack(
                [torch.from_numpy(z).to(**tkwargs) for z in z_temp])

        elif args.method == "bo":
            if args.base_model == "saasgp":
                base_model_class = botorch.models.fully_bayesian.SaasFullyBayesianSingleTaskGP
            elif args.base_model == "gp":
                base_model_class = botorch.models.SingleTaskGP
            else:
                raise NotImplementedError(
                    f"Unknown base model class {args.base_model}")

            model, ecdfs = initialize_model(
                train_X=Z,
                train_Y=Y.clone(),
                base_model_class=base_model_class,
                optimizer_kwargs=model_kwargs,
                apply_copula=not args.no_copula,
            )

            Y_tf = Y if args.no_copula else apply_normal_copula_transform(Y=Y, ecdfs=ecdfs)[
                0]

            # Update reference point
            if is_moo:
                # if inferred_ref_point:
                #     non_dominated_points = Y_tf[is_non_dominated(Y_tf)]
                #     ref_point = infer_reference_point(non_dominated_points)
                #     logger.info(f"Inferred reference point = {ref_point}")
                # else:
                #     ref_point = f.ref_point.to(**tkwargs)
                # the
                ref_point_tf = ref_point if args.no_copula else apply_normal_copula_transform(
                    Y=ref_point.reshape(1, -1), ecdfs=ecdfs)[0].squeeze()
            # Choose the best points or the pareto front and pass as baseline
            acq_func = get_acqf(
                model,
                X_baseline=Z,
                train_Y=Y_tf,
                label="nehvi" if is_moo else "ei",
                ref_point=ref_point_tf if is_moo else None,
            )

            # Finding the Pareto front or the best points seen so far.
            if is_moo:
                topk_ind = is_non_dominated(Y).nonzero().view(-1).tolist()
            else:
                _, topk_ind = torch.topk(Y.reshape(-1), min(3, Y.shape[0]))
            best_points = [X[i] for i in topk_ind]

            # generate candidates by optimizing the acqf function
            candidates_x, candidates_z, _ = optimize_acqf(
                acqf=acq_func,
                search_space=ss,
                optim_method=args.acq_optim_method,
                q=batch_size,
                X_baseline=best_points,
                device=tkwargs["device"],
                dtype=tkwargs["dtype"],
                optim_kwargs=acqf_optim_kwargs,
                fraction=default_fraction
            )
        else:
            raise NotImplementedError(
                f"Method named {args.method} is not currently supported!")
        # evaluate the problem``
        new_y = f(candidates_x).to(**tkwargs)

        X += candidates_x
        Y = torch.cat([Y, new_y], dim=0)
        Z = torch.cat([Z, candidates_z], dim=0)

        wall_time[i] = time() - start_time

        if is_moo:
            bd = DominatedPartitioning(ref_point=ref_point, Y=Y)
            best_obj = bd.compute_hypervolume()
        else:
            obj = Y
            best_obj = obj.max().view(-1)[0].cpu()

        best_objs = torch.cat([best_objs, best_obj.view(-1).cpu()], dim=0)
        # Periodically save the output.
        if save_frequency is not None and iterations % save_frequency == 0:

            output_dict = {
                "Z": Z.detach().cpu(),
                "X": X,
                "Y": Y.detach().cpu(),
                "wall_time": wall_time[: i + 1],
                "best_obj": best_objs,
            }
            with open(os.path.join(save_path, f"result_stats.pt"), "wb") as fp:
                torch.save(output_dict, fp)

    # Save the final output
    output_dict = {
        "Z": Z.detach().cpu(),
        "X": X,
        "Y": Y.detach().cpu(),
        "wall_time": wall_time,
        "best_obj": best_objs,
    }
    with open(os.path.join(save_path, f"result_stats.pt"), "wb") as fp:
        torch.save(output_dict, fp)

    y = output_dict['Y']
    if args.task in ["mnli", "qqp"]:
        _, non_dom_idx = torch.topk(y[:, 1].reshape(-1), 1)
    else:
        if is_moo:
            non_dom_idx = is_non_dominated(y).nonzero().squeeze()
        else:
            _, non_dom_idx = torch.topk(Y.reshape(-1), min(5, Y.shape[0]))
    for inx in non_dom_idx:
        logger.info(f"pareto point: {y[inx]}")
        logger.info(f"point architecture: {X[inx]}")
        results_list = []
        for test_seed in range(40, 45):
            save_test_path = os.path.join(save_path, f"test/seed_{test_seed}/")
            f_result = Problem(
                adapter_name=args.adapter_name,
                task_name=args.task,
                search_space=ss,
                save_path=save_test_path,
                data_path=data_path,
                model_path=model_path,
                objectives=args.objectives,
                logger=logger,
                seed=test_seed,
                mock_run=args.mock_run,
                resplit_dataset=args.resplit_dataset,
                is_large=is_large,
                final_test=True,
            )

            Y_result = f_result(X[inx]).to(**tkwargs)
            results_list.append(Y_result)
        logger.info(f"test results: {results_list}")
        logger.info(
            f"test results mean: {torch.mean(torch.stack(results_list), dim=0)}")
        logger.info(
            f"test results std: {torch.std(torch.stack(results_list), dim=0)}")


if __name__ == "__main__":
    run_one_replication()
