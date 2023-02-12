import json
import shutil
import sys
from unittest import mock
import ConfigSpace as CS
import subprocess
import os
from .search_space import AdapterSearchSpace
from typing import Optional, Union, Dict, Any, List
import torch
from settings import TASK_SETTINGS, TASK_SETTINGS_SO
from definition import ROOT_DIR
from botorch.test_functions.multi_objective import DH1
from botorch.test_functions.synthetic import Levy
import math


class Problem:

    def __init__(
            self,
            adapter_name: str,
            task_name: str,
            search_space: AdapterSearchSpace,
            save_path: str,
            data_path: str,
            model_path: str,
            noise_std: float = 1e-4,
            objectives: List[str] = None,
            maximization: bool = True,
            logger=None,
            seed: int = None,
            resplit_dataset: bool = False,
            final_test: bool = False,
            is_large: bool = False,
            mock_run: bool = False) -> None:
        self.adapter_name = adapter_name
        self.task_name = task_name
        self.is_large = is_large
        self.final_test = final_test
        # available_objectives = ["param", "f1", "acc"]
        # assert all([o in available_objectives for o in objectives])
        self.objectives = objectives or TASK_SETTINGS[self.task_name].get(
            "objectives", None)
        if len(self.objectives) == 1:
            TASK_SETTINGS[self.task_name]['ref_point'] = TASK_SETTINGS_SO[self.task_name]['ref_point']
            TASK_SETTINGS[self.task_name]['objectives'] = TASK_SETTINGS_SO[self.task_name]['objectives']
            print('the task settings are: ', TASK_SETTINGS[self.task_name])

        print('we are optimising for the following objectives: ', self.objectives)
        if self.objectives is None:
            raise ValueError(
                "either pass the objective input directly or specify in settings.py!")
        # whether we have an multi-objective problem
        self.num_objectives = len(self.objectives)
        self.save_path = save_path
        self.data_path = data_path
        self.model_path = model_path
        self.logger = logger
        self.search_space = search_space
        self.seed = seed
        self.maximization = int(maximization)
        self.mock_run = mock_run
        self.resplit_dataset = resplit_dataset
        # negate the ref_points accordingly
        if self.num_objectives > 1:
            ref_point = TASK_SETTINGS[self.task_name].get("ref_point", None)
            assert len(ref_point) == self.num_objectives
            for i, element in enumerate(ref_point):
                if self.objectives[i] == "param":
                    ref_point[i] *= float(((-1) ** self.maximization))
                else:
                    ref_point[i] *= float((-1) ** (self.maximization + 1))
            self.ref_point = torch.tensor(ref_point)
        else:
            self.ref_point = None
        self.noise_std = noise_std

    def _evaluate_single(
        self,
        config: Union[CS.Configuration, Dict[str, Any]],
        optim_kwargs: Optional[Dict[str, Any]] = None
    ):

        assert self.task_name in TASK_SETTINGS.keys(), \
            f"{self.task_name} is not specified in settings/TASK_SETTINGS. "
        options = TASK_SETTINGS[self.task_name]
        options.update(optim_kwargs or {})
        if "low_resource" in options:
            low_resource = options["low_resource"]
        else:
            low_resource = None
        patience = options["patience"]
        if self.final_test:
            patience = 10
        num_epochs = options["num_epochs"]
        if self.final_test:
            num_epochs = 20
        save_type = options["save_type"]
        num_steps_per_save = options["num_steps_per_save"]

        if isinstance(config, CS.Configuration):
            config_dict = self.search_space.config_to_dict(config)
        else:
            config_dict = config
        config_id = self.search_space.get_config_id(config)
        save_path_this_config = os.path.join(self.save_path, config_id)
        if os.path.exists(save_path_this_config):
            shutil.rmtree(save_path_this_config)
        os.makedirs(save_path_this_config)

        save_file = os.path.join(save_path_this_config, "config.json")

        with open(save_file, "w", encoding="utf8") as fp:
            json.dump(config_dict, fp, indent=2, ensure_ascii=False)
        pass_output_dir = save_path_this_config if save_path_this_config[-1] == "/" else save_path_this_config + "/"
        if self.logger is not None:
            self.logger.info(f"Start training config ID = {config_id}")
        commands = f"""
        cd {ROOT_DIR}/adapterhub
        {sys.executable} nas_search.py\
         --local_dataset_path={self.data_path} --nas_adapter_config_path={save_file} --model_name_or_path={self.model_path} --task_name={self.task_name}\
         --do_train --do_eval --max_seq_length=128 --per_device_train_batch_size=32\
         --learning_rate=1e-4 --num_train_epochs={num_epochs} --output_dir={pass_output_dir} --patience={patience}\
         --seed={self.seed} --logging_strategy={save_type} --save_strategy={save_type} --logging_steps={num_steps_per_save}\
         --save_steps={num_steps_per_save} --overwrite_output_dir --train_adapter --adapter_name={self.adapter_name}
        """
        commands_resplit = f"""
        cd {ROOT_DIR}/adapterhub
        {sys.executable} nas_search.py\
         --local_dataset_path={self.data_path} --nas_adapter_config_path={save_file} --model_name_or_path={self.model_path} --task_name={self.task_name}\
         --do_train --do_eval --do_predict --resplit_dataset --max_seq_length=128 --per_device_train_batch_size=32\
         --learning_rate=1e-4 --num_train_epochs={num_epochs} --output_dir={pass_output_dir} --patience={patience}\
         --seed={self.seed} --logging_strategy={save_type} --save_strategy={save_type} --logging_steps={num_steps_per_save}\
         --save_steps={num_steps_per_save} --overwrite_output_dir --train_adapter --adapter_name={self.adapter_name}
        """
        commands_low_resource = f"""
        cd {ROOT_DIR}/adapterhub
        {sys.executable} nas_search.py\
         --local_dataset_path={self.data_path} --nas_adapter_config_path={save_file} --model_name_or_path={self.model_path} --task_name={self.task_name}\
         --do_train --do_eval --max_seq_length=128 --per_device_train_batch_size=32\
         --learning_rate=1e-4 --num_train_epochs={num_epochs} --output_dir={pass_output_dir}\
         --seed={self.seed} --logging_strategy={save_type} --save_strategy={save_type} --logging_steps={num_steps_per_save}\
         --save_steps={num_steps_per_save} --overwrite_output_dir --train_adapter --max_train_samples={low_resource}
         {sys.executable} run_glue_eval.py --local_dataset_path={self.data_path} --model_name_or_path={self.model_path}\
         --task_name={self.task_name} --do_eval --max_seq_length 128 --per_device_train_batch_size 32\
         --learning_rate=1e-4 --num_train_epochs={num_epochs} --output_dir={pass_output_dir}\
         --overwrite_output_dir --train_adapter
        """
        if not self.resplit_dataset:
            subprocess.call(commands, shell=True)
        else:
            subprocess.call(commands_resplit, shell=True)
        trainer_state = json.load(
            open(os.path.join(save_path_this_config, "trainer_state.json")))
        metrics = trainer_state["log_history"]
        params = json.load(
            open(os.path.join(save_path_this_config, "model_param_dict.json")))
        # when we are maximizing param is negated
        params_adapter = params["adapter"]
        # this is for prefix only
        max_layer = 12
        if self.is_large:
            max_layer = 24
        if self.adapter_name == "pfeiffer":
            num_adapter_layers = 0
            for i in range(0, max_layer):
                if not config_dict[f"leave_out_{i}"]:
                    num_adapter_layers += 1
            bottleneck_size = 1024 / config_dict['reduction_factor']
            params_adapter = num_adapter_layers * bottleneck_size * 1024 * 2
        if self.adapter_name == "prefix":
            num_adapter_layers = 0
            for i in range(0, max_layer):
                if not config_dict[f"leave_out_{i}"]:
                    num_adapter_layers += 1
            print("we have this number of layers with adapter", num_adapter_layers)
            # outdated
            params_adapter = config_dict['reduction_factor'] * \
                num_adapter_layers * 768 * 2

        if self.adapter_name == "mam":
            num_adapter_layers = 0
            for i in range(0, max_layer):
                if not config_dict[f"leave_out_{i}"]:
                    num_adapter_layers += 1
            print("we have this number of layers with adapter", num_adapter_layers)
            prefix_length = 768 / config_dict['reduction_prefix']
            bottleneck_size = 768 / config_dict['reduction_factor']
            if config_dict['reduction_prefix'] == 512:
                prefix_length = 1
            if config_dict['reduction_factor'] == 512:
                bottleneck_size = 1
            parallel_size = bottleneck_size * 2 * 768 * num_adapter_layers
            prefix_size = prefix_length * 768 * num_adapter_layers * 2
            if config_dict['reduction_prefix'] > 768:
                prefix_size = 0
            if config_dict['reduction_factor'] > 768:
                parallel_size = 0
            params_adapter = prefix_size + parallel_size

        if self.adapter_name == "unipelt":
            num_adapter_layers = 0
            for i in range(0, max_layer):
                if not config_dict[f"leave_out_{i}"]:
                    num_adapter_layers += 1
            print("we have this number of layers with adapter", num_adapter_layers)
            prefix_length = 768 / config_dict['reduction_prefix']
            bottleneck_size = 768 / config_dict['reduction_factor']
            lora_rank = 64 / config_dict['reduction_rank']
            if config_dict['reduction_prefix'] == 512:
                prefix_length = 1
            if config_dict['reduction_factor'] == 512:
                bottleneck_size = 1
            parallel_size = bottleneck_size * 2 * 768 * num_adapter_layers
            prefix_size = prefix_length * 768 * num_adapter_layers * 2
            lora_size = lora_rank * 4 * 12 * 768
            if config_dict['reduction_prefix'] > 768:
                prefix_size = 0
            if config_dict['reduction_factor'] > 768:
                parallel_size = 0
            if config_dict['reduction_rank'] > 64:
                lora_size = 0
            params_adapter = prefix_size + parallel_size + lora_size

        if self.adapter_name == "sappa":
            num_adapter_layers = 0
            for i in range(0, max_layer):
                if not config_dict[f"leave_out_{i}"]:
                    num_adapter_layers += 1
            print("we have this number of layers with adapter", num_adapter_layers)
            prefix_length = 768 / config_dict['reduction_prefix']
            serial_size = 768 / config_dict['reduction_serial']
            bottleneck_size = 768 / config_dict['reduction_factor']
            if config_dict['reduction_prefix'] == 512:
                prefix_length = 1

            if config_dict['reduction_factor'] == 512:
                bottleneck_size = 1

            if config_dict['reduction_serial'] == 512:
                serial_size = 1

            parallel_size = bottleneck_size * 2 * 768 * num_adapter_layers
            prefix_size = prefix_length * 768 * num_adapter_layers * 2
            sa_size = serial_size * 2 * 768 * num_adapter_layers
            if config_dict['reduction_prefix'] > 768:
                prefix_size = 0
            if config_dict['reduction_factor'] > 768:
                parallel_size = 0
            if config_dict['reduction_serial'] > 768:
                sa_size = 0

            params_adapter = prefix_size + parallel_size + sa_size
        param = float(params_adapter) / float(params['model']) * \
            100. * float(((-1) ** self.maximization))
        # if not self.maximization:
        #     param = math.log10(param)
        best_acc = 0
        best_f1 = 0
        find_best = True
        if find_best:
            for metric in metrics:
                if "eval_accuracy" in metric:
                    if metric['eval_accuracy'] > best_acc:
                        best_acc = metric['eval_accuracy']
                        if self.task_name == "mrpc":
                            best_f1 = metric['eval_f1']
                if 'eval_matthews_correlation' in metric:
                    if metric['eval_matthews_correlation'] > best_acc:
                        best_acc = metric['eval_matthews_correlation']
                if 'eval_spearmanr' in metric:
                    if metric['eval_spearmanr'] > best_acc:
                        best_acc = metric['eval_spearmanr']
        print(f"Best acc: {best_acc}")

        best_acc *= float((-1) ** (self.maximization + 1))
        best_f1 *= float((-1) ** (self.maximization + 1))
        all_objectives = ["param", "acc", "f1"]
        idx_to_keep = [all_objectives.index(o) for o in self.objectives]
        all_res = torch.tensor([param, best_acc, best_f1])[idx_to_keep]
        if self.logger is not None:
            self.logger.info(f"Config = {config_dict}. ID = {config_id}. "
                             f"Result: #params = {param}, Best acc = {best_acc}. Best F1 = {best_f1}")
        if self.final_test:
            return all_res
        else:
            return all_res + self.noise_std * torch.randn_like(all_res)

    def evaluate_true(self, X: list, ) -> torch.Tensor:
        # todo: change the sequential evaluation to parallel (dependeing on VRAM usage)
        f = self._evaluate_single_mock if self.mock_run else self._evaluate_single
        if self.final_test:
            return f(X)
        res = (
            torch.stack(
                [f(x) for x in X],
            )
            .view(len(X), self.num_objectives)
        )
        # if self.num_objectives == 1:
        #     return res.squeeze(-1)
        return res

    def _evaluate_single_mock(self, X: Union[CS.Configuration, Dict[str, Any]], ) -> torch.Tensor:
        if isinstance(X, dict):
            X = self.search_space.dict_to_config(X)
        X_array = torch.from_numpy(X.get_array())
        if self.num_objectives == 2:
            mock_problem = DH1(X_array.shape[0])
        elif self.num_objectives == 1:
            mock_problem = Levy(X_array.shape[0], negate=True)
        return mock_problem(X_array).squeeze(-1)

    def __call__(self, X: list) -> torch.Tensor:
        return self.evaluate_true(X)
