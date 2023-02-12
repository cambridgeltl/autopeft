#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import json
import logging
import os
import random
import sys
sys.path.append("../")
from dataclasses import dataclass, field
from typing import Optional
from definition import ROOT_DIR
import re
import shutil
import datasets
import numpy as np
from datasets import load_dataset, load_metric
import transformers
import transformers.adapters.composition as ac
from transformers import (
    AdapterConfig,
    AdapterTrainer,
    AutoAdapterModel,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    MultiLingAdapterArguments,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    ConfigUnion,
    ParallelConfig,
    PrefixTuningConfig,
    AutoPEFTConfig,
    LoRAConfig,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import numpy as np
import transformers.adapters.composition as ac

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.19.0")

require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


def split_datasets(train_ds, n: int = None):
    logger.info(
        "Spliting the train/eval datasets into train/eval by "
        "using 90% and 10% of train as train and eval and eval as test."
    )
    if n is None:
        n = len(train_ds)
        logger.info(f"Using the whole train dataset of {n} samples.")
    else:
        logger.info(f"Reducing the train dataset to only {n} samples.")

    split_at = int(n * 0.90)
    train_ds = train_ds.shuffle()
    new_eval_ds = train_ds.select(range(split_at, n))
    new_train_ds = train_ds.select(range(split_at))
    return new_train_ds, new_eval_ds


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    local_dataset_path: str = field(
        metadata={"help": "load the local copy of dataset"}
    )
    patience: int = field(
        default=10,
        metadata={
            "help": "the number of epochs to wait before early stopping"
        },
    )
    resplit_dataset: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to resplit the dataset."},
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " +
                  ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={
                                     "help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    adapter_name: str = field(
        metadata={"help": "The name of the adapter to use."}
    )
    nas_adapter_config_path: str = field(
        metadata={"help": "nas_adapter config path"}
    )
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


def get_all_checkpoint(folder):
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return checkpoints


default_arg = {
    "non_linearity": "relu",
    "residual_before_ln": True,     # default is True, PA is true, previous exps False
    "adapter_residual_before_ln": False,
    "ln_after": False,
    "ln_before": False,
    "reduction_factor": 64,
    "leave_out": [],
    "mh_adapter": False,
    "output_adapter": True,
    "original_ln_before": True,
    "original_ln_after": True,
    "is_parallel": False,
    # "scaling": 1.0,
}

default_prefix_arg = {
}

default_mam_arg = {
}



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    training_args.evaluation_strategy = training_args.logging_strategy
    training_args.eval_steps = training_args.logging_steps
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    raw_datasets = datasets.load_from_disk(data_args.local_dataset_path)

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in [
            "float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # We use the AutoAdapterModel class here for better adapter support.
    model = AutoAdapterModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model_param_dict = {}
    model_param_dict['model'] = model.num_parameters()
    logger.info(
        f"model number of parameters before heads{model.num_parameters()}")

    model.add_classification_head(
        data_args.task_name or "glue",
        num_labels=num_labels,
        id2label={i: v for i, v in enumerate(
            label_list)} if not is_regression else None,
    )
    logger.info(
        f"model number of parameters before adapter{model.num_parameters()}")
    model_param_dict['w. heads'] = model.num_parameters()

    # Setup adapters
    if adapter_args.train_adapter:
        task_name = data_args.task_name or "glue"
        # random_seed = int(np.random.randint(1000))
        logger.info(f"random seed for adapter training: {training_args.seed}")
        f = open(model_args.nas_adapter_config_path)
        random_args = json.load(f)

        leave_out_list = []
        number_layer = 12
        if 'large' in model_args.model_name_or_path:
            number_layer = 24
        for i in range(0, number_layer):
            if random_args[f"leave_out_{i}"]:
                leave_out_list.append(int(i))
            del random_args[f"leave_out_{i}"]
        random_args["leave_out"] = leave_out_list

        # prefix tuning only
        if model_args.adapter_name == 'prefix':
            random_args['prefix_length'] = random_args['reduction_factor']
            del random_args['reduction_factor']
            default_arg = default_prefix_arg.copy()

        if model_args.adapter_name == 'pfeiffer':
            default_arg = default_mam_arg.copy()

        if model_args.adapter_name == 'mam':
            if random_args['reduction_prefix'] == 512:
                random_args['prefix_length'] = 1
            else:
                random_args['prefix_length'] = 768 / \
                    random_args['reduction_prefix']
            if random_args['reduction_factor'] == 512:
                random_args['reduction_factor'] = 768
            del random_args['reduction_prefix']
            default_arg = default_mam_arg.copy()

        if model_args.adapter_name == 'unipelt':
            if random_args['reduction_prefix'] == 512:
                random_args['prefix_length'] = 1
            else:
                random_args['prefix_length'] = 768 / \
                    random_args['reduction_prefix']
            if random_args['reduction_factor'] == 512:
                random_args['reduction_factor'] = 768
            del random_args['reduction_prefix']
            random_args['r'] = 64 / random_args['reduction_rank']
            del random_args['reduction_rank']
            default_arg = default_mam_arg.copy()

        if model_args.adapter_name == 'sapa':
            if random_args['reduction_prefix'] == 512:
                random_args['reduction_serial'] = 768
            else:
                random_args['reduction_serial'] = random_args['reduction_prefix']
            if random_args['reduction_factor'] == 512:
                random_args['reduction_factor'] = 768
            del random_args['reduction_prefix']
            default_arg = default_mam_arg.copy()

        if model_args.adapter_name == 'sappa':
            exclude_pa = False
            exclude_sa = False
            exclude_prefix = False
            if random_args['reduction_prefix'] == 512:
                random_args['prefix_length'] = 1
            else:
                random_args['prefix_length'] = 768 / \
                    random_args['reduction_prefix']
            if random_args['reduction_factor'] == 512:
                random_args['reduction_factor'] = 768
            if random_args['reduction_serial'] == 512:
                random_args['reduction_serial'] = 768
            del random_args['reduction_prefix']
            if random_args['reduction_serial'] > 768:
                random_args['reduction_serial'] = 768
                exclude_sa = True
            if random_args['reduction_factor'] > 768:
                random_args['reduction_factor'] = 768
                exclude_pa = True
            if random_args['prefix_length'] < 1:
                random_args['prefix_length'] = 1
                exclude_prefix = True
            default_arg = default_mam_arg.copy()

        default_arg.update(random_args)
        logger.info(f"random_args {random_args}")
        logger.info(f"overall_args {default_arg}")
        if model_args.adapter_name == 'prefix':
            adapter_config = transformers.PrefixTuningConfig(**default_arg)

        if model_args.adapter_name == 'pfeiffer':
            adapter_config = transformers.AutoPEFTConfig(reduction_factor=default_arg['reduction_factor'], leave_out=default_arg['leave_out'], is_sa_alone=True)

        if model_args.adapter_name == 'mam':
            parallel_flag = default_arg['reduction_factor'] <= 768
            prefix_flag = default_arg['prefix_length'] >= 1
            config_flag_list = [prefix_flag, parallel_flag]
            config_list = [PrefixTuningConfig(prefix_length=int(default_arg['prefix_length']), bottleneck_size=800, leave_out=default_arg['leave_out']),
                           ParallelConfig(
                               reduction_factor=default_arg['reduction_factor'], leave_out=default_arg['leave_out']),
                           ]
            adapter_config = ConfigUnion(
                *[config_list[i] for i in range(len(config_list)) if config_flag_list[i]])

        if model_args.adapter_name == 'unipelt':
            parallel_flag = default_arg['reduction_factor'] <= 768
            prefix_flag = default_arg['prefix_length'] >= 1
            lora_flag = default_arg['r'] >= 1
            config_flag_list = [prefix_flag, parallel_flag, lora_flag]
            config_list = [PrefixTuningConfig(prefix_length=int(default_arg['prefix_length']), bottleneck_size=800, leave_out=default_arg['leave_out']),
                           ParallelConfig(
                               reduction_factor=default_arg['reduction_factor'], leave_out=default_arg['leave_out']),
                           LoRAConfig(r=int(default_arg['r'])),
                           ]
            adapter_config = ConfigUnion(
                *[config_list[i] for i in range(len(config_list)) if config_flag_list[i]])

        if model_args.adapter_name == 'sapa':
            if default_arg['reduction_serial'] > 768 and default_arg['reduction_factor'] <= 768:
                adapter_config = ParallelConfig(
                    reduction_factor=default_arg['reduction_factor'], leave_out=default_arg['leave_out'])
            elif default_arg['reduction_serial'] <= 768 and default_arg['reduction_factor'] > 768:
                adapter_config = AutoPEFTConfig(
                    reduction_factor=int(default_arg['reduction_serial']))
            else:
                adapter_config = transformers.AutoPEFTConfig(reduction_factor=int(
                    default_arg['reduction_serial']), leave_out=default_arg['leave_out'])
                model.add_adapter('sa', config=adapter_config)
                adapter_config = transformers.ParallelConfig(
                    reduction_factor=default_arg['reduction_factor'], leave_out=default_arg['leave_out'])
                model.add_adapter('pa', config=adapter_config)
                model.active_adapters = ac.Aggregate("sa", "pa")
        # adapter_config = AdapterConfig(adapter_residual_before_ln=True, is_parallel=False, ln_after=True, ln_before=True, mh_adapter=True, non_linearity='tanh', original_ln_after=True, original_ln_before=True, output_adapter=True, reduction_factor=32, residual_before_ln=True, scaling=2.0)
        # model.add_adapter(task_name, config=adapter_config)
        if model_args.adapter_name == 'sappa':
            if not exclude_pa and not exclude_sa and not exclude_prefix:
                adapter_config = ConfigUnion(
                    transformers.PrefixTuningConfig(prefix_length=int(
                        default_arg['prefix_length']), bottleneck_size=800, leave_out=default_arg['leave_out']),
                    transformers.AutoPEFTConfig(
                        reduction_factor=default_arg['reduction_serial'], reduction_factor_pa=default_arg['reduction_factor'], leave_out=default_arg['leave_out']),
                )
            if not exclude_pa and not exclude_sa and exclude_prefix:
                adapter_config = ConfigUnion(
                    transformers.AutoPEFTConfig(
                        reduction_factor=default_arg['reduction_serial'], reduction_factor_pa=default_arg['reduction_factor'], leave_out=default_arg['leave_out']),
                )
            if not exclude_pa and exclude_sa and not exclude_prefix:
                adapter_config = ConfigUnion(
                    transformers.PrefixTuningConfig(prefix_length=int(
                        default_arg['prefix_length']), bottleneck_size=800, leave_out=default_arg['leave_out']),
                    transformers.AutoPEFTConfig(
                        reduction_factor_pa=default_arg['reduction_factor'], leave_out=default_arg['leave_out'], is_pa_alone=True),
                )
            if not exclude_pa and exclude_sa and exclude_prefix:
                adapter_config = ConfigUnion(
                    transformers.AutoPEFTConfig(
                        reduction_factor_pa=default_arg['reduction_factor'], leave_out=default_arg['leave_out'], is_pa_alone=True),
                )
            if exclude_pa and not exclude_sa and not exclude_prefix:
                adapter_config = ConfigUnion(
                    transformers.PrefixTuningConfig(prefix_length=int(
                        default_arg['prefix_length']), bottleneck_size=800, leave_out=default_arg['leave_out']),
                    transformers.AutoPEFTConfig(
                        reduction_factor=default_arg['reduction_serial'], leave_out=default_arg['leave_out'], is_sa_alone=True),
                )
            if exclude_pa and not exclude_sa and exclude_prefix:
                adapter_config = ConfigUnion(
                    transformers.AutoPEFTConfig(
                        reduction_factor=default_arg['reduction_serial'], leave_out=default_arg['leave_out'], is_sa_alone=True),
                )
            if exclude_pa and exclude_sa and not exclude_prefix:
                adapter_config = ConfigUnion(
                    transformers.PrefixTuningConfig(prefix_length=int(
                        default_arg['prefix_length']), bottleneck_size=800, leave_out=default_arg['leave_out']),
                )

        if model_args.adapter_name == 'sapa':
            if default_arg['reduction_serial'] > 768 and default_arg['reduction_factor'] <= 768:
                model.add_adapter(task_name, config=adapter_config)
                model.train_adapter(task_name)
                model.set_active_adapters(task_name)
            elif default_arg['reduction_serial'] <= 768 and default_arg['reduction_factor'] > 768:
                model.add_adapter(task_name, config=adapter_config)
                model.train_adapter(task_name)
                model.set_active_adapters(task_name)
            else:
                model.train_adapter(ac.Aggregate("sa", "pa"))
        else:
            model.add_adapter(task_name, config=adapter_config)
            model.train_adapter(task_name)
            model.set_active_adapters(task_name)

    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )
    logger.info(
        f"model number of parameters after adapter{model.num_parameters()}")
    model_param_dict['w. heads & adapter'] = model.num_parameters()
    model_param_dict['heads'] = model_param_dict['w. heads'] - \
        model_param_dict['model']
    model_param_dict['adapter'] = model_param_dict['w. heads & adapter'] - \
        model_param_dict['w. heads']

    # make the output dir if output dir not exist
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, "model_param_dict.json"), "w", encoding='utf8') as f:
        json.dump(model_param_dict, f, indent=2, ensure_ascii=False)
    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(
                num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {
            k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {
            id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {
            id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding,
                           max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1)
                               for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name ==
                                    "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name ==
                                       "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(
                range(max_predict_samples))

    if data_args.resplit_dataset:
        print('original length of training dataset: ',
              len(raw_datasets["train"]))
        # print('original length of eval dataset: ', len(raw_datasets["validation"]))
        train_dataset, eval_dataset = split_datasets(raw_datasets["train"])
        print('length of training dataset after resplit: ', len(train_dataset))
        print('length of eval dataset after resplit: ', len(eval_dataset))
        predict_dataset = raw_datasets["validation_matched" if data_args.task_name ==
                                       "mnli" else "validation"]

    # temperal script for low fidelity experiments
    # n = len(raw_datasets["train"])
    # split_at = int(n * 0.01)
    # train_ds = raw_datasets["train"].shuffle()
    # train_dataset = train_ds.select(range(split_at))
    # print('length of training dataset after resplit: ', len(train_dataset))

    # Log a few random samples from the training set:
    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric(
            f"{ROOT_DIR}/glue_metrics.py", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(
            p.predictions, tuple) else p.predictions
        preds = np.squeeze(
            preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(
                    list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = 'accuracy'

    if data_args.task_name == 'cola':
        training_args.metric_for_best_model = 'eval_matthews_correlation'
    if data_args.task_name == 'stsb':
        training_args.metric_for_best_model = 'eval_spearmanr'
    training_args.greater_is_better = True
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=data_args.patience)]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
                    eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics(
                "eval", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict and data_args.resplit_dataset:
        logger.info("*** test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["validation_mismatched"])
            combined = {}

        for predict_dataset, task in zip(predict_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=predict_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
                    predict_dataset)
            )
            metrics["test_samples"] = min(
                max_eval_samples, len(predict_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("test", metrics)
            trainer.save_metrics(
                "test", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict and not data_args.resplit_dataset:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(
                predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(
                predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(
                training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path,
              "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    all_checkpoints = get_all_checkpoint(training_args.output_dir)
    for checkpoint in all_checkpoints:
        last_checkpoint = os.path.join(training_args.output_dir, checkpoint)
        shutil.rmtree(last_checkpoint, ignore_errors=True)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
