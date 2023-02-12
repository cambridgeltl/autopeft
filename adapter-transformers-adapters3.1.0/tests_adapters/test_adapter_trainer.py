import os
import unittest
from tempfile import TemporaryDirectory

import torch

from transformers import (
    AutoAdapterModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    GlueDataset,
    GlueDataTrainingArguments,
    Trainer,
    TrainingArguments,
)
from transformers.adapters.composition import Fuse, Stack
from transformers.adapters.trainer import AdapterTrainer, logger
from transformers.testing_utils import slow


class TestAdapterTrainer(unittest.TestCase):
    def test_resume_training(self):

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model.add_adapter("adapter")
        model.add_adapter("additional_adapter")
        model.set_active_adapters("adapter")
        model.train_adapter("adapter")

        training_args = TrainingArguments(
            output_dir="./output",
            do_train=True,
            learning_rate=0.1,
            logging_steps=1,
            max_steps=1,
            save_steps=1,
            remove_unused_columns=False,
        )
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()
        # create second model that should resume the training of the first
        model_resume = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model_resume.add_adapter("adapter")
        model_resume.add_adapter("additional_adapter")
        model_resume.set_active_adapters("adapter")
        model_resume.train_adapter("adapter")
        trainer_resume = AdapterTrainer(
            model=model_resume,
            args=TrainingArguments(do_train=True, max_steps=1, output_dir="./output"),
            train_dataset=train_dataset,
        )
        trainer_resume.train(resume_from_checkpoint=True)

        self.assertEqual(model.config.adapters.adapters, model_resume.config.adapters.adapters)

        for ((k1, v1), (k2, v2)) in zip(trainer.model.state_dict().items(), trainer_resume.model.state_dict().items()):
            self.assertEqual(k1, k2)
            if "adapter" in k1:
                self.assertTrue(torch.equal(v1, v2), k1)

    def test_resume_training_with_fusion(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model.add_adapter("adapter")
        model.add_adapter("additional_adapter")
        model.add_adapter_fusion(Fuse("adapter", "additional_adapter"))
        model.set_active_adapters(Fuse("adapter", "additional_adapter"))
        model.train_fusion(Fuse("adapter", "additional_adapter"))

        training_args = TrainingArguments(
            output_dir="./output",
            do_train=True,
            learning_rate=0.1,
            logging_steps=1,
            max_steps=1,
            save_steps=1,
            remove_unused_columns=False,
        )
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()
        model_resume = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model_resume.add_adapter("adapter")
        model_resume.add_adapter("additional_adapter")
        model_resume.add_adapter_fusion(Fuse("adapter", "additional_adapter"))
        model_resume.set_active_adapters(Fuse("adapter", "additional_adapter"))
        model_resume.train_fusion(Fuse("adapter", "additional_adapter"))
        trainer_resume = AdapterTrainer(
            model=model_resume,
            args=TrainingArguments(do_train=True, max_steps=1, output_dir="./output"),
            train_dataset=train_dataset,
        )
        trainer_resume.train(resume_from_checkpoint=True)

        self.assertEqual(model.config.adapters.adapters, model_resume.config.adapters.adapters)

        for ((k1, v1), (k2, v2)) in zip(
            trainer.model.to("cpu").state_dict().items(), trainer_resume.model.to("cpu").state_dict().items()
        ):
            self.assertEqual(k1, k2)
            if "adapter" in k1:
                self.assertTrue(torch.equal(v1, v2), k1)

    def test_auto_set_save_adapters(self):
        model = BertForSequenceClassification(
            BertConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=37,
            )
        )
        model.add_adapter("adapter1")
        model.add_adapter("adapter2")
        model.add_adapter_fusion(Fuse("adapter1", "adapter2"))
        model.train_adapter_fusion(Fuse("adapter1", "adapter2"))

        training_args = TrainingArguments(
            output_dir="./output",
        )
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
        )
        self.assertTrue(trainer.train_adapter_fusion)

    @slow
    def test_training_load_best_model_at_end_full_model(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model.add_adapter("adapter")
        model.train_adapter("adapter")

        training_args = TrainingArguments(
            output_dir="./output",
            do_train=True,
            learning_rate=0.001,
            max_steps=1,
            save_steps=1,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=2,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()
        self.assertIsNotNone(trainer.model.active_adapters)

    def test_training_load_best_model_at_end_adapter(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model.add_adapter("adapter")
        model.train_adapter("adapter")

        training_args = TrainingArguments(
            output_dir="./output",
            do_train=True,
            learning_rate=0.001,
            max_steps=1,
            save_steps=1,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=2,
        )
        trainer = AdapterTrainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
        )
        with self.assertLogs(logger) as cm:
            trainer.train()
            self.assertTrue(any("Loading best adapter(s) from" in line for line in cm.output))
        self.assertEqual(Stack("adapter"), trainer.model.active_adapters)

    def test_training_load_best_model_at_end_fusion(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model.add_adapter("fuse_adapter_1")
        model.add_adapter("fuse_adapter_2")
        model.add_adapter_fusion(Fuse("fuse_adapter_1", "fuse_adapter_2"))
        model.train_adapter_fusion(Fuse("fuse_adapter_1", "fuse_adapter_2"))

        training_args = TrainingArguments(
            output_dir="./output",
            do_train=True,
            learning_rate=0.001,
            max_steps=1,
            save_steps=1,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=2,
        )
        trainer = AdapterTrainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
        )
        with self.assertLogs(logger) as cm:
            trainer.train()
            self.assertTrue(any("Loading best adapter fusion(s) from" in line for line in cm.output))
        self.assertEqual(Fuse("fuse_adapter_1", "fuse_adapter_2"), trainer.model.active_adapters)

    def test_reloading_prediction_head(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")

        model = AutoAdapterModel.from_pretrained("bert-base-uncased")

        model.add_classification_head("adapter", num_labels=3)
        model.add_classification_head("dummy", num_labels=2)

        # add the adapters to be fused
        model.add_adapter("adapter")
        model.add_adapter("additional_adapter")

        # setup fusion
        adapter_setup = Fuse("adapter", "additional_adapter")
        model.add_adapter_fusion(adapter_setup)
        model.train_adapter_fusion(adapter_setup)
        model.set_active_adapters(adapter_setup)
        self.assertEqual(adapter_setup, model.active_adapters)
        self.assertEqual("dummy", model.active_head)
        with TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir,
                do_train=True,
                learning_rate=0.1,
                logging_steps=1,
                max_steps=1,
                save_steps=1,
                remove_unused_columns=False,
            )
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )

            trainer.train()
            # create second model that should resume the training of the first
            model_resume = AutoAdapterModel.from_pretrained("bert-base-uncased")

            model_resume.add_classification_head("adapter", num_labels=3)
            model_resume.add_classification_head("dummy", num_labels=2)
            model_resume.add_adapter("adapter")
            model_resume.add_adapter("additional_adapter")
            # setup fusion
            adapter_setup = Fuse("adapter", "additional_adapter")
            model_resume.add_adapter_fusion(adapter_setup)
            model_resume.train_adapter_fusion(adapter_setup)
            model_resume.set_active_adapters(adapter_setup)
            trainer_resume = AdapterTrainer(
                model=model_resume,
                args=TrainingArguments(do_train=True, max_steps=1, output_dir=tempdir),
                train_dataset=train_dataset,
            )
            trainer_resume.train(resume_from_checkpoint=True)

            self.assertEqual("dummy", model.active_head)
            self.assertEqual(model.config.adapters.adapters, model_resume.config.adapters.adapters)

            for ((k1, v1), (k2, v2)) in zip(
                trainer.model.to("cpu").state_dict().items(), trainer_resume.model.to("cpu").state_dict().items()
            ):
                self.assertEqual(k1, k2)
                if "adapter" in k1 or "dummy" in k1:
                    self.assertTrue(torch.equal(v1, v2), k1)

    def test_general(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")

        model = AutoAdapterModel.from_pretrained("bert-base-uncased")

        model.add_classification_head("task", num_labels=3)

        # add the adapters to be fused
        model.add_adapter("task")
        model.add_adapter("additional_adapter")

        model.train_adapter("task")
        self.assertEqual("task", model.active_head)
        self.assertEqual(Stack("task"), model.active_adapters)
        with TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir,
                do_train=True,
                learning_rate=0.1,
                logging_steps=1,
                max_steps=1,
                save_steps=1,
                remove_unused_columns=False,
            )
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )

            trainer.train()

            # Check that adapters are actually saved but the full model is not
            files_dir_checkpoint = [file_or_dir for file_or_dir in os.listdir(os.path.join(tempdir, "checkpoint-1"))]
            self.assertTrue("task" in files_dir_checkpoint)
            self.assertTrue("additional_adapter" in files_dir_checkpoint)
            # Check that full model weights are not stored
            self.assertFalse("pytorch_model.bin" in files_dir_checkpoint)

            # this should always be false in the adapter trainer
            self.assertFalse(trainer.args.remove_unused_columns)
            self.assertEqual("task", model.active_head)
            self.assertEqual(Stack("task"), model.active_adapters)


if __name__ == "__main__":
    unittest.main()
