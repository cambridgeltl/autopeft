#!/bin/bash

adapt_name="finetune"

for TASK_NAME in rte stsb cola mrpc sst2 qnli qqp mnli;
do
  for seed in 40 41 42 43 44;
  do
    python run_glue.py \
      --model_name_or_path bert-base-uncased \
      --task_name $TASK_NAME \
      --do_train \
      --do_eval \
      --max_seq_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate 2e-5 \
      --seed $seed\
      --num_train_epochs 20 \
      --logging_strategy "epoch"\
      --save_strategy "epoch"\
      --logging_steps 400\
      --save_steps 400\
      --overwrite_output_dir\
      --output_dir "./output/$TASK_NAME/bert-base/finalbaseline/$adapt_name/$seed/"
  done
done
