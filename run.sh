#!/bin/bash


for task in mrpc;
do
  python3 run_one_replicate.py \
      --overwrite \
      -t $task \
      -mi 200 \
      -an sappa \
      -ni 100 \
      -mp bert-base-uncased
done
