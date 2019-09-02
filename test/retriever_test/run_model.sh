#!/usr/bin/env bash
python build_linkprediction_model.py \
    --encoder bert \
    --data_dir ~/ProjectData/knowNet/ \
    --save_dir ~/ProjectModels/knowNets/bertEnc/ \
    --gradient_accumulation_steps 8 \
    --train_batch_size 16 \
    --eval_step_size 5000 \
    --eval_batch_size 16  \
    --max_steps 100000 \
    --warmup_steps 8000 \
    --train_verbose 200 \
    --train_load_size 1000000 \
    --eval_load_size 10000
