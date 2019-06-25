#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES='0'
python build_linkprediction_model.py \
    --model_name LinkNet \
    --data_dir /home/LAB/zhangzy/ProjectData/knowNet/ \
    --bert_model /home/LAB/zhangzy/ShareModels/uncased_L-12_H-768_A-12/ \
    --output_dir /home/LAB/zhangzy/ProjectModels/linkNets/ \
    --vocab_file /home/LAB/zhangzy/ProjectModels/linkNets/vocab.txt \
    --local_rank -1 \
    --do_eval \
    --tokenized \
    --train_batch_size 64 \
    --eval_batch_size 128 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \
    --eval_step_size 5000 \
    --do_lower_case \
    --max_seq_length 1500 \
    --overwrite \
    --do_train \
    --task_name q_and_p \
    #/home/LAB/zhangzy/ProjectModels/knowledgeSearcher/model1 \
    #/home/LAB/zhangzy/ShareModels/uncased_L-12_H-768_A-12 \
