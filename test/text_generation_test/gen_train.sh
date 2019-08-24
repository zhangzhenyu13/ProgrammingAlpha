#!/usr/bin/env bash
encoder_lm=$1

export CUDA_VISIBLE_DEVICES='0,1,2,3'
python /home/LAB/zhangzy/ProgrammingAlpha/test/text_generation_test/train.py \
                   -data /home/LAB/zhangzy/ProjectData/openNMT/answer_data/${encoder_lm}Gen/data \
                   -save_model /home/LAB/zhangzy/ProjectModels/answerNets/${encoder_lm}Gen/model \
		   -model_dtype fp32 \
                   -layers 4 \
                   -rnn_size 768 \
                   -word_vec_size 768 \
                   -transformer_ff 3072 \
                   -max_grad_norm 0 \
                   -optim adam \
                   -pretrained_encoder  ${encoder_lm} \
                   -encoder_type transformer \
                   -decoder_type transformer \
                   -input_feed 0 \
                   -position_encoding \
                   -dropout 1e-1 \
                   -param_init 0 \
                   -warmup_steps 8000 \
                   -learning_rate 1e-5 \
                   -decay_method noam \
                   -label_smoothing 0.1 \
                   -adam_beta2 0.998 \
                   -batch_size 16 \
                   -valid_batch_size 16 \
                   -batch_type sents \
                   -normalization sents \
                   -max_generator_batches 2 \
                   -train_steps 200000 \
                   -valid_steps 10000 \
                   -save_checkpoint_steps 10000 \
                   -keep_checkpoint 5 \
                   -tensorboard \
                   -report_every 50 \
                   -accum_count 4 \
                   -copy_attn \
                   -param_init_glorot \
                   -world_size 1 \
                   -gpu_ranks 0 \
		   -train_from /home/LAB/zhangzy/ProjectModels/seq2seq_lm/${encoder_lm}/model_step_70000.pt
                   #-tensorboard_log_dir trainCopy.log \
                   #-train_from

