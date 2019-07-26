#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='3'
maxLen=200
python /home/LAB/zhangzy/ProgrammingAlpha/libs/OpenNMT-py/translate.py \
                    -batch_size 4 \
                    -beam_size 30 \
                    -model /home/LAB/zhangzy/ProjectModels/answerNets/model_step_60000.pt \
                    -src /home/LAB/zhangzy/ProjectData/seq2seq/valid-src \
                    -output /home/LAB/zhangzy/ProjectData/predictions/"predict-${maxLen}.txt" \
                    -min_length 35 \
                    -max_length ${maxLen} \
                    -verbose \
                    -stepwise_penalty \
                    -coverage_penalty summary \
                    -beta 5 \
                    -length_penalty wu \
                    -alpha 0.9 \
                    -block_ngram_repeat 2 \
                    -n_best 1\
                    -ignore_when_blocking "." "</t>" "<t>" \
                    -report_bleu \
                    -report_rouge \
                    -share_vocab \
                    -dynamic_dict \
                    -gpu -1 \
                    -verbose \
                    #-tgt /home/LAB/zhangzy/ProjectData/seq2seq/valid-dst \


