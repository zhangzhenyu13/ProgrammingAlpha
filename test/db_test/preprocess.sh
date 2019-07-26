#!/usr/bin/env bash
prefix=/home/LAB/zhangzy/ProjectData/seq2seq/
python ~/ProgrammingAlpha/libs/OpenNMT-py/preprocess.py -train_src $prefix/train-src \
                     -train_tgt $prefix/train-dst \
                     -valid_src $prefix/valid-src \
                     -valid_tgt $prefix/valid-dst \
                     -save_data /home/LAB/zhangzy/ProjectData/openNMT/answerNetData \
                     -src_seq_length 5000 \
                     -tgt_seq_length 5000 \
                     -src_seq_length_trunc 2000 \
                     -tgt_seq_length_trunc 2000 \
                     -dynamic_dict \
                     -share_vocab \
                     -shard_size 20000 \
		             -filter_valid \
		             -src_vocab /home/LAB/zhangzy/ProjectModels/answerNets/vocab.txt \
                     -tgt_vocab /home/LAB/zhangzy/ProjectModels/answerNets/vocab.txt \
