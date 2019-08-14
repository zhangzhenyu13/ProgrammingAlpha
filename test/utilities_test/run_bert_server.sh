bert-serving-start \
        -num_worker=4 \
        -model_dir /home/LAB/zhangzy/ShareModels/uncased_L-12_H-768_A-12/ \
        -pooling_layer -2 \
        -max_seq_len 502 \
        -num_worker=1 \
        -http_max_connect 10 \
        -pooling_strategy REDUCE_MEAN \
        -max_batch_size 32
