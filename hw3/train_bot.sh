#!/bin/bash

mkdir -p ./logs

for layers in {1,2,4} ; do
    for kernel_size in {3,5} ; do
        for channels in {2,4,8} ; do
            #for learning_rate in {0.1,0.001,0.0001} ; do
            learning_rate=0.01
                echo "Training: $layers $kernel_size $channels $learning_rate"
                python train.py $layers $kernel_size $channels $learning_rate > ./logs/log_"$layers"_"$kernel_size"_"$channels"_"$learning_rate".txt
            #done
        done
    done
done