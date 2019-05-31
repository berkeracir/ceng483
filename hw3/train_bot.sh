#!/bin/bash

mkdir -p ./logs

# for layers in {1,2,4} ; do
#     for kernel_size in {3,5} ; do
#         for channels in {2,4,8} ; do
#             #for learning_rate in {0.1,0.001,0.0001} ; do
#             learning_rate=0.01
#                 echo "Training: $layers $kernel_size $channels $learning_rate"
#                 python train.py $layers $kernel_size $channels $learning_rate > ./logs/log_"$layers"_"$kernel_size"_"$channels"_"$learning_rate".txt
#             #done
#         done
#     done
# done

# echo 2 5 8 0.1
# python train.py 2 5 8 0.1 > ./logs/log_2_5_8_0.1_tanh.txt

# echo 2 3 8 0.1
# python train.py 2 3 8 0.1 > ./logs/log_2_3_8_0.1_tanh.txt

# echo 4 5 8 0.1
# python train.py 4 5 8 0.1 > ./logs/log_4_5_8_0.1_tanh.txt

# echo 2 5 4 0.1
# python train.py 2 5 4 0.1 > ./logs/log_2_5_4_0.1_tanh.txt

layers=16
for kernel_size in {3,5} ; do
    for channels in {4,8} ; do
        learning_rate=0.1
        echo "Training: $layers $kernel_size $channels $learning_rate"
        python train.py $layers $kernel_size $channels $learning_rate > ./logs/log_"$layers"_"$kernel_size"_"$channels"_"$learning_rate".txt

    done
done