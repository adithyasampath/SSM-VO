#!/bin/bash
DATA_PATH="./dataset"
LOG_PATH="./train_logs_lstm"

# specify the gpu
export CUDA_VISIBLE_DEVICES="4"

python train.py \
--data_path $DATA_PATH \
--load_weights_folder "./Pretrained_MonoDepth2" \
--ppnet ./networks/ppnet_weights/lstm/ppnet_lstm_35.pth \
--learning_rate 1e-7 \
--conf_thres 0.825 \
--batch_size 32 \
--num_workers 12 \
--log_dir $LOG_PATH \
--model_name finetuned_gt \
--ppnet_model lstm \
--png
# --update_once   # if open, you can use validate loss to choose model.
