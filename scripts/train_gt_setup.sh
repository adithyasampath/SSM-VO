#!/bin/bash
DATA_PATH="YOUR_DATA_PATH"
LOG_PATH="YOUR_LOG_PATH/DATE"

# specify the gpu
export CUDA_VISIBLE_DEVICES=$1

python train.py \
--data_path $DATA_PATH \
--load_weights_folder "PRETRAINED_MONODEPTH2" \
--ppnet ./networks/ppnet_weights/ppnet_slstm_best_model.pth \
--learning_rate 1e-7 \
--conf_thres 0.825 \
--num_workers 6 \
--log_dir $LOG_PATH \
--model_name finetuned_gt \
--ppnet_model slstm \
--png
# --update_once   # if open, you can use validate loss to choose model.
