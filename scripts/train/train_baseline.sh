#!/bin/bash

export OMP_NUM_THREADS=1
MASTER_PORT=$(($RANDOM % 500 + 29500))
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Train 480p baseline
torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_GPUS run_trainer.py \
--cfg cfgs/train_baseline.yaml -p 100 --replace \
--csv_file k400_2023_train_cls400_50_480p.js --out_path checkpoints/nervenc --name pre_finetune \
--frame_num 8 --input_size 480x640 \
-b 8 -j 16 --opts \
train_dataset.args.cls_vid_num 400_25    train_dataset.args.rand_augment 1_2_5      model.args.tokenizer.args.patch_size 32  \
model.args.hyponet.args.strds_h 5_4_4_3_2    model.args.hyponet.args.strds_w 5_4_4_4_2    model.args.hyponet.args.ks 1_3 \
model.args.hyponet.args.hid_dim 32   model.args.hyponet.args.pe_dim 32 \
model.args.n_tokens 32_256_32_24_0    model.args.token_dims 200_288_288_288_0 \
train_dataset.args.clips_per_video 1 \
max_epoch 150   eval_epoch 50 \
--tag '' --instance_tag pre_finetune_480p_baseline


# Finetune 480p baseline
torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_GPUS run_trainer.py \
--cfg cfgs/train_baseline_finetune.yaml -p 100 --replace \
--csv_file k400_2023_train_cls400_50_480p.js --out_path checkpoints/ --name nervenc \
--frame_num 8 --input_size 480x640 \
-b 8 -j 16 --opts \
train_dataset.args.cls_vid_num 400_25    train_dataset.args.rand_augment 1_2_5      model.args.tokenizer.args.patch_size 32  \
model.args.hyponet.args.strds_h 5_4_4_3_2    model.args.hyponet.args.strds_w 5_4_4_4_2    model.args.hyponet.args.ks 1_3 \
model.args.hyponet.args.hid_dim 32   model.args.hyponet.args.pe_dim 32 \
model.args.n_tokens 32_256_32_24_0    model.args.token_dims 200_288_288_288_0 \
train_dataset.args.clips_per_video 1 \
max_epoch 50   eval_epoch 50 \
finetune_model checkpoints/nervenc/pre_finetune/pre_finetune_480p_baseline/epoch-last.pth \
finetune_same_model true \
--tag '' --instance_tag 480p_finetuned_baseline
