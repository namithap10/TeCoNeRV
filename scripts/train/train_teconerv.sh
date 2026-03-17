#!/bin/bash

export OMP_NUM_THREADS=1
MASTER_PORT=$(($RANDOM % 500 + 29500))
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Finetune with temporal coherence regularization
# 320x160 size patch
torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_GPUS run_trainer.py  \
	--cfg cfgs/train_teconerv.yaml --replace  \
	--csv_file k400_2023_train_cls400_50_480p.js --out_path checkpoints/ --name teconerv  \
	--frame_num 8 --input_size 480x640 --tubelet_size 160x320  \
	-b 32 -j 16 --opts  \
	train_dataset.args.cls_vid_num 400_25    model.args.tokenizer.args.patch_size 32    \
	model.args.hyponet.args.pe_dim 14    model.args.hyponet.args.hid_dim 14  \
	model.args.n_tokens 5_56_4_0   model.args.token_dims 196_252_196_0  \
	model.args.hyponet.args.strds_h 5_4_4_2    model.args.hyponet.args.strds_w 5_4_4_4  \
	optimizer.args.lr 0.0001     optimizer.lr_type step     max_epoch 50  eval_epoch 50   \
	finetune_model checkpoints/patch_tubelet/pre_finetune/pre_finetune_320x160_patch/epoch-last.pth \
	finetune_same_model false \
     param_reg_lambda_l1 0.1  param_reg_lambda_l2 0.0  param_reg_mode mod  \
	--tag '' --instance_tag 320x160_pairs_teco

# 320x240 size patch
torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_GPUS run_trainer.py  \
	--cfg cfgs/train_teconerv.yaml --replace  \
	--csv_file k400_2023_train_cls400_50_480p.js --out_path checkpoints/ --name teconerv  \
	--frame_num 8 --input_size 480x640 --tubelet_size 240x320  \
	-b 32 -j 16 --opts  \
	train_dataset.args.cls_vid_num 400_25    model.args.tokenizer.args.patch_size 32    \
	model.args.hyponet.args.pe_dim 16    model.args.hyponet.args.hid_dim 20  \
	model.args.n_tokens 10_80_16_0   model.args.token_dims 200_240_240_0  \
	model.args.hyponet.args.strds_h 5_4_4_3    model.args.hyponet.args.strds_w 5_4_4_4  \
	optimizer.args.lr 0.0001     optimizer.lr_type step     max_epoch 50  eval_epoch 50   \
	finetune_model checkpoints/patch_tubelet/pre_finetune/pre_finetune_320x240_patch_train_480p/epoch-last.pth \
	finetune_same_model false \
     param_reg_lambda_l1 0.1  param_reg_lambda_l2 0.0  param_reg_mode mod  \
	--tag '' --instance_tag 320x240_pairs_teco

# 384x270 size patch
torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_GPUS run_trainer.py  \
	--cfg cfgs/train_teconerv.yaml --replace  \
	--csv_file k400_2023_train_cls400_50_480p.js --out_path checkpoints/ --name teconerv  \
	--frame_num 8 --input_size 480x640 --tubelet_size 270x384  \
	-b 32 -j 16 --opts  \
	train_dataset.args.cls_vid_num 400_25    model.args.tokenizer.args.patch_size 32    \
	model.args.hyponet.args.pe_dim 20    model.args.hyponet.args.hid_dim 20  \
	model.args.n_tokens 16_100_16_0   model.args.token_dims 180_240_180_0  \
	model.args.hyponet.args.strds_h 6_5_3_3    model.args.hyponet.args.strds_w 6_4_4_4  \
	optimizer.args.lr 0.0001     optimizer.lr_type step     max_epoch 50  eval_epoch 50   \
	finetune_model checkpoints/patch_tubelet/pre_finetune/pre_finetune_384x270_patch_train_480p/epoch-last.pth \
	finetune_same_model false \
     param_reg_lambda_l1 0.1  param_reg_lambda_l2 0.0  param_reg_mode mod  \
	--tag '' --instance_tag 384x270_pairs_teco
