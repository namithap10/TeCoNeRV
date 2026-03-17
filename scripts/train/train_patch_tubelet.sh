#!/bin/bash

export OMP_NUM_THREADS=1
MASTER_PORT=$(($RANDOM % 500 + 29500))
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Train patch tubelet
torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_GPUS run_trainer.py  \
	--cfg cfgs/train_patch_tubelet.yaml --replace  \
	--csv_file k400_2023_train_cls400_50_480p.js --out_path checkpoints/patch_tubelet --name pre_finetune  \
	--frame_num 8 --input_size 480x640 --tubelet_size 160x320  \
	-b 32 -j 16 --opts  \
	train_dataset.args.cls_vid_num 400_25    model.args.tokenizer.args.patch_size 32    \
	model.args.hyponet.args.pe_dim 14    model.args.hyponet.args.hid_dim 14  \
	model.args.n_tokens 5_56_4_0   model.args.token_dims 196_252_196_0  \
	model.args.hyponet.args.strds_h 5_4_4_2    model.args.hyponet.args.strds_w 5_4_4_4  \
	optimizer.args.lr 0.0001     optimizer.lr_type step     max_epoch 150  eval_epoch 50   \
	--tag '' --instance_tag pre_finetune_320x160_patch

# Finetune patch tubelet
torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_GPUS run_trainer.py  \
	--cfg cfgs/train_patch_tubelet_finetune.yaml --replace  \
	--csv_file k400_2023_train_cls400_50_480p.js --out_path checkpoints/ --name patch_tubelet  \
	--frame_num 8 --input_size 480x640 --tubelet_size 160x320  \
	-b 32 -j 16 --opts  \
	train_dataset.args.cls_vid_num 400_25    model.args.tokenizer.args.patch_size 32    \
	model.args.hyponet.args.pe_dim 14    model.args.hyponet.args.hid_dim 14  \
	model.args.n_tokens 5_56_4_0   model.args.token_dims 196_252_196_0  \
	model.args.hyponet.args.strds_h 5_4_4_2    model.args.hyponet.args.strds_w 5_4_4_4  \
	optimizer.args.lr 0.0001     optimizer.lr_type step     max_epoch 50  eval_epoch 50   \
	finetune_model checkpoints/patch_tubelet/pre_finetune/pre_finetune_320x160_patch/epoch-last.pth \
	finetune_same_model true \
	--tag '' --instance_tag 320x160_finetuned_patch
