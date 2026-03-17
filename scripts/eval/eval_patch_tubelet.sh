#!/bin/bash

# Inference at 480p resolution
python run_trainer.py  \
	--cfg cfgs/eval_patch_uvg.yaml --replace  \
	--name eval_uvg_patch --out_path output/eval_logs/patch_uvg/  \
	--frame_num 8 --input_size 480x640 --tubelet_size 160x320  \
	-b 1 -j 8 --opts  \
	eval_model checkpoints/patch_tubelet/320x160_finetuned_patch/epoch-last.pth  \
	eval_metrics_path checkpoints/patch_tubelet/320x160_finetuned_patch  \
	eval_same_model true eval_residuals true \
	quant_bit 4 encoding_type arithmetic
