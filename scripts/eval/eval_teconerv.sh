#!/bin/bash

# Without overlap, evaluate the 320x160 patch-size model at 480p resolution inference on UVG
python run_trainer.py  \
	--cfg cfgs/eval_patch_uvg.yaml --replace  \
	--name eval_uvg_teconerv --out_path output/eval_logs/teconerv_uvg/  \
	--frame_num 8 --input_size 480x640 --tubelet_size 160x320 \
	-b 1 -j 8 --opts  \
	eval_model checkpoints/teconerv/320x160_pairs_teco/epoch-last.pth  \
	eval_metrics_path checkpoints/teconerv/320x160_pairs_teco  \
	eval_same_model false eval_saver nerv_enc_full_res \
	eval_residuals true \
	quant_bit 4 encoding_type arithmetic

# Evaluation on Kinetics-400
python run_trainer.py  \
	--cfg cfgs/eval_patch_k400_2023.yaml --replace  \
	--name eval_k400_teconerv --out_path output/eval_logs/teconerv_k400/  \
	--frame_num 8 --input_size 480x640 --tubelet_size 160x320 \
	-b 1 -j 8 --opts  \
	eval_model checkpoints/teconerv/320x160_pairs_teco/epoch-last.pth  \
	eval_metrics_path checkpoints/teconerv/320x160_pairs_teco  \
	eval_same_model false eval_saver nerv_enc_full_res \
	eval_residuals true \
	quant_bit 4 encoding_type arithmetic