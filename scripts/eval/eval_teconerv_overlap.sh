#!/bin/bash

# With overlap enabled, evaluate the 320x160 patch-size model at 720p resolution inference
python run_trainer.py  \
	--cfg cfgs/eval_patch_overlap_uvg.yaml --replace  \
	--name eval_uvg_teconerv_overlap_720p --out_path output/eval_logs/teconerv_uvg_overlap_720p/  \
	--frame_num 8 --input_size 720x1280 --tubelet_size 160x320 \
	-b 1 -j 8 --opts  \
	eval_model checkpoints/teconerv/320x160_pairs_teco/epoch-last.pth  \
	eval_metrics_path checkpoints/teconerv/320x160_pairs_teco  \
	eval_same_model false eval_saver nerv_enc_full_res \
	eval_residuals true \
	eval_csv_prefix overlap_720p \
	quant_bit 4 encoding_type arithmetic \
	overlap_h 20 overlap_w 20 blend_overlap false \
	chunk_pred_batch_size None


python run_trainer.py  \
	--cfg cfgs/eval_patch_overlap_uvg.yaml --replace  \
	--name eval_uvg_teconerv_overlap_1080p --out_path output/eval_logs/teconerv_uvg_overlap_1080p/  \
	--frame_num 8 --input_size 1080x1920 --tubelet_size 240x320 \
	-b 1 -j 8 --opts  \
	eval_model checkpoints/teconerv/320x240_pairs_teco/epoch-last.pth  \
	eval_metrics_path checkpoints/teconerv/320x240_pairs_teco  \
	eval_same_model false eval_saver nerv_enc_full_res \
	eval_residuals true \
	eval_csv_prefix overlap_1080p \
	quant_bit 4 encoding_type arithmetic \
	overlap_h 20 overlap_w 20 blend_overlap false \
	chunk_pred_batch_size None