#!/bin/bash

# Inference at 480p resolution
python run_trainer.py  \
	--cfg cfgs/eval_baseline_uvg.yaml --replace  \
	--name eval_uvg_baseline_480p --out_path output/eval_logs/baseline_uvg_480p/  \
	--frame_num 8 --input_size 480x640   \
	-b 1 -j 8 --opts  \
	eval_model checkpoints/nervenc/480p_finetuned_baseline/epoch-last.pth  \
	eval_metrics_path checkpoints/nervenc/480p_finetuned_baseline  \
	eval_same_model true eval_residuals true \
	quant_bit 4  encoding_type arithmetic


# Inference at 720p resolution
python run_trainer.py  \
	--cfg cfgs/eval_baseline_uvg.yaml --replace  \
	--name eval_uvg_baseline_720p --out_path output/eval_logs/baseline_uvg_720p/  \
	--frame_num 8 --input_size 720x1280   \
	-b 1 -j 8 --opts  \
	eval_model checkpoints/nervenc/720p_finetuned_baseline/epoch-last.pth  \
	eval_metrics_path checkpoints/nervenc/720p_finetuned_baseline  \
	eval_same_model true eval_residuals true \
	quant_bit 4  encoding_type arithmetic