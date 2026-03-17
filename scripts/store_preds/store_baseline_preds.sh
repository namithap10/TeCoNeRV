#!/bin/bash

# Store baseline predictions with quantization (default is 8-bit)
python run_trainer.py \
    --cfg cfgs/store_baseline_preds.yaml --replace \
    --name store_baseline_preds_uvg_beauty \
    --out_path output/store_logs/baseline_preds_480p_uvg \
    --frame_num 8 --input_size 480x640 \
    -b 1 -j 8 \
    --opts \
    eval_model checkpoints/nervenc/480p_finetuned_baseline/epoch-last.pth \
    save_path save/baseline_preds_quant/baseline_preds_480p_uvg \
    eval_same_model true \
    save_weights_no_quant no \
    save_weights_quant yes \
    target_vid beauty