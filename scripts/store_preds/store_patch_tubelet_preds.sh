#!/bin/bash

# Store patch tubelet predictions with quantization (default is 8-bit)
python run_trainer.py \
    --cfg cfgs/store_patch_preds.yaml --replace \
    --name store_patch_tubelet_preds_uvg_beauty \
    --out_path output/store_logs/patch_tubelet_preds_480p_160x320_uvg \
    --frame_num 8 --input_size 480x640 --tubelet_size 160x320 \
    -b 1 -j 8 \
    --opts \
    eval_model checkpoints/patch_tubelet/320x160_finetuned_patch/epoch-last.pth \
    save_path save/patch_tubelet_preds_quant/patch_tubelet_preds_480p_160x320_uvg \
    eval_same_model true \
    save_weights_no_quant no \
    save_weights_quant yes \
    target_vid beauty
