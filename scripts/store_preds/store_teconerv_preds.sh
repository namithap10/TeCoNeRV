#!/bin/bash

# Without overlap
python run_trainer.py \
    --cfg cfgs/store_patch_preds.yaml --replace \
    --name store_teconerv_preds_uvg_beauty \
    --out_path output/store_logs/teconerv_preds_480p_160x320_uvg \
    --frame_num 8 --input_size 480x640 --tubelet_size 160x320 \
    -b 1 -j 8 \
    --opts \
    eval_model checkpoints/teconerv/320x160_pairs_teco/epoch-last.pth \
    save_path save/teconerv_preds_quant/teconerv_preds_480p_160x320_uvg \
    eval_same_model false eval_saver nerv_enc_full_res \
    save_weights_no_quant no \
    save_weights_quant yes \
    target_vid beauty

# With overlap - evaluate at 720p (quantization, 8-bit)
# Bosphore, UVG
python run_trainer.py \
    --cfg cfgs/store_patch_preds_overlap.yaml --replace \
    --name store_teconerv_preds_overlap_720p_uvg_bosphore \
    --out_path output/store_logs/teconerv_preds_overlap_720p_160x320_uvg \
    --frame_num 8 --input_size 720x1280 --tubelet_size 160x320 \
    -b 1 -j 8 \
    --opts \
    eval_model checkpoints/teconerv/320x160_pairs_teco/epoch-last.pth \
    save_path save/teconerv_preds_quant/teconerv_preds_overlap_720p_160x320_uvg \
    eval_same_model false eval_saver nerv_enc_full_res \
    save_weights_no_quant no \
    save_weights_quant yes \
    overlap_h 20 overlap_w 20 \
    target_vid bosphore

# Johnny, HEVC Class E
python run_trainer.py \
    --cfg cfgs/store_patch_preds_overlap_hevc.yaml --replace \
    --name store_teconerv_preds_overlap_720p_hevc_e_johnny \
    --out_path output/store_logs/teconerv_preds_overlap_720p_160x320_hevc_e \
    --frame_num 8 --input_size 720x1280 --tubelet_size 160x320 \
    -b 1 -j 8 \
    --opts \
    eval_model checkpoints/teconerv/320x160_pairs_teco/epoch-last.pth \
    save_path save/teconerv_preds_quant/teconerv_preds_overlap_720p_160x320_hevc_e \
    eval_same_model false eval_saver nerv_enc_full_res \
    save_weights_no_quant no \
    save_weights_quant yes \
    overlap_h 20 overlap_w 20 \
    target_vid johnny

# # With overlap - evaluate at 1080p (quantization, 8-bit)
python run_trainer.py \
    --cfg cfgs/store_patch_preds_overlap.yaml --replace \
    --name store_teconerv_preds_overlap_1080p_uvg_jockey \
    --out_path output/store_logs/teconerv_preds_overlap_1080p_240x320 \
    --frame_num 8 --input_size 1080x1920 --tubelet_size 240x320 \
    -b 1 -j 8 \
    --opts \
    eval_model checkpoints/teconerv/320x240_pairs_teco/epoch-last.pth \
    save_path save/teconerv_preds_quant/teconerv_preds_overlap_1080p_240x320_uvg \
    eval_same_model false eval_saver nerv_enc_full_res \
    save_weights_no_quant no \
    save_weights_quant yes \
    overlap_h 20 overlap_w 20 \
    target_vid jockey