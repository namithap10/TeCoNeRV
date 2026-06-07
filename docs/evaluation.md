# Evaluation

- [Reproducing results from the paper](#reproducing-results-from-the-paper)
- [Overlapped inference](#overlapped-inference-for-patch-based-models)
- [Saving hyponet predictions and residuals](#saving-hyponet-predictions-and-residuals)
- [Visualization](#visualization)

This document covers primary evaluation for all three model families (`nervenc`, `patch_tubelet`, `teconerv`) and overlapped inference for patch-based models. We also provide instructions for saving weight predictions as bitstream to disk and performing visualization.

## Reproducing results from the paper

To reproduce metrics reported in the paper for UVG, HEVC, MCL-JCV, or the Kinetics-400 validation subset, set `eval_residuals: true`. This dispatches to `evaluate_x_dict_residuals_epoch()`, which reconstructs each video in full as an ordered sequence of consecutive `frame_num`-frame clips. We evaluate three transmission modes:

- **direct**: transmit each clip's `x_dict` parameters directly
- **from_first**: transmit residuals against the first clip
- **from_prev**: transmit residuals against the previous clip

The direct mode corresponds to the per-clip unique-parameter bitstream used by the baseline. The `from_first` and `from_prev` modes are the residual encoding schemes proposed in our paper for patch-based models. Bitrate and throughput are estimated by quantizing and entropy coding the `x_dict` tensors. Residual bitstreams are not written to disk during metric evaluation; for storing these artifacts, see [Saving hyponet predictions and residuals](#saving-hyponet-predictions-and-residuals).

The trainer used for residual evaluation depends on the model family:

- baseline: [nerv_enc_trainer.py](../trainers/nerv_enc_trainer.py)
- patch-tubelet and teconerv: [nerv_enc_trainer_full_res.py](../trainers/nerv_enc_trainer_full_res.py)
- overlap evaluation: [nerv_enc_trainer_full_res_eval_overlap.py](../trainers/nerv_enc_trainer_full_res_eval_overlap.py)

For the baseline (`nervenc`), the numbers reported in Table 1 correspond to the **direct** transmission mode for fair comparison, as residual encoding is our contribution for patch-based models. Table 3 includes baseline results under both direct and from_prev transmission as an ablation.

Results in the main paper (excluding the quantization ablation) are reported at `quant_bit 4` with `encoding_type arithmetic`. Both are passed as `--opts` overrides, as in the reference scripts. Multiple quantization levels can be evaluated by passing an underscore-separated list, e.g. `quant_bit 8_7_6_5_4`, which produces one set of metrics per level. Using a different lossless encoding type (e.g. `huffman`) will produce slightly different bpp estimates.

Note: The periodic `evaluate_epoch()` loop (which writes `eval_metrics.csv`) used during training is **not** the evaluation protocol reported in the paper. It samples clips from `test_dataset` and is useful as a sanity check during training, but it does not produce the sequential video-by-video metrics reported.

### Reference scripts and configs

The shell scripts under `scripts/eval/` are the reference entry points for the evaluation settings reported in the paper:

```bash
bash scripts/eval/eval_baseline.sh
bash scripts/eval/eval_patch_tubelet.sh
bash scripts/eval/eval_teconerv.sh
```

Reference configs:

- `cfgs/eval_baseline_uvg.yaml`
- `cfgs/eval_patch_uvg.yaml`
- `cfgs/eval_patch_overlap_uvg.yaml`
- `cfgs/eval_baseline_k400_2023.yaml`
- `cfgs/eval_patch_k400_2023.yaml`

These scripts evaluate UVG. For Kinetics validation, use the same `run_trainer.py` launcher with `cfgs/eval_baseline_k400_2023.yaml` or `cfgs/eval_patch_k400_2023.yaml`. For HEVC and MCL-JCV, adapt the UVG configs by changing `csv_paths` to the corresponding metadata files after extracting frames (see [datasets.md](datasets.md)).

### Dataset and loader selection

The repository uses two input formats during evaluation:

- Kinetics-400 validation is read from MP4 files via the inference dataset classes in [vidrec_dataset.py](../datasets/vidrec_dataset.py) and [vidrec_dataset_patches.py](../datasets/vidrec_dataset_patches.py).
- UVG, HEVC, and MCL-JCV are read from RGB frame folders, as described in [datasets.md](datasets.md).

In both cases, the residual evaluation path operates on ordered per-video datasets and reconstructs all consecutive clips in sequence. At config level:

- `test_dataset` - validation loader used by the standard `evaluate_epoch()` loop
- `eval_residuals_dataset` - used by baseline residual evaluation
- `eval_full_res_dataset` - used by full-resolution patch-based residual evaluation

### `eval_same_model` and `eval_saver`

For `nervenc` and `patch_tubelet`, evaluation uses the checkpoint model definition directly:

- `eval_same_model: true`
- `eval_saver: null` (no override)

For `teconerv`, the checkpoint was trained with the paired model `nerv_enc_full_res_pairs`, but evaluation runs inference one clip at a time using `nerv_enc_full_res`. The TeCoNeRV evaluation scripts therefore set:

- `eval_same_model: false`
- `eval_saver: nerv_enc_full_res`

This override is required whenever evaluating TeCoNeRV checkpoints, including on datasets other than UVG. See `scripts/eval/eval_teconerv.sh` for the reference.

### Inference resolution

The target inference resolution is set by `--input_size`. Videos are resized on the shorter side and center-cropped to `input_size` using the transforms in [vidrec_dataset.py](../datasets/vidrec_dataset.py), with antialiasing during resize.

For patch-based models:

- `input_size` - full reconstruction resolution
- `tubelet_size` - patch size used for tiling the full frame

### Throughput and timing

Evaluation runs from the paper were measured with `-j 16` workers for all models under the following conditions:

- 480p and 720p: single NVIDIA RTX A5000 GPU
- 1080p: single NVIDIA RTX A6000

Reported throughput may vary with GPU, CPU allocation, and machine load.

Encoding and decoding FPS are computed as follows.

For the baseline ([nerv_enc_trainer.py](../trainers/nerv_enc_trainer.py)):

- encode time = hypernetwork forward pass to predict hyponet `x_dict` + quantization and entropy coding
- decode time = entropy decoding and dequantization + hyponet reconstruction

For patch-based models ([nerv_enc_trainer_full_res.py](../trainers/nerv_enc_trainer_full_res.py), [nerv_enc_trainer_full_res_eval_overlap.py](../trainers/nerv_enc_trainer_full_res_eval_overlap.py)):

- encode time = patch processing to predict hyponet `x_dict` for all patches + quantization and entropy coding
- decode time = entropy decoding and dequantization + hyponet reconstruction + patch tiling

Videos are processed sequentially. Parallelizing across multiple GPUs is possible but `from_prev` evaluation requires explicit keyframe boundary handling and metric aggregation.

## Overlapped Inference for Patch-Based Models

Overlapped evaluation is supported for `patch_tubelet` and `teconerv` via [nerv_enc_trainer_full_res_eval_overlap.py](../trainers/nerv_enc_trainer_full_res_eval_overlap.py). It enables inference at any target resolution regardless of whether the patch size divides the resolution evenly, and mitigates boundary artifacts at patch edges.

Reference script: `scripts/eval/eval_teconerv_overlap.sh`
Reference config: `cfgs/eval_patch_overlap_uvg.yaml`

For patch-tubelet overlap evaluation, use the same path with `eval_same_model: true`.

The overlap grid is constructed in [vidrec_dataset_patches.py](../datasets/vidrec_dataset_patches.py). The effective stride per dimension is `tubelet_size - overlap`, with the last patch snapped to the image boundary. Increasing overlap increases the number of tubelets processed per clip.

Overlap-specific options:

- `overlap_h`, `overlap_w` - overlap in pixels along each spatial dimension
- `blend_overlap` - reconstruction mode for overlapping regions:
  - `false`: each output pixel is assigned to exactly one patch by cropping the overlap region
  - `true`: overlap regions are blended with spatial weights; this results in lower decoding speed
- `chunk_pred_batch_size` - number of patch-tubelets processed together; reduce to lower GPU memory usage at some throughput cost

The released overlap script uses `input_size 720x1280`, `tubelet_size 160x320`, `overlap_h 20`, `overlap_w 20`, and `blend_overlap false`.

## Saving Hyponet Predictions and Residuals

To store hyponet weight predictions to disk rather than only compute metrics, use the weight-saver trainers:

- [nerv_enc_trainer_weight_saver.py](../trainers/nerv_enc_trainer_weight_saver.py) — baseline
- [nerv_enc_trainer_full_res_weight_saver.py](../trainers/nerv_enc_trainer_full_res_weight_saver.py) — patch-based models
- [nerv_enc_trainer_full_res_weight_saver_overlap.py](../trainers/nerv_enc_trainer_full_res_weight_saver_overlap.py) — patch-based models with overlap

These support saving direct parameters for the baseline and `from_prev` residuals for patch-based models, and optionally, storage after lossy quantization. Lossless entropy coding is not implemented and can be added if needed. Reconstructions for visualization use 8-bit quantization by default.

Reference scripts:

- `scripts/store_preds/store_baseline_preds.sh`
- `scripts/store_preds/store_patch_tubelet_preds.sh`
- `scripts/store_preds/store_teconerv_preds.sh`

Reference configs:

- `cfgs/store_baseline_preds.yaml`
- `cfgs/store_patch_preds.yaml`
- `cfgs/store_patch_preds_overlap.yaml`

The target dataset is set via `csv_paths` in the YAML config; the specific sequence to store is selected through `target_vid`. Weights can be saved with or without quantization via `save_weights_quant` and `save_weights_no_quant`.

The same `eval_same_model` and `eval_saver` overrides from metric evaluation apply here. For TeCoNeRV:

- `eval_same_model: false`
- `eval_saver: nerv_enc_full_res`

These configs and scripts are reference starting points and can be adapted for custom export formats, quantization settings, and dataset splits.

## Visualization

Saved weight or residual dumps can be reconstructed into RGB frames for qualitative inspection using:

- [scripts/visualize/reconstruct_from_saved_weights.py](../scripts/visualize/reconstruct_from_saved_weights.py)

This script reconstructs clips from the outputs written by the weight-saver trainers described in [Saving hyponet predictions and residuals](#saving-hyponet-predictions-and-residuals). Reconstruction mode is set via `--mode`:

- `baseline` - direct clip-wise reconstruction from baseline dumps under `direct/<video>/`
- `patch` - non-overlap patch-based reconstruction from `from_prev/<video>/tubelet_*/`
- `patch_overlap` - overlap-aware reconstruction from `from_prev/<video>/tubelet_*/`, with either crop- or blend-based stitching

The script writes one output folder per reconstructed clip under `out_dir`. Use `--all_clips` to reconstruct the full video, or omit it to sample 4 clips by default. Add `--save_mp4` to additionally write a stitched MP4 of the reconstructed clips, adjusting the FPS as required.

### Examples

Baseline:

```bash
python scripts/visualize/reconstruct_from_saved_weights.py \
    --mode baseline \
    --model_dir checkpoints/nervenc/480p_finetuned_baseline \
    --pred_dir save/baseline_preds_quant/baseline_preds_480p_uvg \
    --video jockey \
    --out_dir save/qual/baseline_480p_jockey
```

Patch-tubelet or TeCoNeRV without overlap:

```bash
python scripts/visualize/reconstruct_from_saved_weights.py \
    --mode patch \
    --model_dir checkpoints/teconerv/320x160_pairs_teco \
    --pred_dir save/teconerv_preds_quant/teconerv_preds_480p_160x320_uvg \
    --video beauty \
    --out_dir save/qual/teconerv_480p_beauty \
    --all_clips \
    --save_mp4
```

Patch-tubelet or TeCoNeRV with overlap:

```bash
python scripts/visualize/reconstruct_from_saved_weights.py \
    --mode patch_overlap \
    --model_dir checkpoints/teconerv/320x160_pairs_teco \
    --pred_dir save/teconerv_preds_quant/teconerv_preds_overlap_720p_160x320_uvg \
    --video bosphore \
    --out_dir save/qual/teconerv_overlap_720p_bosphore \
    --blend_overlap false \
    --all_clips \
    --save_mp4
```

### Notes

- `--pred_dir` should point to the root directory produced by the corresponding weight-saver run; it must contain `base_params.pth` and the dumped `direct/` or `from_prev/` subdirectories.
- The overlap mode reads `overlap_h` and `overlap_w` from the saved per-tubelet `metadata.json`. `--blend_overlap false` reproduces crop-based stitching; `--blend_overlap true` enables weighted blending.
