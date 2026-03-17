# Model Checkpoints

Our model checkpoints are hosted on Hugging Face at:

- https://huggingface.co/namithap/teconerv-models

These are the hypernetwork training checkpoints — not per-video hyponetwork weight predictions, saved `x_dict` tensors, or stored residual bitstreams. For training details see [training.md](training.md); for evaluation see [evaluation.md](evaluation.md).

## Downloading checkpoints

Install Git LFS and clone the checkpoint repository:

```bash
git lfs install
git clone https://huggingface.co/namithap/teconerv-models
```

Git Xet can also be used, as described on the Hugging Face model page.

After cloning, copy the checkpoint folders into `checkpoints/` in this repository:

```text
checkpoints/
  nervenc/
  patch_tubelet/
  teconerv/
```

The released training and evaluation scripts expect this layout.

If you only plan to download the primary released checkpoints, start with:

- `nervenc/480p_finetuned_baseline`
- `nervenc/720p_finetuned_baseline`
- `teconerv/320x160_pairs_teco`
- `teconerv/320x240_pairs_teco`
- `teconerv/384x270_pairs_teco`

## Checkpoint families

Checkpoints are organized into three families, corresponding to the three training paradigms in [training.md](training.md).

**`nervenc`** — Baseline NeRVEnc hypernetwork. Predicts full-resolution clips directly without spatial patch decomposition.

**`patch_tubelet`** — Patch-tubelet hypernetwork. Predicts parameters for spatial tubelets; full frames are reconstructed by tiling all tubelets. Supports inference at resolutions not seen during training.

**`teconerv`** — Proposed TeCoNeRV method. Initialized from a patch-tubelet checkpoint and fine-tuned with a temporal coherence objective over consecutive clip pairs.

Each checkpoint directory contains a `cfg.yaml` (the training configuration) and `epoch-last.pth` (the model weights).

## Naming conventions

- `pre_finetune_*` — checkpoint before the finetuning stage, trained for 150 epochs and stored under `pre_finetune/` within each family directory
- `*_finetuned_*` — checkpoint after finetuning for 50 additional epochs
- `*_small` — smaller-capacity variant (see [Figure 3 note](#figure-3-small-variants) below)
- `<height>x<width>` — tubelet spatial size for patch-based models
- `*_train_720p` — trained on tubelets sampled from 720p source videos; absent suffix means 480p-trained

## `nervenc`

The baseline NeRVEnc family. These models predict full-resolution clips directly and do not use patch decomposition.

Primary checkpoints:

- `480p_finetuned_baseline` — reported for 480p inference in Table 1
- `720p_finetuned_baseline` — reported for 720p inference in Table 1

Pre-finetune checkpoints (under `pre_finetune/`):

- `pre_finetune/pre_finetune_480p_baseline`
- `pre_finetune/pre_finetune_480p_baseline_small`
- `pre_finetune/pre_finetune_720p_baseline`

These are provided for reproducibility and are not directly evaluated in the paper.

## `patch_tubelet`

The patch-tubelet family, without temporal coherence regularization. Trained on tubelets extracted from source videos; full-frame reconstruction is obtained by tiling all patch predictions.

Primary checkpoints:

- `320x160_finetuned_patch` — 480p-trained patch-tubelet model; the main 480p patch-tubelet baseline reported in the paper
- `320x240_finetuned_patch_train_720p` — 720p-trained patch-tubelet model

These are the only patch-tubelet checkpoints most users will need.

Pre-finetune checkpoints (under `pre_finetune/`):

- `pre_finetune/pre_finetune_320x160_patch`
- `pre_finetune/pre_finetune_320x160_patch_small`
- `pre_finetune/pre_finetune_320x240_patch_train_480p`
- `pre_finetune/pre_finetune_320x240_patch_train_720p`
- `pre_finetune/pre_finetune_384x270_patch_train_480p`
- `pre_finetune/pre_finetune_384x270_patch_train_720p`

These are provided for reproducibility. Each pre-finetune checkpoint is the initialization for two downstream finetuning runs: 50-epoch patch-tubelet finetuning (without temporal coherence loss) to produce the patch-tubelet baselines, and 50-epoch TeCoNeRV finetuning (with temporal coherence loss) to produce the corresponding TeCoNeRV checkpoints.

## `teconerv`

The proposed TeCoNeRV family. Each checkpoint is initialized from the corresponding patch-tubelet pre-finetune checkpoint and finetuned with temporal coherence regularization for 50 epochs.

Final checkpoints (480p-trained):

- `320x160_pairs_teco` — final model for 480p inference; also used for overlapped evaluations at 480p where indicated
- `320x240_pairs_teco` — 480p-trained model reported for higher-resolution inference
- `384x270_pairs_teco` — 480p-trained model reported for higher-resolution inference

These three are the primary TeCoNeRV results in the paper. The key result is that models trained on 480p patches generalize to 720p and 1080p inference without retraining for each target inference resolution, demonstrating resolution-independent inference. Similarly, the models trained on 720p patches can generalize to 1080p inference.

Additional 720p-trained variants:

- `320x240_pairs_teco_train_720p`
- `384x270_pairs_teco_train_720p`

Both the 480p-trained and 720p-trained `320x240` and `384x270` variants are reported for 720p and 1080p inference in Table 1. The 480p-trained variants are additionally used for the overlapped inference results in later tables. No 720p-trained variant was produced for the `320x160` tubelet size; `320x160_pairs_teco` is the sole final model at that tubelet size and covers 480p inference.

For evaluation, TeCoNeRV checkpoints require the overrides documented in [evaluation.md](evaluation.md): `eval_same_model: false` with `eval_saver: nerv_enc_full_res`.

## Figure 3 small variants

The PSNR-bpp Pareto frontier in Figure 3 uses smaller-capacity variants of each family alongside the full models. The relevant checkpoints are:

- `nervenc/480p_finetuned_baseline_small`
- `patch_tubelet/320x160_finetuned_patch_small`
- `teconerv/320x160_pairs_teco_small`
