# Training

The repository includes models with three training paradigms:

- `nervenc`: Baseline model training, followed by finetuning.
- `patch_tubelet`: Patch-Tubelet training, followed by finetuning.
- `teconerv`: Temporal-coherence finetuning initialized from a patch-tubelet checkpoint — our proposed TeCoNeRV method.

`nervenc` predicts full-resolution clips directly. `patch_tubelet` and `teconerv` are patch-based models trained on tubelets sampled from the target-resolution videos.

Reference training scripts and configs:

- `scripts/train/train_baseline.sh`, `cfgs/train_baseline.yaml`
- `scripts/train/train_patch_tubelet.sh`, `cfgs/train_patch_tubelet.yaml`
- `scripts/train/train_teconerv.sh`, `cfgs/train_teconerv.yaml`

The shell scripts, with the provided configs and overrides, are the reference for the released training settings.

## Launching training

The scripts under `scripts/train/` use `torchrun` for multi-GPU training. For single-GPU training, `python run_trainer.py ...` is sufficient.

The output directory is determined by `--out_path`, `--name`, `--tag`, and `--instance_tag`. Later finetuning stages reference these saved checkpoints via `finetune_model`.

## Training datasets

Training uses the Kinetics-400 splits described in [datasets.md](datasets.md):

- `k400_2023_train_cls400_50_480p.js` - videos at 480p resolution
- `k400_2023_train_cls400_50_720p.js` - videos at 720p resolution

The split is selected via `--csv_file`. `--frame_num` defines the number of frames per sampled clip; each clip is treated as a local video instance for reconstruction and parameter prediction. For our released checkpoints, we train the hypernetwork on 50 videos from each of the 400 classes in Kinetics, for a total of 10,000 videos.

The training dataset classes are implemented in [vidrec_dataset.py](../datasets/vidrec_dataset.py) and [vidrec_dataset_patches.py](../datasets/vidrec_dataset_patches.py), and selected through `train_dataset` in the training configs.

For video loading at train time, the sampler classes use `video-reader-rs` via `PyVideoReader`, adopted for better memory management than `decord`. Some inference and evaluation code still use `decord` and can be swapped for any preferred backend.

## Training-time evaluation

The `test_dataset` entries in the training configs enable periodic reconstruction checks during training. These are useful sanity checks but are not the evaluation protocol used to report PSNR, SSIM, bpp, and encoding/decoding FPS in the paper. The full evaluation procedure is documented in [evaluation.md](evaluation.md).

## Note on resolution

The main spatial arguments in the training configs are:

- `input_size` - full training or evaluation resolution
- `tubelet_size` - patch size for patch-tubelet and TeCoNeRV models

The released configs follow this convention:

- baseline: `tokenizer.args.input_size == input_size`
- patch-tubelet and TeCoNeRV: `tokenizer.args.input_size == tubelet_size`
- dataset `crop_size` - full spatial resolution
- dataset `tubelet_size` - patch size used to tile the full frame

## Finetuning

Set `finetune_model` to the checkpoint used for initialization at each finetuning stage.

The released configs use:

- Baseline finetuning: `finetune_same_model: true`
- Patch-Tubelet finetuning: `finetune_same_model: true`
- TeCoNeRV finetuning: `finetune_same_model: false`

For TeCoNeRV, `finetune_same_model` must be `false` because the temporal-coherence model is initialized from a patch-tubelet checkpoint but trained with the paired full-resolution model definition ([nerv_enc_full_res_pairs.py](../models/nerv_enc_full_res_pairs.py)).

## Choosing `n_tokens` and `token_dims`

`n_tokens` and `token_dims` are specified per hyponetwork layer in the model configs. Each nonzero `n_tokens[i]` must divide the corresponding layer output channel count `out_ch`. The modulation tensor produced from `token_dims[i]` is repeated and reshaped to match the base weight tensor for that layer. Layer shapes depend on `pe_dim`, `hid_dim`, kernel size (`ks`), and strides (`strds_h`, `strds_w`), as defined in [hypo_convnets_full_res.py](../models/hyponets/hypo_convnets_full_res.py). In the released models, the last layer uses `0` tokens and `0` token dimension.

Parameter counts follow from these settings: the number of unique parameters is `sum(n_tokens[i] * token_dims[i])`, and the number of base parameters is the total number of elements in the initialized `wb` tensors across hyponetwork layers. Following the design heuristic in the paper, the second layer `L2` is typically assigned the largest number of tokens and therefore the largest share of unique parameters.

Users reproducing results from the released checkpoints do not need to modify these settings, this section is primarily relevant when designing new model variants.

## Logging

Checkpoint configs have wandb disabled by default. To enable W&B logging, provide a `wandb.yaml` file and pass `-w` or `--wandb-upload` when launching training.