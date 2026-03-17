import json
import os

import einops
import numpy as np
import torch
from pytorch_msssim import ssim
from tqdm import tqdm

from trainers import register
from trainers.nerv_enc_trainer_full_res import NeRVEncTrainerFullRes


@register("nerv_enc_trainer_full_res_weight_saver")
class NeRVEncTrainerFullResWeightSaver(NeRVEncTrainerFullRes):

    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)

    def add_param_dict_residuals(self, base_dict, residual_dict):
        """Add residuals to a base dictionary layer-wise"""
        result = {}
        for key in base_dict.keys():
            if base_dict[key] is not None and residual_dict[key] is not None:
                result[key] = base_dict[key] + residual_dict[key]
            else:
                result[key] = None
        return result

    def save_weights_no_quant(self, save_path=None, residual_type='from_prev', target_vid='jockey'):
        """
        Save x_dict weights at full resolution, storing residuals from previous clips.

        For each target video:
        - Store base params
        - First clip: Store full x_dict for each tubelet position
        - Subsequent clips: Store residuals from previous clip for each tubelet position
        - Each tubelet position has its own subdirectory, along with metadata for tiling

        """
        if residual_type != 'from_prev':
            raise ValueError(
                f"Invalid residual type: {residual_type}. Only 'from_prev' is currently supported.")

        self.model_ddp.eval()

        root_weights_dir = save_path
        os.makedirs(root_weights_dir, exist_ok=True)

        # Save base params
        if hasattr(self.model_ddp, "module"):
            base_params = self.model_ddp.module.base_params
        else:
            base_params = self.model_ddp.base_params
        torch.save(base_params, os.path.join(
            root_weights_dir, "base_params.pth"))

        for dataset_name, loader in self.test_loader_dict.items():
            dataset = loader.dataset
            videos = dataset.vid_list if hasattr(dataset, 'vid_list') else []

            # Filter target videos
            if isinstance(target_vid, str):
                target_videos = [v for v in videos if target_vid in v.lower()]
            elif isinstance(target_vid, list):
                target_videos = [v for v in videos if any(
                    t in v.lower() for t in target_vid)]

            if not target_videos:
                self.log(f"No target videos found in {dataset_name}")
                continue

            self.log(
                f'Processing {len(target_videos)} target videos in {dataset_name}')

            for video in tqdm(target_videos):
                # Create video-specific directory
                vid_name = video.split('/')[-1].replace('_1080', '')
                video_dir = os.path.join(
                    root_weights_dir, residual_type, vid_name)
                os.makedirs(video_dir, exist_ok=True)

                # Create ordered dataset and loader for this video
                ordered_dataset = self._make_ordered_dataset(dataset, video)
                ordered_loader = self._make_data_loader(
                    ordered_dataset, batch_size=1, num_workers=loader.num_workers
                )

                # Store reconstructed x_dict per position
                prev_x_dict_recon_per_pos = {}  # Dict[pos_key, x_dict_recon]

                video_psnr = []
                video_ssim = []

                for clip_idx, data in enumerate(ordered_loader):
                    # Removing batch dimension since one clip is processed at a time
                    data = {k: v[0].cuda() if isinstance(v, torch.Tensor) else v
                            for k, v in data.items()}

                    # Get patch-tubelets and positions
                    patches = data['patches']  # list of patches for each frame
                    # positions of patches in the full frame
                    positions = data['patch_positions']
                    start_frame = data['start_frame']

                    # Process patches for all frames in clip together
                    all_patches = [[] for _ in range(len(patches[0]))]
                    for frame_patches in patches:
                        for i, patch in enumerate(frame_patches):
                            all_patches[i].append(patch.squeeze(0))
                    all_patches = [torch.stack(patch_seq, dim=1)
                                   for patch_seq in all_patches]

                    x_dict_all, _ = self._process_patches_batch(all_patches)
                    x_dict_recon_patches = []

                    # Process each patch position
                    for pos_idx, pos_tuple in enumerate(positions[0]):
                        pos_key = tuple(p.item() for p in pos_tuple)

                        tubelet_dir = os.path.join(
                            video_dir, f"tubelet_{pos_key[0]}_{pos_key[1]}")
                        os.makedirs(tubelet_dir, exist_ok=True)

                        # Extract x_dict for current patch position
                        cur_x_dict = {
                            k: v[pos_idx:pos_idx+1].cuda()
                            if v is not None else None
                            for k, v in x_dict_all.items()
                        }

                        if start_frame == 0:
                            # First clip: store full x_dict, and store as previous state
                            torch.save(cur_x_dict, os.path.join(
                                tubelet_dir, f"clip_0_frame_0_full.pth"))
                            x_dict_recon_patches.append(cur_x_dict)
                            prev_x_dict_recon_per_pos[pos_key] = self._clone_dict(
                                cur_x_dict)

                            metadata = {
                                'position': pos_key,
                                'type': 'full',
                                'clip_idx': 0,
                                'frame_idx': 0,
                                'patch_size': (dataset.tubelet_size[0], dataset.tubelet_size[1]),
                                'num_frames': len(positions[0]),
                                'full_res': (dataset.crop_size[0], dataset.crop_size[1])
                            }
                            with open(os.path.join(tubelet_dir, "metadata.json"), 'w') as f:
                                json.dump(metadata, f, indent=4)

                        else:
                            # Subsequent clips: store residual from previous
                            if pos_key not in prev_x_dict_recon_per_pos:
                                raise RuntimeError(
                                    f"Missing previous state for position {pos_key}")

                            residual = self.compute_param_dict_residuals(
                                cur_x_dict, prev_x_dict_recon_per_pos[pos_key])

                            torch.save(residual, os.path.join(tubelet_dir,
                                                              f"clip_{clip_idx}_frame_{start_frame}_residual.pth"))

                            # Update previous state by adding residual to previous state
                            x_dict_recon_prev = self.add_param_dict_residuals(
                                prev_x_dict_recon_per_pos[pos_key], residual)

                            x_dict_recon_patches.append(x_dict_recon_prev)
                            prev_x_dict_recon_per_pos[pos_key] = self._clone_dict(
                                x_dict_recon_prev)

                    # Reconstruct full frames and compute metrics
                    recon_patches = self.reconstruct_from_x_dict(
                        len(positions[0]), dataset.frame_num, self._combine_batch_reconstructions(x_dict_recon_patches), device=data['gt'].device)
                    recon_clip = self._tile_clip_from_patches(
                        recon_patches, positions, data['metadata'])

                    gt_clip = data['gt'].squeeze(0)
                    gt_clip = einops.rearrange(gt_clip, "c t h w -> t c h w")

                    mse = ((recon_clip - gt_clip) ** 2).mean()
                    psnr = -10 * torch.log10(mse)
                    ssim_val = ssim(recon_clip, gt_clip, data_range=1.0)

                    video_psnr.append(psnr.item())
                    video_ssim.append(ssim_val.item())

                avg_psnr = np.mean(video_psnr)
                avg_ssim = np.mean(video_ssim)
                self.log(
                    f"{vid_name}: avg_psnr={avg_psnr:.4f}, avg_ssim={avg_ssim:.4f}")

        self.log(f"Saved weights at {root_weights_dir}")

    def save_weights_quant(self, save_path=None, residual_type='from_prev', quant_axis=0, target_vid='jockey'):
        """Save x_dict weights at full resolution with quantization of residuals.

        For each target video:
        - Store base params
        - First clip: Store full x_dict for each tubelet position
        - Subsequent clips: Store residuals from previous clip for each tubelet position
        - Each tubelet position has its own subdirectory, along with metadata for tiling
        """
        if residual_type != 'from_prev':
            raise ValueError(
                f"Invalid residual type: {residual_type}. Only 'from_prev' is currently supported.")

        quant_bit = 8
        self.model_ddp.eval()

        root_weights_dir = save_path
        os.makedirs(root_weights_dir, exist_ok=True)

        # Save base params
        if hasattr(self.model_ddp, "module"):
            base_params = self.model_ddp.module.base_params
        else:
            base_params = self.model_ddp.base_params
        torch.save(base_params, os.path.join(
            root_weights_dir, "base_params.pth"))

        for dataset_name, loader in self.test_loader_dict.items():
            dataset = loader.dataset
            videos = dataset.vid_list if hasattr(dataset, 'vid_list') else []

            # Filter target videos
            if isinstance(target_vid, str):
                target_videos = [v for v in videos if target_vid in v.lower()]
            elif isinstance(target_vid, list):
                target_videos = [v for v in videos if any(
                    t in v.lower() for t in target_vid)]

            if not target_videos:
                self.log(f"No target videos found in {dataset_name}")
                continue

            self.log(
                f'Processing {len(target_videos)} target videos in {dataset_name}')

            for video in tqdm(target_videos):
                vid_name = video.split('/')[-1].replace('_1080', '')
                video_dir = os.path.join(
                    root_weights_dir, residual_type, vid_name)
                os.makedirs(video_dir, exist_ok=True)

                ordered_dataset = self._make_ordered_dataset(dataset, video)
                ordered_loader = self._make_data_loader(
                    ordered_dataset, batch_size=1, num_workers=loader.num_workers
                )

                # Store reconstructed x_dict per position
                prev_x_dict_recon_per_pos = {}  # Dict[pos_key, x_dict_recon]

                video_psnr = []
                video_ssim = []

                for clip_idx, data in enumerate(ordered_loader):
                    # Removing batch dimension since one clip is processed at a time
                    data = {k: v[0].cuda() if isinstance(v, torch.Tensor) else v
                            for k, v in data.items()}

                    # Get patch-tubelets and positions
                    patches = data['patches']  # list of patches for each frame
                    # positions of patches in the full frame
                    positions = data['patch_positions']
                    start_frame = data['start_frame']

                    # Process patches for all frames in clip together
                    all_patches = [[] for _ in range(len(patches[0]))]
                    for frame_patches in patches:
                        for i, patch in enumerate(frame_patches):
                            all_patches[i].append(patch.squeeze(0))
                    all_patches = [torch.stack(patch_seq, dim=1)
                                   for patch_seq in all_patches]

                    x_dict_all, _ = self._process_patches_batch(all_patches)
                    x_dict_recon_patches = []

                    # Process each patch position
                    for pos_idx, pos_tuple in enumerate(positions[0]):
                        pos_key = tuple(p.item() for p in pos_tuple)

                        tubelet_dir = os.path.join(
                            video_dir, f"tubelet_{pos_key[0]}_{pos_key[1]}")
                        os.makedirs(tubelet_dir, exist_ok=True)

                        # Extract x_dict for current patch position
                        cur_x_dict = {
                            k: v[pos_idx:pos_idx+1].cuda()
                            if v is not None else None
                            for k, v in x_dict_all.items()
                        }

                        if start_frame == 0:
                            # First clip: quantize and save full x_dict
                            # Quantize x_dict to 8 bits
                            quantized_x_dict, scales, t_mins = self.quantize_param_dict(
                                cur_x_dict, quant_bit, quant_axis)

                            # Dequantize x_dict
                            dequantized_x_dict = self.recover_param_dict_from_quantized_residuals(
                                None, quantized_x_dict, scales, t_mins)

                            # Save dequantized x_dict and quantization parameters
                            torch.save(dequantized_x_dict, os.path.join(
                                tubelet_dir, f"clip_0_frame_0_full.pth"))

                            # Store as previous state
                            x_dict_recon_patches.append(dequantized_x_dict)
                            prev_x_dict_recon_per_pos[pos_key] = self._clone_dict(
                                dequantized_x_dict)

                            metadata = {
                                'position': pos_key,
                                'type': 'full',
                                'clip_idx': 0,
                                'frame_idx': 0,
                                'patch_size': (data['gt'].shape[-2], data['gt'].shape[-1]),
                                'num_frames': len(positions[0]),
                                'full_res': (dataset.crop_size[0], dataset.crop_size[1]),
                                'quant_bit': quant_bit,
                                'quant_axis': quant_axis,
                                'scales': {k: v.tolist() if v is not None else None for k, v in scales.items()},
                                't_mins': {k: v.tolist() if v is not None else None for k, v in t_mins.items()}
                            }
                            with open(os.path.join(tubelet_dir, "metadata.json"), 'w') as f:
                                json.dump(metadata, f, indent=4)

                        else:
                            # Subsequent clips: compute, quantize, and save residuals
                            if pos_key not in prev_x_dict_recon_per_pos:
                                raise RuntimeError(
                                    f"Missing previous state for position {pos_key}")

                            residual = self.compute_param_dict_residuals(
                                cur_x_dict, prev_x_dict_recon_per_pos[pos_key])

                            quantized_residual, scales, t_mins = self.quantize_param_dict(
                                residual, quant_bit, quant_axis)

                            dequantized_residual = self.recover_param_dict_from_quantized_residuals(
                                None, quantized_residual, scales, t_mins)

                            # Save dequantized residual and quantization parameters
                            torch.save(dequantized_residual, os.path.join(
                                tubelet_dir, f"clip_{clip_idx}_frame_{start_frame}_residual.pth"))

                            # Update previous state by adding dequantized residual to previous state
                            x_dict_recon_prev = self.add_param_dict_residuals(
                                prev_x_dict_recon_per_pos[pos_key], dequantized_residual)

                            x_dict_recon_patches.append(x_dict_recon_prev)
                            prev_x_dict_recon_per_pos[pos_key] = self._clone_dict(
                                x_dict_recon_prev)

                    # Reconstruct full frames and compute metrics
                    recon_patches = self.reconstruct_from_x_dict(
                        len(positions[0]), dataset.frame_num, self._combine_batch_reconstructions(x_dict_recon_patches), device=data['gt'].device)
                    recon_clip = self._tile_clip_from_patches(
                        recon_patches, positions, data['metadata'])

                    gt_clip = data['gt'].squeeze(0)
                    gt_clip = einops.rearrange(gt_clip, "c t h w -> t c h w")

                    mse = ((recon_clip - gt_clip) ** 2).mean()
                    psnr = -10 * torch.log10(mse)
                    ssim_val = ssim(recon_clip, gt_clip, data_range=1.0)

                    video_psnr.append(psnr.item())
                    video_ssim.append(ssim_val.item())

                avg_psnr = np.mean(video_psnr)
                avg_ssim = np.mean(video_ssim)

        self.log(f"Saved quantized weights at {root_weights_dir}")
