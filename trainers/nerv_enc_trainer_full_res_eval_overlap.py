import os
import time
from collections import defaultdict
from copy import deepcopy

import einops
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import make
from trainers import register
from trainers.nerv_enc_trainer_full_res import NeRVEncTrainerFullRes
from utils import make_coord_grid


@register("nerv_enc_trainer_full_res_eval_overlap")
class NeRVEncTrainerFullResEvalOverlap(NeRVEncTrainerFullRes):

    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)
    
    def _process_patches_batch_chunked(self, all_patches, batch_size=None):
        """
        Process patches in smaller batches to avoid memory issues.
        Note: This may slow down the encoding speed.
        """
        if batch_size is None:
            return self._process_patches_batch(all_patches)
        
        num_patches = len(all_patches)
        x_dict_results = []
        hyponet_weights_results = []

        for i in range(0, num_patches, batch_size):
            batch_patches = all_patches[i:i+batch_size]
            x_dict_batch, hyponet_weights_batch = self._process_patches_batch(batch_patches)
            x_dict_results.append(x_dict_batch)
            hyponet_weights_results.append(hyponet_weights_batch)
        
        # Combine results from all batches
        x_dict_all = self._combine_param_dict_batches(x_dict_results)
        return x_dict_all, None

    def reconstruct_from_x_dict_chunked(self, B, T, x_dict_recon, device, batch_size=None):
        """Reconstruct from x_dict, optionally in chunks to handle memory constraints."""
        coord = make_coord_grid((T,), (-1, 1), device=device)
        model = self.model_ddp.module if hasattr(self.model_ddp, "module") else self.model_ddp
        base_params = model.base_params
        
        # If no batch_size specified, process all at once (default behavior)
        if batch_size is None:
            coord = einops.repeat(coord, "t d -> b t d", b=B)
            params = self.convert_x_dict_to_params(B, x_dict_recon, base_params, model)
            hyponet_x = model.hyponet
            hyponet_x.set_params(params)    
            pred_x = hyponet_x(coord)
            return pred_x
        
        # Otherwise, process in chunks
        pred_results = []
        for i in range(0, B, batch_size):
            batch_end = min(i + batch_size, B)
            batch_size_actual = batch_end - i
            
            # Extract batch from x_dict_recon
            x_dict_batch = {}
            for key, value in x_dict_recon.items():
                if value is not None:
                    x_dict_batch[key] = value[i:batch_end]
                else:
                    x_dict_batch[key] = None
            
            # Create coordinate batch and reconstruct
            coord_batch = einops.repeat(coord, "t d -> b t d", b=batch_size_actual)
            params_batch = self.convert_x_dict_to_params(batch_size_actual, x_dict_batch, base_params, model)
            hyponet_x = model.hyponet
            hyponet_x.set_params(params_batch)    
            pred_batch = hyponet_x(coord_batch)
            pred_results.append(pred_batch) # t b 3 h w
        
        return torch.cat(pred_results, dim=1)

    def _calculate_patch_grid_info(self, patch_positions, overlap_h, overlap_w):
        """
        Calculate cropping information for each patch based on its position in the grid.
        
        - Interior patches: crop half the overlap from each side
        - Edge and corner patches: only crop overlaps on sides that have neighbors
        """
        num_patches = len(patch_positions[0])
        patch_grid = []
        
        # First, organize patches into a 2D grid to understand neighbors
        patches_by_position = {}
        for idx in range(num_patches):
            h_start = patch_positions[0][idx][0].item()
            w_start = patch_positions[0][idx][1].item()
            patches_by_position[(h_start, w_start)] = idx
        
        # Get unique positions to understand grid layout
        h_positions = sorted(set(pos[0] for pos in patches_by_position.keys()))
        w_positions = sorted(set(pos[1] for pos in patches_by_position.keys()))
        
        for patch_idx in range(num_patches):
            h_start = patch_positions[0][patch_idx][0].item()
            w_start = patch_positions[0][patch_idx][1].item()
            
            # Find position in grid
            h_idx = h_positions.index(h_start)
            w_idx = w_positions.index(w_start)
            
            # Determine if patch has neighbors
            has_top_neighbor = h_idx > 0
            has_bottom_neighbor = h_idx < len(h_positions) - 1
            has_left_neighbor = w_idx > 0
            has_right_neighbor = w_idx < len(w_positions) - 1
            
            # Calculate cropping amounts
            crop_top = (overlap_h // 2) if has_top_neighbor else 0
            crop_bottom = (overlap_h // 2) if has_bottom_neighbor else 0
            crop_left = (overlap_w // 2) if has_left_neighbor else 0
            crop_right = (overlap_w // 2) if has_right_neighbor else 0
            
            # Ensure we do not over-crop small overlaps
            crop_top = min(crop_top, overlap_h)
            crop_bottom = min(crop_bottom, overlap_h) 
            crop_left = min(crop_left, overlap_w)
            crop_right = min(crop_right, overlap_w)
            
            patch_grid.append({
                'crop_top': crop_top,
                'crop_bottom': crop_bottom,
                'crop_left': crop_left,
                'crop_right': crop_right,
                'has_neighbors': {
                    'top': has_top_neighbor,
                    'bottom': has_bottom_neighbor,
                    'left': has_left_neighbor,
                    'right': has_right_neighbor
                }
            })
        
        return patch_grid

    def _tile_clip_from_overlapping_patches_with_cropping(self, recon_patches, patch_positions, metadata):
        """
        Reconstruct full clip from overlapping patch-tubelets by cropping regions
        at the boundaries, assigning each pixel to exactly one patch.
        
        Args:
            recon_patches: Tensor of shape (T, num_patches, C, H, W) - reconstructed patches
            patch_positions: List of patch positions for each frame
            metadata: Metadata containing frame dimensions and overlap info
        
        Returns:
            recon_clip: Tensor of shape (T, C, H, W) - full reconstructed clip
        """
        T, num_patches, C, patch_h, patch_w = recon_patches.shape
        H_full, W_full = metadata['crop_size'][0], metadata['crop_size'][1]
        overlap_h = metadata['overlap_h'] if 'overlap_h' in metadata else 0
        overlap_w = metadata['overlap_w'] if 'overlap_w' in metadata else 0
        
        recon_clip = torch.zeros(T, C, H_full, W_full, device=recon_patches.device)
        
        patch_grid = self._calculate_patch_grid_info(patch_positions, overlap_h, overlap_w)
        
        # Process each patch-tubelet position and crop appropriately
        for patch_idx in range(num_patches):
            h_start = patch_positions[0][patch_idx][0].item()
            w_start = patch_positions[0][patch_idx][1].item() 
            h_end = patch_positions[0][patch_idx][2].item()
            w_end = patch_positions[0][patch_idx][3].item()
            
            patch_pred = recon_patches[:, patch_idx]  # (T, C, patch_h, patch_w)

            grid_info = patch_grid[patch_idx]
            crop_top = grid_info['crop_top']
            crop_bottom = grid_info['crop_bottom'] 
            crop_left = grid_info['crop_left']
            crop_right = grid_info['crop_right']
            
            patch_h_actual, patch_w_actual = h_end - h_start, w_end - w_start
            patch_cropped = patch_pred[:, :, 
                                    crop_top:patch_h_actual-crop_bottom,
                                    crop_left:patch_w_actual-crop_right]
            
            # Calculate where to place the cropped patch in the full image
            out_h_start = h_start + crop_top
            out_h_end = h_end - crop_bottom
            out_w_start = w_start + crop_left
            out_w_end = w_end - crop_right
            
            # Place cropped patch-tubelet in output
            recon_clip[:, :, out_h_start:out_h_end, out_w_start:out_w_end] = patch_cropped
        
        return recon_clip

    def _tile_clip_from_overlapping_patches_with_blending(self, recon_patches, positions, metadata):
        """
        Reconstruct full clip from overlapping patch-tubelets using weighted blending in overlap regions.
        
        Args:
            recon_patches: Tensor of shape (T, num_patches, C, patch_h, patch_w) - reconstructed patches
            positions: List of patch positions for each frame
            metadata: Metadata containing frame dimensions and overlap info
        
        Returns:
            recon_clip: Tensor of shape (T, C, H, W) - full reconstructed clip
        """
        T, num_patches, C, patch_h, patch_w = recon_patches.shape
        H_full, W_full = metadata['crop_size'][0], metadata['crop_size'][1]
        patch_positions = positions[0]

        overlap_h = metadata['overlap_h'] if 'overlap_h' in metadata else 0
        overlap_w = metadata['overlap_w'] if 'overlap_w' in metadata else 0

        # Initialize output tensor and weight tensor for blending
        recon_clip = torch.zeros(T, C, H_full, W_full, device=recon_patches.device)
        weight_map = torch.zeros(T, C, H_full, W_full, device=recon_patches.device)
        
        # Process each patch-tubelet position and crop appropriately
        for patch_idx in range(num_patches):
            h_start = patch_positions[patch_idx][0].item()
            w_start = patch_positions[patch_idx][1].item() 
            h_end = patch_positions[patch_idx][2].item()
            w_end = patch_positions[patch_idx][3].item()
            
            # Create weight mask for this patch-tubelet with fading in each dimension
            patch_h, patch_w = h_end - h_start, w_end - w_start
            weight_h = torch.ones(patch_h)
            weight_w = torch.ones(patch_w)
            
            # Apply fading only in dimensions that have overlap
            if overlap_h > 0:
                fade_h = min(overlap_h // 2, patch_h // 4)
                fade_h = int(fade_h)
                if patch_h > 2 * fade_h and fade_h > 0:
                    weight_h[:fade_h] = torch.linspace(0.1, 1.0, fade_h)
                    weight_h[-fade_h:] = torch.linspace(1.0, 0.1, fade_h)
            
            if overlap_w > 0:
                fade_w = min(overlap_w // 2, patch_w // 4)
                fade_w = int(fade_w)
                if patch_w > 2 * fade_w and fade_w > 0:
                    weight_w[:fade_w] = torch.linspace(0.1, 1.0, fade_w)
                    weight_w[-fade_w:] = torch.linspace(1.0, 0.1, fade_w)
            
            # Create 2D weight map
            patch_weight = weight_h[:, None] * weight_w[None, :]
            patch_weight = patch_weight[None, None, :, :].expand(T, C, -1, -1).to(recon_patches.device)

            # Add weighted patch to output
            patch_pred = recon_patches[:, patch_idx]  # Shape: (T, C, H, W)
            recon_clip[:, :, h_start:h_end, w_start:w_end] += patch_pred * patch_weight
            # For each pixel position, weight_map stores the sum of all weights applied
            weight_map[:, :, h_start:h_end, w_start:w_end] += patch_weight
        
        # Normalize by weight map.
        if overlap_h > 0 or overlap_w > 0:
            recon_clip = recon_clip / (weight_map + 1e-8)
        
        return recon_clip
    
    def _make_ordered_dataset_overlapping(self, dataset, video_path, overlap_h, overlap_w):
        """Create ordered overlappping patch dataset for a video"""
        return make(self.cfg['eval_full_res_dataset'], args={
            'video_path': video_path,
            'frame_num': dataset.frame_num,
            'crop_size': dataset.crop_size,
            'tubelet_size': dataset.tubelet_size,
            'overlap_h': overlap_h,
            'overlap_w': overlap_w,
        })

    def _make_data_loader(self, dataset, batch_size, num_workers):
        """Create data loader with same settings as base loader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    def _log_full_res_metrics_overlapping(self, recon_type, dataset_name, csv_path, csv_path_per_video, 
                                    metrics_per_video, clip_metrics_across_videos, ordered_dataset, videos, 
                                    encoding_type, quant_bit, quant_axis, overlap_h, overlap_w, blend_overlap, 
                                    eval_csv_prefix, log_per_video=True):
        """
        Logging for overlapping patch-tubelets with correct bpp calculation.
        With overlapped patches, we cannot unfairly average bpp across patch-tubelets.
        Instead, we must consider the total number of bits over all patches used
        and divide by the actual number of pixels in the full resolution tubelet. 
        This accounts for overlapping patches having more parameters than non-overlapping patches.
        """
        crop_size = ordered_dataset.crop_size
        pixels_per_tubelet = crop_size[0] * crop_size[1] * ordered_dataset.frame_num # full resolution pixels

        self.log(f'\nFull-res {recon_type} overlapping patches eval on {dataset_name}: '
                f'quant_bit={quant_bit}, quant_axis={quant_axis}, encoding={encoding_type or "none"}, '
                f'overlap_h={overlap_h}, overlap_w={overlap_w}')
        
        if log_per_video:
            for video in videos:
                video_metrics = metrics_per_video[video]
                log_buffer = [f'\nVideo: {os.path.basename(video)}']
                
                # Calculate bpp using total bits / full resolution pixels
                bpp_quant = np.mean(video_metrics['bits_quant']) / pixels_per_tubelet
                overhead_bpp_quant = np.mean(video_metrics['bits_quant_overhead']) / pixels_per_tubelet
                
                for method in ['direct', 'from_first', 'from_prev']:
                    avg_psnr = np.mean(video_metrics[f'{method}_psnr'])
                    avg_ssim = np.mean(video_metrics[f'{method}_ssim'])
                    
                    # Calculate encoded bpp using total bits from all patches / full resolution pixels
                    bpp_encoded = np.mean(video_metrics[f'{method}_bits_encoded_tubelet']) / pixels_per_tubelet if encoding_type else 0
                    # Final bpp is encoded bpp, if enabled, otherwise quant + overhead bpp
                    bpp_total = bpp_encoded if encoding_type else (bpp_quant + overhead_bpp_quant)
                    
                    avg_enc_fps = np.mean(video_metrics[f'{method}_enc_fps'])
                    avg_dec_fps = np.mean(video_metrics[f'{method}_dec_fps'])
                    
                    log_buffer.extend([
                        f'\n{recon_type}_recon - {method}:',
                        f'avg_psnr={avg_psnr:.4f}',
                        f'avg_ssim={avg_ssim:.4f}',
                        f'bpp_total={bpp_total:.4f}',
                        f'bpp_quant={bpp_quant:.4f}',
                        f'bpp_quant_with_overhead={(bpp_quant + overhead_bpp_quant):.4f}',
                        f'bpp_encoded={bpp_encoded:.4f}',
                        f'enc_fps={avg_enc_fps:.2f}',
                        f'dec_fps={avg_dec_fps:.2f}'
                    ])
                
                self.log(', '.join(log_buffer))
        
        self.log('\nAverages across all videos:')
        
        # Calculate average bpp using total bits / full resolution pixels
        avg_bpp_quant = np.mean(clip_metrics_across_videos['bits_quant']) / pixels_per_tubelet
        avg_overhead_bpp_quant = np.mean(clip_metrics_across_videos['bits_quant_overhead']) / pixels_per_tubelet
        
        for method in ['direct', 'from_first', 'from_prev']:
            avg_psnr = np.mean(clip_metrics_across_videos[method]['psnr'])
            avg_ssim = np.mean(clip_metrics_across_videos[method]['ssim'])
            
            # Use total bits from all patches / full resolution pixels
            avg_bpp_encoded = np.mean(clip_metrics_across_videos[method]['bits_encoded_tubelet']) / pixels_per_tubelet if encoding_type else 0
            avg_bpp_total = avg_bpp_encoded if encoding_type else (avg_bpp_quant + avg_overhead_bpp_quant)
            
            avg_enc_fps = np.mean(clip_metrics_across_videos[method]['enc_fps'])
            avg_dec_fps = np.mean(clip_metrics_across_videos[method]['dec_fps'])
            
            self.log(
                f'\n{recon_type}_recon - {method} (average): '
                f'avg_psnr={avg_psnr:.4f}, '
                f'avg_ssim={avg_ssim:.4f}, '
                f'bpp_total={avg_bpp_total:.4f}, '
                f'bpp_quant={avg_bpp_quant:.4f}, '
                f'bpp_quant_with_overhead={(avg_bpp_quant + avg_overhead_bpp_quant):.4f}, '
                f'bpp_encoded={avg_bpp_encoded:.4f}, '
                f'enc_fps={avg_enc_fps:.2f}, '
                f'dec_fps={avg_dec_fps:.2f}'
            )

        if csv_path:
            csv_columns = [
                'quant_bit',
                'residual_type',
                'psnr',
                'ssim',
                'bpp_total',
                'enc_fps',
                'dec_fps',
                'video',
                'reconstruction_type',
                'quant_axis',
                'encoding_type',
                'bpp_quant',
                'bpp_quant_with_overhead',
                'bpp_encoded',
                'overlap_h',
                'overlap_w',
                'num_patches_total',
                'blend_overlap',
            ]
            
            if log_per_video:
                file_exists = os.path.isfile(csv_path_per_video)
                mode = 'a' if file_exists else 'w'
                with open(csv_path_per_video, mode) as f:
                    if not file_exists:
                        f.write(','.join(csv_columns) + '\n')
                    
                    for video in videos:
                        video_metrics = metrics_per_video[video]
                        video_name = os.path.basename(video)

                        bpp_quant = np.mean(video_metrics['bits_quant']) / pixels_per_tubelet
                        overhead_bpp_quant = np.mean(video_metrics['bits_quant_overhead']) / pixels_per_tubelet    
                        
                        for method in ['direct', 'from_first', 'from_prev']:
                            avg_psnr = np.mean(video_metrics[f'{method}_psnr'])
                            avg_ssim = np.mean(video_metrics[f'{method}_ssim'])
                            
                            # Use total bits / full resolution pixels for bpp
                            bpp_encoded = np.mean(video_metrics[f'{method}_bits_encoded_tubelet']) / pixels_per_tubelet if encoding_type else 0
                            bpp_total = bpp_encoded if encoding_type else (bpp_quant + overhead_bpp_quant)
                            
                            avg_enc_fps = np.mean(video_metrics[f'{method}_enc_fps'])
                            avg_dec_fps = np.mean(video_metrics[f'{method}_dec_fps'])
                            
                            # Calculate total patches for this configuration
                            num_patches_total = ordered_dataset.num_patches
                            row = [
                                f'{quant_bit}',
                                method,
                                f'{avg_psnr:.4f}',
                                f'{avg_ssim:.4f}',
                                f'{bpp_total:.4f}',
                                f'{avg_enc_fps:.2f}',
                                f'{avg_dec_fps:.2f}',
                                video_name,
                                recon_type,
                                f'{quant_axis}',
                                f'{encoding_type or "none"}',
                                f'{bpp_quant:.4f}',
                                f'{bpp_quant + overhead_bpp_quant:.4f}',
                                f'{bpp_encoded:.4f}',
                                f'{overlap_h}',
                                f'{overlap_w}',
                                f'{num_patches_total}',
                                f'{blend_overlap}'
                            ]
                            f.write(','.join(row) + '\n')
            
            file_exists = os.path.isfile(csv_path)
            mode = 'a' if file_exists else 'w'
            with open(csv_path, mode) as f:
                if not file_exists:
                    f.write(','.join(csv_columns) + '\n')

                # Write averages across all videos
                for method in ['direct', 'from_first', 'from_prev']:
                    avg_psnr = np.mean(clip_metrics_across_videos[method]['psnr'])
                    avg_ssim = np.mean(clip_metrics_across_videos[method]['ssim'])
                    
                    # Use total bits / full resolution pixels for bpp
                    avg_bpp_quant = np.mean(clip_metrics_across_videos['bits_quant']) / pixels_per_tubelet
                    avg_overhead_bpp_quant = np.mean(clip_metrics_across_videos['bits_quant_overhead']) / pixels_per_tubelet
                    avg_bpp_encoded = np.mean(clip_metrics_across_videos[method]['bits_encoded_tubelet']) / pixels_per_tubelet if encoding_type else 0
                    avg_bpp_total = avg_bpp_encoded if encoding_type else (avg_bpp_quant + avg_overhead_bpp_quant)
                    
                    avg_enc_fps = np.mean(clip_metrics_across_videos[method]['enc_fps'])
                    avg_dec_fps = np.mean(clip_metrics_across_videos[method]['dec_fps'])
                    
                    num_patches_total = ordered_dataset.num_patches

                    row = [
                        f'{quant_bit}',
                        method,
                        f'{avg_psnr:.4f}',
                        f'{avg_ssim:.4f}',
                        f'{avg_bpp_total:.4f}',
                        f'{avg_enc_fps:.2f}',
                        f'{avg_dec_fps:.2f}',
                        'all',
                        recon_type,
                        f'{quant_axis}',
                        f'{encoding_type or "none"}',
                        f'{avg_bpp_quant:.4f}',
                        f'{avg_bpp_quant + avg_overhead_bpp_quant:.4f}',
                        f'{avg_bpp_encoded:.4f}',
                        f'{overlap_h}',
                        f'{overlap_w}',
                        f'{num_patches_total}',
                        f'{blend_overlap}'
                    ]
                    f.write(','.join(row) + '\n')
            
            self.log(f'Full-res overlapping patches eval {dataset_name} saved to {csv_path} and {csv_path_per_video}')    

    def _init_clip_metrics(self):
        """Initialize metrics dictionary for a clip"""
        # Common metrics across all residual types
        metrics = {
            'bits_quant':0,
            'bits_quant_overhead': 0,
        }
        
        base = {
            'psnr': [],
            'ssim': [],
            'bits_encoded_tubelet': [],
            'enc_fps': [],
            'dec_fps': [],
        }
        metrics.update({k: deepcopy(base) for k in ['from_first', 'from_prev', 'direct']})
        
        return metrics
    
    def _init_clip_metrics_across_videos(self):
        """Initialize metrics dictionary for a clip"""
        # Common metrics across all residual types
        metrics = {
            'bits_quant':[],
            'bits_quant_overhead': [],
        }
        
        base = {
            'psnr': [],
            'ssim': [],
            'bits_encoded_tubelet': [],
            'enc_fps': [],
            'dec_fps': [],
        }
        metrics.update({k: deepcopy(base) for k in ['from_first', 'from_prev', 'direct']})
        
        return metrics
    
    def evaluate_x_dict_residuals_epoch(self, quant_bit=8, quant_axis=0, 
                                                encoding_type='arithmetic', 
                                                eval_csv_prefix='', log_per_video=True,
                                                chunk_pred_batch_size=None):
        """
        Evaluate model performance with overlapping patches for cases where patch
        size does not perfectly divide the full resolution.
        # Args:
        #     overlap_h: Height overlap in pixels
        #     overlap_w: Width overlap in pixels
        """
        if encoding_type not in [None, 'arithmetic', 'huffman']:
            raise ValueError(f"Invalid encoding_type: {encoding_type}")
        
        overlap_h = self.cfg.get('overlap_h', 0)
        overlap_w = self.cfg.get('overlap_w', 0)
        blend_overlap = self.cfg.get('blend_overlap', False)

        self.model_ddp.eval()
        metrics_per_video = {}
        
        # can be extended to evaluate multiple datasets
        dataset_name, loader = next(iter(self.test_loader_dict.items()))
        dataset = loader.dataset
        videos = dataset.vid_list if hasattr(dataset, 'vid_list') else []
        clip_metrics_across_videos = self._init_clip_metrics_across_videos()
        
        self.log(f'overlapping patches eval: Processing {dataset_name}')
        self.log(f'Using adaptive overlaps (h={overlap_h}, w={overlap_w}), chunking predictions in batches of {chunk_pred_batch_size}')
        self.log(f'blend_overlap={blend_overlap}')

        for video in tqdm(videos):
            ordered_dataset = self._make_ordered_dataset_overlapping(dataset, video, overlap_h, overlap_w)
            ordered_loader = self._make_data_loader(ordered_dataset, batch_size=1, num_workers=loader.num_workers)
            
            first_x_dicts_recon_per_pos = {}
            prev_x_dicts_recon_per_pos = {}
            video_metrics = defaultdict(list)
            print_flag = True

            for clip_idx, data in enumerate(ordered_loader):
                data = {k: v[0].cuda() if isinstance(v, torch.Tensor) else v 
                    for k, v in data.items()}
                
                patches = data['patches']
                positions = data['patch_positions']
                start_frame = data['start_frame']
                
                # Process patch-tubelets
                start_common_encode = time.time()
                all_patches = [[] for _ in range(len(patches[0]))]
                for frame_patches in patches:
                    for i, patch in enumerate(frame_patches):
                        all_patches[i].append(patch.squeeze(0))
                        
                if print_flag:
                    self.log(f'Processing {len(all_patches)} overlapping patches')
                    print_flag = False
                    
                all_patches = [torch.stack(patch_seq, dim=1) for patch_seq in all_patches]
                x_dict_all, _ = self._process_patches_batch_chunked(all_patches, chunk_pred_batch_size)
                t_common_encode = time.time() - start_common_encode

                clip_metrics = self._init_clip_metrics()
                x_dict_recon_patches = {'direct': [], 'from_first': [], 'from_prev': []}

                clip_t_compress = {
                    'encode': {'direct': 0, 'from_first': 0, 'from_prev': 0},
                    'decode': {'direct': 0, 'from_first': 0, 'from_prev': 0}
                }

                # Store all bits from all patches (not averaged per patch)
                current_clip_total_bits_encoded_tubelet_direct = 0
                current_clip_total_bits_encoded_tubelet_first = 0
                current_clip_total_bits_encoded_tubelet_prev = 0
                current_clip_total_bits_quant = 0
                current_clip_total_bits_overhead = 0

                # Process each patch position
                for pos_idx, pos_tuple in enumerate(positions[0]):
                    pos_key = tuple(p.item() for p in pos_tuple)
                    cur_x_dict = {
                        k: v[pos_idx:pos_idx+1].cuda()
                        if v is not None else None
                        for k, v in x_dict_all.items()
                    }
                    
                    # Direct processing
                    x_dict_recon_direct, bits_direct, t_enc_direct, t_dec_direct = self._process_direct(
                        cur_x_dict, quant_bit, quant_axis, encoding_type)
                    x_dict_recon_patches['direct'].append(x_dict_recon_direct)
                    
                    # Accumulate total bits for tubelet - includes excessive bits from overlapping patches
                    current_clip_total_bits_encoded_tubelet_direct += bits_direct['encoded']
                    clip_t_compress['encode']['direct'] += t_enc_direct
                    clip_t_compress['decode']['direct'] += t_dec_direct

                    if start_frame == 0:
                        x_dict_recon_first = x_dict_recon_direct
                        x_dict_recon_prev = x_dict_recon_direct
                        
                        first_x_dicts_recon_per_pos[pos_key] = self._clone_dict(x_dict_recon_first) 
                        prev_x_dicts_recon_per_pos[pos_key] = self._clone_dict(x_dict_recon_prev)

                        x_dict_recon_patches['from_first'].append(x_dict_recon_first)
                        x_dict_recon_patches['from_prev'].append(x_dict_recon_prev)
                        current_clip_total_bits_encoded_tubelet_first += bits_direct['encoded']
                        current_clip_total_bits_encoded_tubelet_prev += bits_direct['encoded']

                        clip_t_compress['encode']['from_first'] += t_enc_direct
                        clip_t_compress['decode']['from_first'] += t_dec_direct
                        clip_t_compress['encode']['from_prev'] += t_enc_direct
                        clip_t_compress['decode']['from_prev'] += t_dec_direct
                        
                        # Store total quant/overhead bits for the first clip
                        current_clip_total_bits_quant += bits_direct['quant']
                        current_clip_total_bits_overhead += bits_direct['overhead']

                    else:
                        if pos_key not in first_x_dicts_recon_per_pos or pos_key not in prev_x_dicts_recon_per_pos:
                            raise RuntimeError(f"Missing state for position {pos_key}...")

                        # 'from_first'
                        x_dict_recon_first, enc_bits_first, t_enc_p_first, t_dec_p_first = self._process_residual(
                            cur_x_dict, first_x_dicts_recon_per_pos[pos_key], quant_bit, quant_axis,
                            encoding_type)
                        x_dict_recon_patches['from_first'].append(x_dict_recon_first)
                        current_clip_total_bits_encoded_tubelet_first += enc_bits_first
                        clip_t_compress['encode']['from_first'] += t_enc_p_first
                        clip_t_compress['decode']['from_first'] += t_dec_p_first

                        # 'from_prev'
                        x_dict_recon_prev, enc_bits_prev, t_enc_p_prev, t_dec_p_prev = self._process_residual(
                            cur_x_dict, prev_x_dicts_recon_per_pos[pos_key], quant_bit, quant_axis,
                            encoding_type)
                        x_dict_recon_patches['from_prev'].append(x_dict_recon_prev)
                        current_clip_total_bits_encoded_tubelet_prev += enc_bits_prev
                        clip_t_compress['encode']['from_prev'] += t_enc_p_prev
                        clip_t_compress['decode']['from_prev'] += t_dec_p_prev

                        prev_x_dicts_recon_per_pos[pos_key] = self._clone_dict(x_dict_recon_prev)
                
                clip_metrics['bits_quant'] = current_clip_total_bits_quant
                clip_metrics['bits_quant_overhead'] = current_clip_total_bits_overhead
                clip_metrics['direct']['bits_encoded_tubelet'].append(current_clip_total_bits_encoded_tubelet_direct)
                clip_metrics['from_first']['bits_encoded_tubelet'].append(current_clip_total_bits_encoded_tubelet_first)
                clip_metrics['from_prev']['bits_encoded_tubelet'].append(current_clip_total_bits_encoded_tubelet_prev)

                # Reconstruct full frames with cropping or blending in overlapping patches
                for method in ['direct', 'from_first', 'from_prev']:
                    start_hyponet_tile = time.time()
                    recon_batch = self._combine_batch_reconstructions(x_dict_recon_patches[method])
                    num_patches_processed = len(x_dict_recon_patches[method])
                    if num_patches_processed != len(positions[0]):
                        raise ValueError(f"Batch size mismatch for {method}")

                    recon_patches = self.reconstruct_from_x_dict_chunked(
                        num_patches_processed, dataset.frame_num, recon_batch, 
                        device=data['gt'].device, batch_size=chunk_pred_batch_size
                    )
                    
                    # Use overlapping patch tiling instead of regular tiling
                    if overlap_h == 0 and overlap_w == 0:
                        recon_clip = self._tile_clip_from_patches(recon_patches, positions, data['metadata'])
                    else:
                        # Use overlapping patch tiling
                        if blend_overlap:
                            recon_clip = self._tile_clip_from_overlapping_patches_with_blending(
                                recon_patches, positions, data['metadata'])
                        else:
                            recon_clip = self._tile_clip_from_overlapping_patches_with_cropping(
                                recon_patches, positions, data['metadata'])
                    t_hyponet_tile = time.time() - start_hyponet_tile

                    # Calculate total clip times
                    total_encode_time_clip = t_common_encode + clip_t_compress['encode'][method]
                    total_decode_time_clip = clip_t_compress['decode'][method] + t_hyponet_tile

                    clip_metrics[method]['enc_fps'].append(dataset.frame_num / total_encode_time_clip)
                    clip_metrics[method]['dec_fps'].append(dataset.frame_num / total_decode_time_clip)

                    # Compute PSNR, SSIM on full resolution
                    clip_metrics[method] = self._compute_full_clip_metrics(
                        clip_metrics[method], recon_clip, data['gt']
                    )
        
                # Aggregate metrics
                for method in ['direct', 'from_first', 'from_prev']:
                    for metric in ['psnr', 'ssim']:
                        video_metrics[f'{method}_{metric}'].extend(clip_metrics[method][metric])
                    for metric in ['enc_fps', 'dec_fps']:
                        video_metrics[f'{method}_{metric}'].extend(clip_metrics[method][metric])
                    video_metrics[f'{method}_bits_encoded_tubelet'].extend(clip_metrics[method]['bits_encoded_tubelet'])
                
                video_metrics['bits_quant'].append(clip_metrics['bits_quant'])
                video_metrics['bits_quant_overhead'].append(clip_metrics['bits_quant_overhead'])

                clip_metrics_across_videos = self._update_clip_metrics_across_videos(clip_metrics_across_videos, clip_metrics)


            metrics_per_video[video] = video_metrics

        if self.is_master:
            prefix = ''
            if isinstance(eval_csv_prefix, str):
                if eval_csv_prefix.strip().lower() not in ['', 'none', 'null']:
                    prefix = f'{eval_csv_prefix.strip()}_'

            csv_path = os.path.join(
                self.cfg["eval_metrics_path"],
                f'{prefix}eval_full_res_residuals_overlap_{dataset_name}.csv',
            )
            csv_path_per_video = os.path.join(
                self.cfg["eval_metrics_path"],
                f'{prefix}eval_per_vid_full_res_residuals_overlap_{dataset_name}.csv',
            )
            
            self._log_full_res_metrics_overlapping(
                'x_dict', dataset_name, csv_path, csv_path_per_video, metrics_per_video, clip_metrics_across_videos, 
                ordered_dataset, videos, encoding_type, quant_bit, quant_axis, overlap_h, overlap_w, blend_overlap, 
                eval_csv_prefix, log_per_video=log_per_video
            )
    
        return metrics_per_video, csv_path
