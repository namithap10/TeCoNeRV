import os

import numpy as np
import torch
from tqdm import tqdm

from trainers import register
from trainers.nerv_enc_trainer import NeRVEncTrainer


@register("nerv_enc_trainer_weight_saver")
class NeRVEncTrainerWeightSaver(NeRVEncTrainer):

    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)

    def save_weights_no_quant(self, save_path=None, residual_type='direct', target_vid='jockey'):
        """
        Save predicted x_dict weights for NeRVEnc.

        For each target video:
        - Store base params
        - For each clip: Store the predicted x_dict weights
        """
        if residual_type != 'direct':
            raise ValueError(f"Invalid weights type: {residual_type} for no patch model")
        
        self.model_ddp.eval()
        
        root_weights_dir = save_path
        os.makedirs(root_weights_dir, exist_ok=True)
        
        # Save base params
        if hasattr(self.model_ddp, "module"):
            base_params = self.model_ddp.module.base_params
        else:
            base_params = self.model_ddp.base_params
        torch.save(base_params, os.path.join(root_weights_dir, "base_params.pth"))
        
        for dataset_name, loader in self.test_loader_dict.items():
            dataset = loader.dataset
            videos = dataset.vid_list if hasattr(dataset, 'vid_list') else []
            
            # Filter videos to only process target videos
            if isinstance(target_vid, str):
                target_videos = [v for v in videos if target_vid in v.lower()]
            elif isinstance(target_vid, list):
                target_videos = [v for v in videos if any(t in v.lower() for t in target_vid)]
                
            if not target_videos:
                self.log(f"No target videos found in {dataset_name}")
                continue
                
            self.log(f'Processing {len(target_videos)} target videos in {dataset_name}')
            
            for video in tqdm(target_videos):
                # Create video-specific directory
                vid_name = video.split('/')[-1].replace('_1080', '')
                video_dir = os.path.join(root_weights_dir, residual_type, vid_name)
                os.makedirs(video_dir, exist_ok=True)
                batch_size = loader.batch_size
                
                # Initialize metrics for this video
                video_psnr = []
                video_ssim = []
                
                # Create ordered dataset and loader for this video
                ordered_dataset = self._make_ordered_dataset(dataset, video)
                ordered_loader = self._make_data_loader(
                    ordered_dataset, batch_size=batch_size, num_workers=loader.num_workers
                )
                
                for batch_idx, data in enumerate(ordered_loader):
                    start_frames = data.pop("start_frame")
                    data = {k: v.cuda() for k, v in data.items() if k != "name"}
                    
                    # Get x_dict from model forward pass
                    with torch.no_grad():
                        output = self.model_ddp(data)
                        x_dict = output.get('pre_mod', output.get('x_dict'))
                    
                    batch_size = next(v.shape[0] for v in x_dict.values() if v is not None)
                    
                    # Process each clip in the batch
                    for i in range(batch_size):
                        # Get clip-specific x_dict
                        clip_x_dict = {k: v[i:i+1] if v is not None else None for k, v in x_dict.items()}
                        
                        # Save clip weights
                        clip_name = f"{vid_name}_clip_{batch_idx*batch_size + i}_frame_{start_frames[i]}"
                        torch.save(clip_x_dict, os.path.join(video_dir, f"{clip_name}.pth"))
                        
                        # Compute PSNR and SSIM for sanity check
                        _, psnr, ssim_val = self.reconstruct_from_x_dict(data, clip_x_dict)
                        video_psnr.append(psnr.item())
                        video_ssim.append(ssim_val.item())
                
                # Log average metrics for this video
                avg_psnr = np.mean(video_psnr)
                avg_ssim = np.mean(video_ssim)
                self.log(f"{video.split('/')[-1]}: avg_psnr={avg_psnr:.4f}, avg_ssim={avg_ssim:.4f}")
                
        self.log(f"Dumped weights at {root_weights_dir}")

    def save_weights_quant(self, save_path=None, residual_type='direct', quant_axis=0, target_vid='jockey'):
        """
        Save predicted x_dict weights for NeRVEnc, with quantization.

        For each target video:
        - Store base_params
        - For each clip: Store the predicted x_dict weights
        """
        if residual_type != 'direct':
            raise ValueError(f"Invalid weights type: {residual_type} for no patch model")
        
        quant_bit = 8
        
        self.model_ddp.eval()
        
        # Create root directory for weights if it doesn't exist
        root_weights_dir = save_path
        os.makedirs(root_weights_dir, exist_ok=True)
        
        # Save base params
        if hasattr(self.model_ddp, "module"):
            base_params = self.model_ddp.module.base_params
        else:
            base_params = self.model_ddp.base_params
        torch.save(base_params, os.path.join(root_weights_dir, "base_params.pth"))
        
        for dataset_name, loader in self.test_loader_dict.items():
            dataset = loader.dataset
            videos = dataset.vid_list if hasattr(dataset, 'vid_list') else []
            
            # Filter videos to only process target videos
            if isinstance(target_vid, str):
                target_videos = [v for v in videos if target_vid in v.lower()]
            elif isinstance(target_vid, list):
                target_videos = [v for v in videos if any(t in v.lower() for t in target_vid)]
            
            if not target_videos:
                self.log(f"No target videos found in {dataset_name}")
                continue
                
            self.log(f'Processing {len(target_videos)} target videos in {dataset_name}')
            
            for video in tqdm(target_videos):
                # Create video-specific directory
                vid_name = video.split('/')[-1].replace('_1080', '')
                video_dir = os.path.join(root_weights_dir, residual_type, vid_name)
                os.makedirs(video_dir, exist_ok=True)
                batch_size = loader.batch_size
                
                # Initialize metrics for this video
                video_psnr = []
                video_ssim = []
                
                # Create ordered dataset and loader for this video
                ordered_dataset = self._make_ordered_dataset(dataset, video)
                ordered_loader = self._make_data_loader(
                    ordered_dataset, batch_size=batch_size, num_workers=loader.num_workers
                )
                
                for batch_idx, data in enumerate(ordered_loader):
                    start_frames = data.pop("start_frame")
                    data = {k: v.cuda() for k, v in data.items() if k != "name"}
                    
                    # Get x_dict from model forward pass
                    with torch.no_grad():
                        output = self.model_ddp(data)
                        x_dict = output.get('pre_mod', output.get('x_dict'))
                    
                    batch_size = next(v.shape[0] for v in x_dict.values() if v is not None)
                    
                    # Process each clip in the batch
                    for i in range(batch_size):
                        # Get clip-specific x_dict
                        clip_x_dict = {k: v[i:i+1] if v is not None else None for k, v in x_dict.items()}
                        
                        # Quantize the clip x_dict
                        quantized_dict, scales, t_mins = self.quantize_param_dict(
                            clip_x_dict, quant_bit=quant_bit, axis=quant_axis)
                        
                        # Dequantize the clip x_dict
                        dequantized_dict = self.recover_param_dict_from_quantized_residuals(
                            None, quantized_dict, scales, t_mins)
                        
                        # Save dequantized clip weights
                        clip_name = f"{vid_name}_clip_{batch_idx*batch_size + i}_frame_{start_frames[i]}"
                        torch.save(dequantized_dict, os.path.join(video_dir, f"{clip_name}.pth"))
                        
                        # Compute PSNR and SSIM for sanity check
                        _, psnr, ssim_val = self.reconstruct_from_x_dict(data, dequantized_dict)
                        video_psnr.append(psnr.item())
                        video_ssim.append(ssim_val.item())
                
                # Log average metrics for this video
                avg_psnr = np.mean(video_psnr)
                avg_ssim = np.mean(video_ssim)
                
        self.log(f"Dumped quantized weights at {root_weights_dir}")
