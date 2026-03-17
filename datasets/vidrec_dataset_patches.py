import os
import random
import warnings

import decord
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandAugment

from datasets import register

decord.bridge.set_bridge('torch')
try:
    from video_reader import PyVideoReader
except Exception as exc:
    PyVideoReader = None
    warnings.warn(
        f"Falling back to decord.VideoReader because video_reader could not be imported: {exc}",
        RuntimeWarning,
    )

from .vidrec_dataset import VideoTransform, handle_size_param

def _make_video_reader(video_path):
    if PyVideoReader is not None:
        return PyVideoReader(video_path)
    return decord.VideoReader(video_path)

def _ensure_video_tensor(video):
    # decord with torch bridge returns a tensor but PyVideoReader returns a numpy array
    if torch.is_tensor(video):
        return video
    return torch.from_numpy(video)



@register('vidrec_dataset_patch_tubelet_sampler_lazy')
class VideoDataSetPatchTubeletSamplerLazy(Dataset):
    """
    Dataset for lazy sampling of patches from a video with better distributed training support.
    Each worker selects a subset of the dataset based on rank.
    """
    def __init__(self, root_path, frame_num, cls_vid_num, crop_size, tubelet_size, rand_flip='no',
        split='train', csv_file='', scale=1, aspect_ratio=1, rand_augment='no', clips_per_video=1, tubelets_per_clip=None):
        self.crop_size = handle_size_param(crop_size)
        self.tubelet_size = handle_size_param(tubelet_size)
        self.tubelets_per_clip = tubelets_per_clip
        
        # Calculate number of patch-tubelets per frame
        self.patches_per_row = self.crop_size[0] // self.tubelet_size[0]
        self.patches_per_col = self.crop_size[1] // self.tubelet_size[1]
        self.patches_per_frame = self.patches_per_row * self.patches_per_col
        
        # Load video list
        if csv_file != '':
            self.cls_list = None
            csv_file = os.path.join(root_path, csv_file)
            if csv_file.endswith('.csv'):
                import pandas as pd
                self.vid_list = pd.read_csv(csv_file)['path'].tolist()
            elif csv_file.endswith('.js'):
                import json
                with open(csv_file, 'r') as f:
                    vid_dict = json.load(f)
                # Try to load precomputed lengths to speed up training
                lengths_file = os.path.splitext(csv_file)[0] + '_lengths.json'
                try:
                    with open(lengths_file, 'r') as f:
                        self.vid_lengths_dict = json.load(f)
                    print(f"Loaded precomputed video lengths from {lengths_file}")
                except:
                    print(f"No precomputed lengths file found at {lengths_file}")
                    self.vid_lengths_dict = None
                    
                cls_num, vid_num = [int(x) for x in cls_vid_num.split('_')]
                sorted_keys=sorted(vid_dict, key=lambda k: len(vid_dict[k]), reverse=True)
                vid_list = [vid_dict[cls][:vid_num] for cls in sorted_keys[:cls_num]]
                self.vid_list = sum(vid_list, [])
        else:
            self.vid_list = []
            cls_num, vid_num = [int(x) for x in cls_vid_num.split('_')]
            root_path = os.path.join(root_path, split)
            for cur_cls in sorted(os.listdir(root_path)[:cls_num]):
                cur_dir = os.path.join(root_path, cur_cls)
                for cur_vid in sorted(os.listdir(cur_dir))[:vid_num]:
                    self.vid_list.append(os.path.join(cur_dir, cur_vid))
                    
        self.split = split
        self.frame_num = frame_num
        self.rand_flip = rand_flip
        self.scale = scale
        self.aspect_ratio = aspect_ratio
        
        if rand_augment in ['no', '']:
            self.augment = None
        else:
            num_ops, magnitude, num_magnitude_bins = [int(x) for x in rand_augment.split('_')]
            self.augment = RandAugment(num_ops, magnitude, num_magnitude_bins)

        # Setup clip info to avoid pre-generating all indices
        self.clips_per_video = clips_per_video
        self.vid_lens = {}
        self.start_frames_dict = {}
        
        for idx, vid_path in enumerate(self.vid_list):
            try:
                if self.vid_lengths_dict is not None:
                    vid_name = os.path.basename(vid_path)
                    for cls_name, videos in self.vid_lengths_dict.items():
                        if vid_name in videos:
                            length = videos[vid_name]
                            if length > 0:
                                self.vid_lens[idx] = length
                                break
                
                # If valid length is not found in precomputed dict, use decord
                if idx not in self.vid_lens:
                    vr = _make_video_reader(vid_path)
                    self.vid_lens[idx] = len(vr)
                
                # Find valid start frames to avoid storing all of them
                valid_starts = []
                for start in range(0, self.vid_lens[idx], self.frame_num):
                    if start + self.frame_num <= self.vid_lens[idx]:
                        valid_starts.append(start)
                self.start_frames_dict[idx] = valid_starts
            except Exception as e:
                print(f"Error loading video {vid_path}: {e}")
                # Add placeholder for failed video
                self.vid_lens[idx] = 0
                self.start_frames_dict[idx] = []
                
        # Calculate total dataset size
        if self.tubelets_per_clip is not None:
            tubelets_per_clip = min(self.patches_per_frame, self.tubelets_per_clip)
        else:
            tubelets_per_clip = self.patches_per_frame
            
        valid_vids = [idx for idx in range(len(self.vid_list)) 
                     if len(self.start_frames_dict[idx]) > 0]
        self.dataset_size = len(valid_vids) * self.clips_per_video * tubelets_per_clip
        
        self.rng = random.Random(42)
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Calculate video index, clip index, and tubelet index
        if self.tubelets_per_clip is not None:
            tubelets_per_clip = min(self.patches_per_frame, self.tubelets_per_clip)
        else:
            tubelets_per_clip = self.patches_per_frame
            
        vid_idx = (idx // (self.clips_per_video * tubelets_per_clip)) % len(self.vid_list)
        clip_idx = (idx // tubelets_per_clip) % self.clips_per_video
        tubelet_idx = idx % tubelets_per_clip
        
        # Skip videos with no valid frames
        if len(self.start_frames_dict[vid_idx]) == 0:
            # Use the first valid video as fallback
            for alt_idx in range(len(self.vid_list)):
                if len(self.start_frames_dict[alt_idx]) > 0:
                    vid_idx = alt_idx
                    break
                    
        # Use deterministic random selection based on indices
        # This ensures same video is selected with same indices across workers
        seed = hash((vid_idx, clip_idx)) % 10000
        local_rng = random.Random(seed)
        
        # Select start frame
        start_frame = local_rng.choice(self.start_frames_dict[vid_idx])
        
        # Select tubelet position deterministically based on tubelet_idx
        possible_tubelets = list(range(self.patches_per_frame))
        local_rng.shuffle(possible_tubelets)
        actual_tubelet_idx = possible_tubelets[tubelet_idx % len(possible_tubelets)]
        
        # Calculate tubelet grid position
        tubelet_row = actual_tubelet_idx // self.patches_per_col
        tubelet_col = actual_tubelet_idx % self.patches_per_col
        
        # Load the video data on demand
        try:
            vr = _make_video_reader(self.vid_list[vid_idx])
            frame_num = min(self.frame_num, len(vr) - start_frame)
            frame_idx = [int(x + start_frame) for x in range(frame_num)]
            video = vr.get_batch(frame_idx)
            video = _ensure_video_tensor(video)
            vid_name = f'{self.vid_list[vid_idx].split("/")[-1].split(".")[0]}_{start_frame}'
            
            if self.augment is not None:
                video = self.augment(video.permute(0,-1,1,2)).permute(0,2,3,1)
            video = video.permute(-1,0,1,2).float() / 255. # T,H,W,C -> C,T,H,W
            
            if self.split == 'train':
                cur_tfm = VideoTransform(crop_size=self.crop_size, scale=self.scale, 
                    ratio=self.aspect_ratio, eval_tfm=False)
            else:
                cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
                
            video_data = cur_tfm(video)
            video_data = F.pad(video_data, (0,0,0,0,0,self.frame_num-frame_num), mode='replicate')
            
            # Extract tubelet
            h_start = tubelet_row * self.tubelet_size[0]
            w_start = tubelet_col * self.tubelet_size[1]
            tubelet = video_data[:, :, h_start:h_start+self.tubelet_size[0], w_start:w_start+self.tubelet_size[1]]
            
            metadata = {
                'video_name': vid_name,
                'start_frame': start_frame,
                'tubelet_row': tubelet_row,
                'tubelet_col': tubelet_col,
                'patches_per_row': self.patches_per_row,
                'patches_per_col': self.patches_per_col
            }
            
            return {'inp': tubelet, 'gt': tubelet, 'name': vid_name, 'metadata': metadata}
            
        except Exception as e:
            print(f"Error processing video at index {idx}: {e}")
            raise RuntimeError(f"Failed to process video at index {idx}") from e        

@register('vidrec_dataset_patch_tubelet_inference_lazy')
class VideoDataSetPatchTubeletInferenceLazy(Dataset):
    """
    Dataset for lazy sampling of patch-tubelets from a video for evaluation.
    """
    def __init__(self, root_path, frame_num, cls_vid_num, crop_size, tubelet_size, rand_flip='no',
        split='test', csv_file='', scale=1, aspect_ratio=1, rand_augment='no'):
        
        self.crop_size = handle_size_param(crop_size)
        self.tubelet_size = handle_size_param(tubelet_size)
        
        # Calculate number of patch-tubelets per frame
        self.patches_per_row = self.crop_size[0] // self.tubelet_size[0]
        self.patches_per_col = self.crop_size[1] // self.tubelet_size[1]
        self.patches_per_frame = self.patches_per_row * self.patches_per_col
        
        # Load video list
        if csv_file != '':
            self.cls_list = None
            csv_file = os.path.join(root_path, csv_file)
            if csv_file.endswith('.csv'):
                import pandas as pd
                self.vid_list = pd.read_csv(csv_file)['path'].tolist()
            elif csv_file.endswith('.js'):
                import json
                with open(csv_file, 'r') as f:
                    vid_dict = json.load(f)
                cls_num, vid_num = [int(x) for x in cls_vid_num.split('_')]
                sorted_keys=sorted(vid_dict, key=lambda k: len(vid_dict[k]), reverse=True)
                vid_list = [vid_dict[cls][:vid_num] for cls in sorted_keys[:cls_num]]
                self.vid_list = sum(vid_list, [])
        else:
            self.vid_list = []
            cls_num, vid_num = [int(x) for x in cls_vid_num.split('_')]
            root_path = os.path.join(root_path, split)
            for cur_cls in sorted(os.listdir(root_path)[:cls_num]):
                cur_dir = os.path.join(root_path, cur_cls)
                for cur_vid in sorted(os.listdir(cur_dir))[:vid_num]:
                    self.vid_list.append(os.path.join(cur_dir, cur_vid))
        
        self.split = split
        self.frame_num = frame_num
        self.rand_flip = rand_flip
        self.scale = scale
        self.aspect_ratio = aspect_ratio
        
        if rand_augment in ['no', '']:
            self.augment = None
        else:
            num_ops, magnitude, num_magnitude_bins = [int(x) for x in rand_augment.split('_')]
            self.augment = RandAugment(num_ops, magnitude, num_magnitude_bins)
            
        # Generate all possible sample indices, but do not load data yet
        self.samples = []
        
        for vid_idx, vid_path in enumerate(self.vid_list):
            try:
                vr = decord.VideoReader(vid_path)
                vid_len = len(vr)
                vid_name = vid_path.split('/')[-1].split('.')[0]
                
                for start_frame in range(0, vid_len, self.frame_num):
                    if start_frame + self.frame_num > vid_len:
                        break
                        
                    # For each tubelet position
                    for tubelet_row in range(self.patches_per_row):
                        for tubelet_col in range(self.patches_per_col):
                            sample_info = {
                                'vid_idx': vid_idx,
                                'vid_path': vid_path,
                                'vid_name': vid_name,
                                'start_frame': start_frame,
                                'tubelet_row': tubelet_row,
                                'tubelet_col': tubelet_col
                            }
                            self.samples.append(sample_info)
            except Exception as e:
                print(f"Error scanning video {vid_path}: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            vr = decord.VideoReader(sample['vid_path'])
            start_frame = sample['start_frame']
            tubelet_row = sample['tubelet_row']
            tubelet_col = sample['tubelet_col']
            
            frame_idx = [int(x + start_frame) for x in range(self.frame_num)]
            video = vr.get_batch(frame_idx)
            video = _ensure_video_tensor(video)
            video = video.permute(-1,0,1,2).float() / 255. # T,H,W,C -> C,T,H,W
            
            # Use test transform as this is for evaluation
            cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
            video_data = cur_tfm(video)
            
            # Extract tubelet at the specified grid position
            h_start = tubelet_row * self.tubelet_size[0]
            w_start = tubelet_col * self.tubelet_size[1]
            tubelet = video_data[:, :, h_start:h_start+self.tubelet_size[0], w_start:w_start+self.tubelet_size[1]]
            
            metadata = {
                'video_name': sample['vid_name'],
                'start_frame': start_frame,
                'tubelet_row': tubelet_row,
                'tubelet_col': tubelet_col,
                'patches_per_row': self.patches_per_row,
                'patches_per_col': self.patches_per_col,
                'full_frame_size': self.crop_size
            }
            
            return {'inp': tubelet, 'gt': tubelet, 'name': sample['vid_name'], 'metadata': metadata}
            
        except Exception as e:
            print(f"Error processing video at index {idx}: {e}")
            raise RuntimeError(f"Failed to process video at index {idx}") from e

@register('vidrec_dataset_full_res_patches_single_video')
class VideoDataSetFullResPatchesSingleVideo(Dataset):
    """
    Dataset for processing a single video in full-resolution with patches.
    Each clip is frame_num frames long and identified by start_frame.
    Clips are returned in sequential order.
    For each clip, returns both the full frame and the patches.
    """
    def __init__(self, video_path, frame_num, crop_size, tubelet_size):
        self.video_path = video_path
        self.frame_num = frame_num
        
        self.crop_size = handle_size_param(crop_size)
        self.tubelet_size = handle_size_param(tubelet_size)
        
        # Get video info
        vr = decord.VideoReader(video_path)
        self.total_frames = len(vr)
        self.num_clips = self.total_frames // self.frame_num
        if self.total_frames % self.frame_num != 0:
            self.num_clips += 1  # Include the last shorter clip
            
        self.vid_name = os.path.basename(video_path).split('.')[0] # More robust split
        
        # Calculate number of patch-tubelets per frame
        self.patches_per_row = (self.crop_size[0] + self.tubelet_size[0] - 1) // self.tubelet_size[0]
        self.patches_per_col = (self.crop_size[1] + self.tubelet_size[1] - 1) // self.tubelet_size[1]
        self.num_patches = self.patches_per_row * self.patches_per_col

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start_frame = idx * self.frame_num
        vr = decord.VideoReader(self.video_path)
        
        # Handle the last clip which might be shorter
        actual_frame_num = self.frame_num
        if start_frame + self.frame_num > self.total_frames:
            frame_idx = list(range(start_frame, self.total_frames))
            # Pad with last frame if necessary
            actual_frame_num = len(frame_idx)
            frame_idx.extend([self.total_frames - 1] * (self.frame_num - len(frame_idx)))
        else:
            frame_idx = list(range(start_frame, start_frame + self.frame_num))
            
        video = vr.get_batch(frame_idx)
        video = video.permute(-1,0,1,2).float() / 255.  # T,H,W,C -> C,T,H,W
        
        # Use test transform as this is for evaluation
        cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
        video_data = cur_tfm(video)  # C,T,H,W
        
        # Create patches from the full frame to be used as ground truth
        patches = []
        patch_positions = []
        
        for t in range(video_data.shape[1]):  # For each frame in the clip
            frame = video_data[:, t]  # C,H,W
            frame_patches = []
            frame_positions = []
            
            for row in range(self.patches_per_row):
                for col in range(self.patches_per_col):
                    # Calculate patch coordinates
                    h_start = row * self.tubelet_size[0]
                    w_start = col * self.tubelet_size[1]
                    
                    h_end = min(h_start + self.tubelet_size[0], self.crop_size[0])
                    w_end = min(w_start + self.tubelet_size[1], self.crop_size[1])
                    
                    # Extract patch
                    patch = frame[:, h_start:h_end, w_start:w_end]
                    
                    # Pad if necessary
                    if patch.shape[1] < self.tubelet_size[0] or patch.shape[2] < self.tubelet_size[1]:
                        pad_h = self.tubelet_size[0] - patch.shape[1]
                        pad_w = self.tubelet_size[1] - patch.shape[2]
                        patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='replicate')
                    
                    frame_patches.append(patch)
                    frame_positions.append((h_start, w_start, h_end, w_end))
            
            patches.append(frame_patches)
            patch_positions.append(frame_positions)
        
        item_name = f"{self.vid_name}_{start_frame}"

        return {
            'inp': video_data,  # Full frame
            'gt': video_data,   # Full frame
            'name': item_name,
            'start_frame': start_frame,
            'patches': patches,  # List of patches for each frame
            'patch_positions': patch_positions,  # Positions of patches in the full frame
            'metadata': {
                'video_name': self.vid_name,
                'patches_per_row': self.patches_per_row,
                'patches_per_col': self.patches_per_col,
                'num_patches': self.num_patches,
                'tubelet_size': self.tubelet_size,
                'crop_size': self.crop_size,
                'actual_frame_num': actual_frame_num # Number of actual frames before padding
            }
        }

@register('vidrec_dataset_patch_tubelet_inference_lazy_uvg')
class VideoDataSetPatchTubeletInferenceLazyUVG(Dataset):
    """
    Modified UVG dataset that efficiently handles distributed processing and CSV input.
    Each UVG video is a sequence of PNG files in a folder.
    This class provides the same interface as VideoDataSetPatchTubeletInferenceLazy
    but works with UVG PNG sequences instead of MP4 files.
    """
    def __init__(self, root_path, frame_num, cls_vid_num, crop_size, tubelet_size, rand_flip='no',
        split='test', csv_file='', scale=1, aspect_ratio=1, rand_augment='no'):
        
        self.crop_size = handle_size_param(crop_size)
        self.tubelet_size = handle_size_param(tubelet_size)
        
        # Calculate number of patch-tubelets per frame
        self.patches_per_row = self.crop_size[0] // self.tubelet_size[0]
        self.patches_per_col = self.crop_size[1] // self.tubelet_size[1]
        self.patches_per_frame = self.patches_per_row * self.patches_per_col
        
        if csv_file != '':
            csv_file = os.path.join(root_path, csv_file)
            if csv_file.endswith('.csv'):
                self.vid_list = pd.read_csv(csv_file)['path'].tolist()
            elif csv_file.endswith('.js'):
                import json
                with open(csv_file, 'r') as f:
                    vid_dict = json.load(f)
                cls_num, vid_num = [int(x) for x in cls_vid_num.split('_')]
                sorted_keys=sorted(vid_dict, key=lambda k: len(vid_dict[k]), reverse=True)
                vid_list = [vid_dict[cls][:vid_num] for cls in sorted_keys[:cls_num]]
                self.vid_list = sum(vid_list, [])
        else:
            # Default UVG dataset folders
            self.vid_list = []
            uvg_folders = ['beauty_1080', 'bosphore_1080', 'honeybee_1080', 'jockey_1080', 'shakendry_1080', 'yachtride_1080', 'readysteadygo_1080']
            for folder in uvg_folders:
                folder_path = os.path.join(root_path, folder)
                if os.path.exists(folder_path):
                    self.vid_list.append(folder_path)
        
        self.split = split
        self.frame_num = frame_num
        self.rand_flip = rand_flip
        self.scale = scale
        self.aspect_ratio = aspect_ratio
        
        if rand_augment in ['no', '']:
            self.augment = None
        else:
            raise NotImplementedError("Augmentation not implemented for UVG dataset")
            
        # Generate all possible sample indices, but do not load data yet
        self.samples = []
        
        for vid_idx, vid_path in enumerate(self.vid_list):
            try:
                vid_name = os.path.basename(vid_path)
                
                image_files = sorted([f for f in os.listdir(vid_path) 
                                    if f.startswith('f') and f.endswith('.png')])
                if not image_files:
                    print(f"Warning: No PNG files found in {vid_path}")
                    continue
                    
                vid_len = len(image_files)
                
                # Process all clips in this video
                for start_frame in range(0, vid_len, self.frame_num):
                    if start_frame + self.frame_num > vid_len:
                        break
                        
                    # For each tubelet position
                    for tubelet_row in range(self.patches_per_row):
                        for tubelet_col in range(self.patches_per_col):
                            sample_info = {
                                'vid_idx': vid_idx,
                                'vid_path': vid_path,
                                'vid_name': vid_name,
                                'start_frame': start_frame,
                                'tubelet_row': tubelet_row,
                                'tubelet_col': tubelet_col
                            }
                            self.samples.append(sample_info)
            except Exception as e:
                print(f"Error scanning video {vid_path}: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def load_image(self, path, index):
        """Load a single image from the UVG dataset"""
        # UVG has frame indices that start with f00, typically with 3-digit numbering
        img_files = sorted([f for f in os.listdir(path) if f.startswith('f') and f.endswith('.png')])
        img_path = os.path.join(path, img_files[index])
        img = Image.open(img_path).convert('RGB')
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.
        return img
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load the frames data on demand
            vid_path = sample['vid_path']
            start_frame = sample['start_frame']
            tubelet_row = sample['tubelet_row']
            tubelet_col = sample['tubelet_col']
            
            # Load frame sequence
            frames = []
            for i in range(self.frame_num):
                frame_idx = start_frame + i
                frames.append(self.load_image(vid_path, frame_idx))
            
            video = torch.stack(frames, dim=1)  # C,T,H,W
            
            # Use test transform as this is for evaluation
            cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
            video_data = cur_tfm(video)
            
            # Extract tubelet at the specified grid position
            h_start = tubelet_row * self.tubelet_size[0]
            w_start = tubelet_col * self.tubelet_size[1]
            tubelet = video_data[:, :, h_start:h_start+self.tubelet_size[0], w_start:w_start+self.tubelet_size[1]]
            
            # Format name for compatibility with other dataset classes
            name = f"{sample['vid_name']}_{start_frame}"
            
            metadata = {
                'video_name': sample['vid_name'],
                'start_frame': start_frame,
                'tubelet_row': tubelet_row,
                'tubelet_col': tubelet_col,
                'patches_per_row': self.patches_per_row,
                'patches_per_col': self.patches_per_col,
                'full_frame_size': self.crop_size
            }
            
            return {'inp': tubelet, 'gt': tubelet, 'name': name, 'metadata': metadata}
            
        except Exception as e:
            print(f"Error processing video at index {idx}: {e}. {vid_path} start_frame: {start_frame}")
            raise RuntimeError(f"Failed to process video at index {idx}") from e

@register('vidrec_dataset_full_res_patches_single_video_uvg')
class VideoDataSetFullResPatchesSingleVideoUVG(Dataset):
    """
    Dataset for processing a single UVG video in full-resolution with patches, with lazy loading.
    Each clip is frame_num frames long and identified by start_frame.
    For each clip, returns both the full frame and the patch-tubelets.
    """
    def __init__(self, video_path, frame_num, crop_size, tubelet_size):
        self.video_path = video_path
        self.frame_num = frame_num
        
        self.crop_size = handle_size_param(crop_size)
        self.tubelet_size = handle_size_param(tubelet_size)
        
        self.image_files = sorted([f for f in os.listdir(video_path)
                                  if f.startswith('f') and f.endswith('.png')])
        if not self.image_files:
            raise ValueError(f"No matching image files found in {video_path}")
            
        self.total_frames = len(self.image_files)
        self.num_clips = self.total_frames // self.frame_num
        if self.total_frames % self.frame_num != 0:
            self.num_clips += 1  # Include last partial clip
            
        self.vid_name = os.path.basename(video_path)
        
        self.patches_per_row = (self.crop_size[0] + self.tubelet_size[0] - 1) // self.tubelet_size[0]
        self.patches_per_col = (self.crop_size[1] + self.tubelet_size[1] - 1) // self.tubelet_size[1]
        self.num_patches = self.patches_per_row * self.patches_per_col

    def load_image(self, idx):
        """Load a single image and convert to tensor"""
        img_path = os.path.join(self.video_path, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.
        return img
    
    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start_frame = idx * self.frame_num
        
        # Handle the last clip which might be shorter
        if start_frame + self.frame_num > self.total_frames:
            frame_idx = list(range(start_frame, self.total_frames))
            # Pad with last frame if needed
            frame_idx.extend([self.total_frames - 1] * (self.frame_num - len(frame_idx)))
        else:
            frame_idx = list(range(start_frame, start_frame + self.frame_num))
            
        # Load sequence of images
        video = []
        for fidx in frame_idx:
            video.append(self.load_image(fidx))
            
        video = torch.stack(video, dim=1)  # C,T,H,W
        
        cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
        video_data = cur_tfm(video)  # C,T,H,W
        
        # Create patches from the full frame
        patches = []
        patch_positions = []
        
        for t in range(video_data.shape[1]):  # For each frame in the clip
            frame = video_data[:, t]  # C,H,W
            frame_patches = []
            frame_positions = []
            
            for row in range(self.patches_per_row):
                for col in range(self.patches_per_col):
                    # Calculate patch coordinates
                    h_start = row * self.tubelet_size[0]
                    w_start = col * self.tubelet_size[1]
                    
                    h_end = min(h_start + self.tubelet_size[0], self.crop_size[0])
                    w_end = min(w_start + self.tubelet_size[1], self.crop_size[1])
                    
                    # Extract patch
                    patch = frame[:, h_start:h_end, w_start:w_end]
                    
                    # Pad if necessary (for edge patches)
                    if patch.shape[1] < self.tubelet_size[0] or patch.shape[2] < self.tubelet_size[1]:
                        pad_h = self.tubelet_size[0] - patch.shape[1]
                        pad_w = self.tubelet_size[1] - patch.shape[2]
                        patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='replicate')
                    
                    frame_patches.append(patch)
                    frame_positions.append((h_start, w_start, h_end, w_end))
            
            patches.append(frame_patches)
            patch_positions.append(frame_positions)
        
        # For compatibility with the eval function
        self.vid_list = [self.video_path]
        
        return {
            'inp': video_data,  # Full frame
            'gt': video_data,   # Full frame
            'name': f"{self.vid_name}_{start_frame}",
            'start_frame': start_frame,
            'patches': patches,  # List of patches for each frame
            'patch_positions': patch_positions,  # Positions of patches in the full frame
            'metadata': {
                'patches_per_row': self.patches_per_row,
                'patches_per_col': self.patches_per_col,
                'num_patches': self.num_patches,
                'tubelet_size': self.tubelet_size,
                'crop_size': self.crop_size,
                'video_name': self.vid_name
            }
        }

@register('vidrec_dataset_patch_tubelet_sampler_lazy_pairs')
class VideoDataSetPatchTubeletSamplerLazyPairs(Dataset):
    """
    Dataset for sampling pairs of consecutive tubelets from a video with lazy loading.
    Each item is a pair of patch-tubelets (patch over time) from the same spatial location
    but from two consecutive temporal clips.
    """
    def __init__(self, root_path, frame_num, cls_vid_num, crop_size, tubelet_size, rand_flip='no',
        split='train', csv_file='', scale=1, aspect_ratio=1, rand_augment='no', clips_per_video=1, tubelets_per_clip=None):
        
        self.crop_size = handle_size_param(crop_size)
        self.tubelet_size = handle_size_param(tubelet_size)
        self.tubelets_per_clip = tubelets_per_clip # max tubelets to sample per clip-pair location

        self.patches_per_row = self.crop_size[0] // self.tubelet_size[0]
        self.patches_per_col = self.crop_size[1] // self.tubelet_size[1]
        self.patches_per_frame = self.patches_per_row * self.patches_per_col
        
        if csv_file != '':
            self.cls_list = None
            csv_file = os.path.join(root_path, csv_file)
            if csv_file.endswith('.csv'):
                import pandas as pd
                self.vid_list = pd.read_csv(csv_file)['path'].tolist()
            elif csv_file.endswith('.js'):
                import json
                with open(csv_file, 'r') as f:
                    vid_dict = json.load(f)

                # Try to load precomputed lengths to speed up training
                lengths_file = os.path.splitext(csv_file)[0] + '_lengths.json'
                try:
                    with open(lengths_file, 'r') as f:
                        self.vid_lengths_dict = json.load(f)
                    print(f"Loaded precomputed video lengths from {lengths_file}")
                except:
                    print(f"No precomputed lengths file found at {lengths_file}")
                    self.vid_lengths_dict = None

                cls_num, vid_num = [int(x) for x in cls_vid_num.split('_')]
                sorted_keys=sorted(vid_dict, key=lambda k: len(vid_dict[k]), reverse=True)
                vid_list = [vid_dict[cls][:vid_num] for cls in sorted_keys[:cls_num]]
                self.vid_list = sum(vid_list, [])
        else:
            self.vid_list = []
            cls_num, vid_num = [int(x) for x in cls_vid_num.split('_')]
            root_path = os.path.join(root_path, split)
            for cur_cls in sorted(os.listdir(root_path)[:cls_num]):
                print(cur_cls, flush=True)
                cur_dir = os.path.join(root_path, cur_cls)
                for cur_vid in sorted(os.listdir(cur_dir))[:vid_num]:
                    print(len(os.listdir(cur_dir)), flush=True)
                    self.vid_list.append(os.path.join(cur_dir, cur_vid))

        self.split = split
        self.frame_num = frame_num
        self.rand_flip = rand_flip
        self.scale = scale
        self.aspect_ratio = aspect_ratio

        if rand_augment in ['no', '']:
            self.augment = None
        else:
            num_ops, magnitude, num_magnitude_bins = [int(x) for x in rand_augment.split('_')]
            self.augment = RandAugment(num_ops, magnitude, num_magnitude_bins)

        # Setup clip pair info - store valid start frames for the first clip of a pair
        self.clips_per_video = clips_per_video
        self.vid_lens = {}
        self.start_frames_dict = {}

        print("Scanning videos for valid consecutive clip pairs...")
        for idx, vid_path in enumerate(self.vid_list):
            try:
                # Attempt to obtain valid length from precomputed dict
                if hasattr(self, 'vid_lengths_dict') and self.vid_lengths_dict is not None:
                    vid_name = os.path.basename(vid_path)
                    # Find the class that contains this video
                    for cls_name, videos in self.vid_lengths_dict.items():
                        if vid_name in videos:
                            length = videos[vid_name]
                            if length > 0:
                                self.vid_lens[idx] = length
                                break
                
                # If valid length is not found in precomputed dict, obtain it from decord
                if idx not in self.vid_lens:
                    vr = _make_video_reader(vid_path)
                    self.vid_lens[idx] = len(vr)
                
                vid_len = self.vid_lens[idx]
                
                # Find valid start frames for the first clip, ensuring the second clip also fits
                valid_starts = []
                for start in range(0, vid_len - self.frame_num + 1, self.frame_num):
                    # Check if start + 2*frame_num is within bounds
                    if start + 2 * self.frame_num <= vid_len:
                        valid_starts.append(start)
                        
                self.start_frames_dict[idx] = valid_starts
            except Exception as e:
                print(f"Error loading video {vid_path}: {e}")
                # Add placeholder for failed video
                self.vid_lens[idx] = 0
                self.start_frames_dict[idx] = []
        
        # Calculate total dataset size based on valid pairs
        if self.tubelets_per_clip is not None:
            tubelets_per_clip_pair = min(self.patches_per_frame, self.tubelets_per_clip)
        else:
            tubelets_per_clip_pair = self.patches_per_frame

        valid_vids = [idx for idx in range(len(self.vid_list))
                     if len(self.start_frames_dict[idx]) > 0]
                     
        if not valid_vids:
             raise RuntimeError("No videos found with enough frames for consecutive clip pairs!")
             
        self.dataset_size = len(valid_vids) * self.clips_per_video * tubelets_per_clip_pair
        self.valid_vid_indices = valid_vids # Store indices of videos that have valid consecutive pairs

        self.rng = random.Random(42)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Calculate video index, clip pair index, and tubelet index
        if self.tubelets_per_clip is not None:
            tubelets_per_clip_pair = min(self.patches_per_frame, self.tubelets_per_clip)
        else:
            tubelets_per_clip_pair = self.patches_per_frame

        num_items_per_valid_vid = self.clips_per_video * tubelets_per_clip_pair
        valid_vid_list_idx = (idx // num_items_per_valid_vid) % len(self.valid_vid_indices)
        vid_idx = self.valid_vid_indices[valid_vid_list_idx]
        
        clip_pair_instance_idx = (idx // tubelets_per_clip_pair) % self.clips_per_video # Which sampling instance for this video
        tubelet_idx = idx % tubelets_per_clip_pair # Which spatial tubelet location

        # Ensure same video and start frame is selected with same indices across workers/epochs
        seed = hash((vid_idx, clip_pair_instance_idx)) % 10000 # Simple hash for seed
        local_rng = random.Random(seed)

        # Select start frame for the first clip in the pair
        start_frame = local_rng.choice(self.start_frames_dict[vid_idx])
        start_frame_next = start_frame + self.frame_num

        # Select tubelet position based on tubelet_idx for this instance
        possible_tubelets = list(range(self.patches_per_frame))
        local_rng.shuffle(possible_tubelets)
        actual_tubelet_idx = possible_tubelets[tubelet_idx % len(possible_tubelets)] # Handle tubelets_per_clip < patches_per_frame

        tubelet_row = actual_tubelet_idx // self.patches_per_col
        tubelet_col = actual_tubelet_idx % self.patches_per_col

        vr = _make_video_reader(self.vid_list[vid_idx])

        frame_idx_1 = [int(x + start_frame) for x in range(self.frame_num)]
        video1 = vr.get_batch(frame_idx_1)
        video1 = _ensure_video_tensor(video1)

        frame_idx_2 = [int(x + start_frame_next) for x in range(self.frame_num)]
        video2 = vr.get_batch(frame_idx_2)
        video2 = _ensure_video_tensor(video2)
        
        # Use start frame of the first clip for naming consistency
        vid_name = f'{self.vid_list[vid_idx].split("/")[-1].split(".")[0]}_{start_frame}'

        if self.augment is not None:
            video1 = self.augment(video1.permute(0,-1,1,2)).permute(0,2,3,1)
            video2 = self.augment(video2.permute(0,-1,1,2)).permute(0,2,3,1)

        # Preprocess: T,H,W,C -> C,T,H,W and normalize
        video1 = video1.permute(-1,0,1,2).float() / 255.
        video2 = video2.permute(-1,0,1,2).float() / 255.

        _, _, h, w = video1.shape
        target_h, target_w = self.crop_size

        if (h, w) != (target_h, target_w):
            if self.split == 'train':
                cur_tfm = VideoTransform(crop_size=self.crop_size, scale=self.scale, 
                    ratio=self.aspect_ratio, eval_tfm=False, rand_flip=self.rand_flip)
            else:
                cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True) # No random flip in eval
            video_data1 = cur_tfm(video1)
            video_data2 = cur_tfm(video2)
        else:
            cur_tfm = None
            video_data1 = video1
            video_data2 = video2

        h_start = tubelet_row * self.tubelet_size[0]
        w_start = tubelet_col * self.tubelet_size[1]
        h_end = h_start + self.tubelet_size[0]
        w_end = w_start + self.tubelet_size[1]
        
        # Ensure slicing does not go out of bounds (should not happen with // division, but safe)
        h_end = min(h_end, self.crop_size[0])
        w_end = min(w_end, self.crop_size[1])

        tubelet1 = video_data1[:, :, h_start:h_end, w_start:w_end]
        tubelet2 = video_data2[:, :, h_start:h_end, w_start:w_end]

        # Stack the patch-tubelets into a pair: [2, C, T, H_tubelet, W_tubelet]
        tubelet_pair = torch.stack([tubelet1, tubelet2], dim=0)

        metadata = {
            'video_name': vid_name, # Based on first clip's start
            'start_frame': start_frame,
            'start_frame_next': start_frame_next,
            'tubelet_row': tubelet_row,
            'tubelet_col': tubelet_col,
            'patches_per_row': self.patches_per_row,
            'patches_per_col': self.patches_per_col,
            'crop_size': self.crop_size,
            'tubelet_size': self.tubelet_size
        }

        return {'inp': tubelet_pair, 'gt': tubelet_pair, 'name': vid_name, 'metadata': metadata}

@register('vidrec_dataset_patch_tubelet_inference_lazy_pairs_uvg')
class VideoDataSetPatchTubeletInferenceLazyPairsUVG(Dataset):
    """
    Dataset for sampling pairs of consecutive tubelets from UVG videos with lazy loading.
    Each item is a pair of patch-tubelets (patch over time) from the same spatial location
    but from two consecutive temporal clips. Works with UVG PNG sequences.
    """
    def __init__(self, root_path, frame_num, cls_vid_num, crop_size, tubelet_size, rand_flip='no',
        split='train', csv_file='', scale=1, aspect_ratio=1, rand_augment='no', clips_per_video=1, tubelets_per_clip=None):
        
        self.crop_size = handle_size_param(crop_size)
        self.tubelet_size = handle_size_param(tubelet_size)
        self.tubelets_per_clip = tubelets_per_clip
        
        # Calculate number of patch-tubelets per frame
        self.patches_per_row = self.crop_size[0] // self.tubelet_size[0]
        self.patches_per_col = self.crop_size[1] // self.tubelet_size[1]
        self.patches_per_frame = self.patches_per_row * self.patches_per_col
        
        # Load video list
        csv_file = os.path.join(root_path, csv_file)
        if csv_file.endswith('.csv'):
            self.vid_list = pd.read_csv(csv_file)['path'].tolist()
        elif csv_file.endswith('.js'):
            import json
            with open(csv_file, 'r') as f:
                vid_dict = json.load(f)
            cls_num, vid_num = [int(x) for x in cls_vid_num.split('_')]
            sorted_keys=sorted(vid_dict, key=lambda k: len(vid_dict[k]), reverse=True)
            vid_list = [vid_dict[cls][:vid_num] for cls in sorted_keys[:cls_num]]
            self.vid_list = sum(vid_list, []) # beauty, bosphore, ...
        
        self.split = split
        self.frame_num = frame_num
        self.rand_flip = rand_flip
        self.scale = scale
        self.aspect_ratio = aspect_ratio
        self.clips_per_video = clips_per_video
        
        if rand_augment in ['no', '']:
            self.augment = None
        else:
            raise NotImplementedError("Augmentation not implemented for UVG dataset")
            
        # Setup clip pair info - store valid start frames for the first clip of a pair
        self.start_frames_dict = {}
        self.vid_lens = {}
        
        print("Scanning videos for valid consecutive clip pairs...")
        for idx, vid_path in enumerate(self.vid_list):
            try:
                image_files = sorted([f for f in os.listdir(vid_path) 
                                    if f.startswith('f') and f.endswith('.png')])
                if not image_files:
                    print(f"Warning: No PNG files found in {vid_path}")
                    continue
                    
                vid_len = len(image_files)
                self.vid_lens[idx] = vid_len
                
                # Find valid start frames for the first clip, ensuring the second clip also fits
                valid_starts = []
                for start in range(0, vid_len - self.frame_num + 1, self.frame_num):
                    if start + 2 * self.frame_num <= vid_len:
                        valid_starts.append(start)
                        
                self.start_frames_dict[idx] = valid_starts
                
            except Exception as e:
                print(f"Error scanning video {vid_path}: {e}")
                self.vid_lens[idx] = 0
                self.start_frames_dict[idx] = []
        
        # Calculate total dataset size based on valid consecutive pairs
        if self.tubelets_per_clip is not None:
            tubelets_per_clip_pair = min(self.patches_per_frame, self.tubelets_per_clip)
        else:
            tubelets_per_clip_pair = self.patches_per_frame
            
        valid_vids = [idx for idx in range(len(self.vid_list))
                     if len(self.start_frames_dict[idx]) > 0]
                     
        if not valid_vids:
            raise RuntimeError("No videos found with enough frames for consecutive clip pairs!")
             
        self.dataset_size = len(valid_vids) * self.clips_per_video * tubelets_per_clip_pair
        self.valid_vid_indices = valid_vids
        
        self.rng = random.Random(42)
        
    def __len__(self):
        return self.dataset_size
    
    def load_image(self, path, index):
        """Load a single image from the UVG dataset"""
        img_files = sorted([f for f in os.listdir(path) if f.startswith('f') and f.endswith('.png')])
        img_path = os.path.join(path, img_files[index])
        img = Image.open(img_path).convert('RGB')
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.
        return img
    
    def __getitem__(self, idx):
        if self.tubelets_per_clip is not None:
            tubelets_per_clip_pair = min(self.patches_per_frame, self.tubelets_per_clip)
        else:
            tubelets_per_clip_pair = self.patches_per_frame
            
        num_items_per_valid_vid = self.clips_per_video * tubelets_per_clip_pair
        valid_vid_list_idx = (idx // num_items_per_valid_vid) % len(self.valid_vid_indices)
        vid_idx = self.valid_vid_indices[valid_vid_list_idx]
        
        clip_pair_instance_idx = (idx // tubelets_per_clip_pair) % self.clips_per_video
        tubelet_idx = idx % tubelets_per_clip_pair
        
        seed = hash((vid_idx, clip_pair_instance_idx)) % 10000
        local_rng = random.Random(seed)
        
        # Select start frame for the first clip
        start_frame = local_rng.choice(self.start_frames_dict[vid_idx])
        start_frame_next = start_frame + self.frame_num
        
        possible_tubelets = list(range(self.patches_per_frame))
        local_rng.shuffle(possible_tubelets)
        actual_tubelet_idx = possible_tubelets[tubelet_idx % len(possible_tubelets)]
        
        tubelet_row = actual_tubelet_idx // self.patches_per_col
        tubelet_col = actual_tubelet_idx % self.patches_per_col
        
        vid_path = self.vid_list[vid_idx]
        vid_name = os.path.basename(vid_path)
        
        # Load frame sequences for both clips
        frames1 = []
        frames2 = []
        for i in range(self.frame_num):
            frames1.append(self.load_image(vid_path, start_frame + i))
            frames2.append(self.load_image(vid_path, start_frame_next + i))
        
        video1 = torch.stack(frames1, dim=1)  # C,T,H,W
        video2 = torch.stack(frames2, dim=1)  # C,T,H,W
        
        if self.split == 'train':
            cur_tfm = VideoTransform(crop_size=self.crop_size, scale=self.scale,
                ratio=self.aspect_ratio, eval_tfm=False, rand_flip=self.rand_flip)
        else:
            cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
            
        video_data1 = cur_tfm(video1)
        video_data2 = cur_tfm(video2)
        
        # Extract patch-tubelets
        h_start = tubelet_row * self.tubelet_size[0]
        w_start = tubelet_col * self.tubelet_size[1]
        tubelet1 = video_data1[:, :, h_start:h_start+self.tubelet_size[0], 
                              w_start:w_start+self.tubelet_size[1]]
        tubelet2 = video_data2[:, :, h_start:h_start+self.tubelet_size[0], 
                              w_start:w_start+self.tubelet_size[1]]
        
        # Stack the patch-tubelets into a pair:
        tubelet_pair = torch.stack([tubelet1, tubelet2], dim=0) # 2, C, T, H, W
        
        metadata = {
            'video_name': vid_name,
            'start_frame': start_frame,
            'start_frame_next': start_frame_next,
            'tubelet_row': tubelet_row,
            'tubelet_col': tubelet_col,
            'patches_per_row': self.patches_per_row,
            'patches_per_col': self.patches_per_col,
            'crop_size': self.crop_size,
            'tubelet_size': self.tubelet_size
        }
        
        return {'inp': tubelet_pair, 'gt': tubelet_pair, 
                'name': f"{vid_name}_{start_frame}", 'metadata': metadata}

@register('vidrec_dataset_full_res_overlapping_patches_single_video_uvg')
class VideoDataSetFullResOverlappingPatchesSingleVideoUVG(Dataset):
    """
    Dataset for processing a single UVG video with overlapping patches when patch size 
    doesn't perfectly divide the full resolution.
    """
    def __init__(self, video_path, frame_num, crop_size, tubelet_size, overlap_h=None, overlap_w=None):
        self.video_path = video_path
        self.frame_num = frame_num
        
        assert overlap_h is not None and overlap_w is not None, "overlap_h and overlap_w must be provided"
        self.overlap_h = overlap_h
        self.overlap_w = overlap_w
        
        self.crop_size = handle_size_param(crop_size)
        self.tubelet_size = handle_size_param(tubelet_size)
        
        # Get all image files and sort them
        if os.path.isdir(video_path):
            self.image_files = sorted([f for f in os.listdir(video_path)
                                  if f.startswith('f') and f.endswith('.png')])
            if not self.image_files:
                raise ValueError(f"No matching image files found in {video_path}")

            self.total_frames = len(self.image_files)
            self.vid_name = os.path.basename(video_path)
        else:
            vr = decord.VideoReader(video_path)
            self.total_frames = len(vr)
            self.vid_name = os.path.basename(video_path).split('.')[0] # More robust split
            self.image_files = []


        self.num_clips = self.total_frames // self.frame_num
        if self.total_frames % self.frame_num != 0:
            self.num_clips += 1
        
        # Calculate overlapping patch-tubelet grid
        self.patch_grid = self._calculate_overlapping_patch_grid()
        self.num_patches = len(self.patch_grid)
    
    def _calculate_overlapping_patch_grid(self):
        """Calculate the grid of overlapping patch-tubelets needed to cover the full frame"""
        patch_grid = []
        
        # Height dimension
        h_positions = []
        h_start = 0
        while h_start < self.crop_size[0]:
            h_end = min(h_start + self.tubelet_size[0], self.crop_size[0])
            h_positions.append((h_start, h_end))
            
            if h_end >= self.crop_size[0]:
                break
                
            # Calculate next start position with overlap
            next_h_start = h_start + self.tubelet_size[0] - self.overlap_h
            
            # If the next patch would extend beyond the frame, adjust it
            if next_h_start + self.tubelet_size[0] > self.crop_size[0]:
                next_h_start = self.crop_size[0] - self.tubelet_size[0]
                
            h_start = next_h_start
        
        # Width dimension  
        w_positions = []
        w_start = 0
        while w_start < self.crop_size[1]:
            w_end = min(w_start + self.tubelet_size[1], self.crop_size[1])
            w_positions.append((w_start, w_end))
            
            if w_end >= self.crop_size[1]:
                break
                
            next_w_start = w_start + self.tubelet_size[1] - self.overlap_w
            if next_w_start + self.tubelet_size[1] > self.crop_size[1]:
                next_w_start = self.crop_size[1] - self.tubelet_size[1]
                
            w_start = next_w_start
        
        # Create grid of all patch positions
        for h_start, h_end in h_positions:
            for w_start, w_end in w_positions:
                patch_grid.append({
                    'h_start': h_start,
                    'h_end': h_end,
                    'w_start': w_start,
                    'w_end': w_end
                })
        
        return patch_grid
    
    def load_image(self, idx):
        """Load a single image and convert to tensor"""
        img_path = os.path.join(self.video_path, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.
        return img
    
    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start_frame = idx * self.frame_num
        
        # Handle last clip which might be shorter
        if start_frame + self.frame_num > self.total_frames:
            frame_idx = list(range(start_frame, self.total_frames))
            frame_idx.extend([self.total_frames - 1] * (self.frame_num - len(frame_idx)))
        else:
            frame_idx = list(range(start_frame, start_frame + self.frame_num))
            
        # Load sequence of images
        if self.image_files:
            video = []
            for fidx in frame_idx:
                video.append(self.load_image(fidx))
            video = torch.stack(video, dim=1)  # C,T,H,W
        else:
            vr = decord.VideoReader(self.video_path)
            video = vr.get_batch(frame_idx)
            video = video.permute(-1,0,1,2).float() / 255.  # T,H,W,C -> C,T,H,W
            
        # Apply test transform as this is for evaluation
        cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
        video_data = cur_tfm(video)  # C,T,H,W
        
        # Create overlapping patch-tubelets
        patches = []
        patch_positions = []
        
        for t in range(video_data.shape[1]):  # For each frame
            frame = video_data[:, t]  # C,H,W
            frame_patches = []
            frame_positions = []
            
            for patch_info in self.patch_grid:
                h_start, h_end = patch_info['h_start'], patch_info['h_end']
                w_start, w_end = patch_info['w_start'], patch_info['w_end']
                
                # Extract patch-tubelet
                patch = frame[:, h_start:h_end, w_start:w_end]
                
                # Pad if necessary (should not happen with proper grid calculation)
                if patch.shape[1] < self.tubelet_size[0] or patch.shape[2] < self.tubelet_size[1]:
                    pad_h = self.tubelet_size[0] - patch.shape[1]
                    pad_w = self.tubelet_size[1] - patch.shape[2]
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='replicate')
                
                frame_patches.append(patch)
                frame_positions.append((h_start, w_start, h_end, w_end))
            
            patches.append(frame_patches)
            patch_positions.append(frame_positions)
        
        self.vid_list = [self.video_path]
        
        return {
            'inp': video_data,
            'gt': video_data,
            'name': f"{self.vid_name}_{start_frame}",
            'start_frame': start_frame,
            'patches': patches,
            'patch_positions': patch_positions,
            'metadata': {
                'num_patches': self.num_patches,
                'tubelet_size': self.tubelet_size,
                'crop_size': self.crop_size,
                'video_name': self.vid_name,
                'overlap_h': self.overlap_h,
                'overlap_w': self.overlap_w,
                'patch_grid': self.patch_grid
            }
        }
