import json
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
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    RandAugment,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
)

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


def _make_video_reader(video_path):
    if PyVideoReader is not None:
        return PyVideoReader(video_path)
    return decord.VideoReader(video_path)


def handle_size_param(size_param):
    """
    Utility function to handle size parameters (crop_size, tubelet_size etc.)
    Args:
        size_param: Can be int, tuple/list of (h,w), or string 'h_w'
    Returns:
        Tuple of (height, width)
    """
    if isinstance(size_param, (tuple, list)):
        return tuple(size_param)  # Convert to tuple for consistency
    elif isinstance(size_param, int):
        return (size_param, size_param)
    elif isinstance(size_param, str) and '_' in size_param:
        return tuple(int(x) for x in size_param.split('_'))
    else:
        raise ValueError(f"Unsupported size parameter format: {size_param}")

def VideoTransform(crop_size=128, scale=1.05, ratio=1.05, eval_tfm=False, 
    rand_flip='no'):
    if isinstance(crop_size, (tuple, list)):
        resize_size = min(crop_size)  # Resize based on shorter edge
    elif isinstance(crop_size, str) and '_' in crop_size:
        dims = [int(x) for x in crop_size.split('_')]
        resize_size = min(dims)
        crop_size = tuple(dims)
    else:
        resize_size = crop_size
        
    if eval_tfm:
        transform = transforms.Compose([
            Resize(size=resize_size, antialias=True),  # Resize shorter edge
            CenterCrop(crop_size)
        ])
    else:
        if scale == 1 and ratio==1:
            tfm_list = [Resize(size=resize_size, antialias=True), CenterCrop(crop_size)]
        else:
            tfm_list = [
                Resize(size=int(resize_size/scale), antialias=True),
                RandomResizedCrop(crop_size, (1./scale**2, 1), (1./ratio, ratio), antialias=True)
            ]
        if rand_flip != 'no':
            tfm_list.append(RandomHorizontalFlip())
        transform = transforms.Compose(tfm_list)

    return transform


@register('vidrec_dataset_clip_sampler')
class VideoDataSetClipSampler(Dataset):
    def __init__(self, root_path, frame_num, cls_vid_num, crop_size, rand_flip='no',
        split='train', csv_file='', scale=1, aspect_ratio=1, rand_augment='no', clips_per_video=1):

        if csv_file != '':
            self.cls_list = None
            csv_file = os.path.join(root_path, csv_file)
            if csv_file.endswith('.csv'):
                self.vid_list = pd.read_csv(csv_file)['path'].tolist()
            elif csv_file.endswith('.js'):
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
                print(cur_cls, flush=True)
                cur_dir = os.path.join(root_path, cur_cls)
                print(len(os.listdir(cur_dir)), flush=True)
                for cur_vid in sorted(os.listdir(cur_dir))[:vid_num]:
                    self.vid_list.append(os.path.join(cur_dir, cur_vid))
        self.split, self.frame_num, self.rand_flip = split, frame_num, rand_flip
        self.crop_size, self.scale, self.aspect_ratio = crop_size, scale, aspect_ratio
        if rand_augment in ['no', '']:
            self.augment = None
        else:
            num_ops, magnitude, num_magnitude_bins = [int(x) for x in rand_augment.split('_')]
            self.augment = RandAugment(num_ops, magnitude, num_magnitude_bins)

        self.start_frames = {}
        for idx in range(len(self.vid_list)):
            self.start_frames[idx] = []
            vr = _make_video_reader(self.vid_list[idx])
            cur_len = len(vr)
            num_clips = cur_len // (self.frame_num)
            start_frame = 0
            for _ in range(num_clips):
                self.start_frames[idx].append(start_frame)
                start_frame += (self.frame_num)

        self.clips_per_video = clips_per_video
        
    def __len__(self):
        return len(self.vid_list) * self.clips_per_video

    def __getitem__(self, idx):
        vid_idx = idx // self.clips_per_video
        vid_idx = vid_idx.item() if torch.is_tensor(vid_idx) else vid_idx
        start_frame = random.choice(self.start_frames[vid_idx])
        vr = _make_video_reader(self.vid_list[vid_idx])
        frame_num = min(self.frame_num, len(vr))
        frame_idx = [int(x + start_frame) for x in range(frame_num)]
        video = vr.get_batch(frame_idx)
        video = _ensure_video_tensor(video)
        vid_name = f'{self.vid_list[vid_idx].split("/")[-1].split(".")[0]}_{start_frame}'
        if self.augment is not None:
            video = self.augment(video.permute(0,-1,1,2)).permute(0,2,3,1)
        video = video.permute(-1,0,1,2).float() / 255. # # T,H,W,C -> C,T,H,W
        if self.split == 'train':
            cur_tfm = VideoTransform(crop_size=self.crop_size, scale=self.scale, 
                ratio=self.aspect_ratio, eval_tfm=False)
        elif self.split == 'test':
            cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
        else:
            NotImplementedError
        video_data = cur_tfm(video)
        video_data = F.pad(video_data, (0,0,0,0,0,self.frame_num-frame_num), mode='replicate')

        return {'inp': video_data, 'gt': video_data, 'name': vid_name}


@register('vidrec_dataset_single_video_clips')
class VideoDataSetSingleVideoClips(Dataset):
    """
    Dataset for processing all clips from a single video.
    Each clip is frame_num frames long and identified by start_frame.
    Clips are returned in sequential order.
    """
    def __init__(self, video_path, frame_num, crop_size):
        self.video_path = video_path
        self.frame_num = frame_num
        self.crop_size = crop_size
        
        # Get video info
        vr = decord.VideoReader(video_path)
        self.total_frames = len(vr)
        self.num_clips = self.total_frames // self.frame_num
        if self.total_frames % self.frame_num != 0:
            self.num_clips += 1  # Include last partial clip
            
        self.vid_name = os.path.basename(video_path).replace('.mp4', '')

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start_frame = idx * self.frame_num
        vr = decord.VideoReader(self.video_path)
        
        # Handle last clip which might be shorter
        if start_frame + self.frame_num > self.total_frames:
            frame_idx = list(range(start_frame, self.total_frames))
            # Pad with last frame if needed
            frame_idx.extend([self.total_frames - 1] * (self.frame_num - len(frame_idx)))
        else:
            frame_idx = list(range(start_frame, start_frame + self.frame_num))
            
        video = vr.get_batch(frame_idx)
        video = video.permute(-1,0,1,2).float() / 255.  # T,H,W,C -> C,T,H,W
        
        # Use test transform as this is for evaluation
        cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
        video_data = cur_tfm(video)
        
        return {
            'inp': video_data, 
            'gt': video_data, 
            'name': self.vid_name,
            'start_frame': start_frame
        }        

@register('vidrec_dataset_single_video_clips_uvg')
class VideoDataSetSingleVideoClipsUVG(Dataset):
    """
    Dataset for processing all clips from a sequence of images in a folder.
    Each clip is frame_num frames long and identified by start_frame.
    Expects images named as f00xxx.png where xxx starts from 001.
    """
    def __init__(self, video_path, frame_num, crop_size):
        self.video_path = video_path
        self.frame_num = frame_num
        self.crop_size = crop_size
        
        # Get all image files and sort them
        self.image_files = sorted([f for f in os.listdir(video_path) 
                                 if f.startswith('f00') and f.endswith('.png')])
        
        if not self.image_files:
            raise ValueError(f"No matching image files found in {video_path}")
        
        self.total_frames = len(self.image_files)
        self.num_clips = self.total_frames // self.frame_num
        if self.total_frames % self.frame_num != 0:
            self.num_clips += 1  # Include last partial clip
            
        self.seq_name = os.path.basename(video_path)

    def __len__(self):
        return self.num_clips

    def load_image(self, idx):
        """Load a single image and convert to tensor"""
        img_path = os.path.join(self.video_path, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        # Convert to tensor and normalize to [0, 1]
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.
        return img

    def __getitem__(self, idx):
        start_frame = idx * self.frame_num
        
        # Handle last clip which might be shorter
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
        
        # Stack frames into a single tensor
        video = torch.stack(video, dim=1)  # C,T,H,W
        
        # Use test transform as this is for evaluation
        cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
        video_data = cur_tfm(video)
        
        return {
            'inp': video_data, 
            'gt': video_data, 
            'name': self.seq_name,
            'start_frame': start_frame
        }
        
@register('vidrec_dataset_clip_sampler_lazy')
class VideoDataSetClipSamplerLazy(Dataset):
    """
    Lazy loading version of VideoDataSetClipSampler with better distributed training support.
    Each worker can efficiently select clips without loading all videos upfront.
    """
    def __init__(self, root_path, frame_num, cls_vid_num, crop_size, rand_flip='no',
        split='train', csv_file='', scale=1, aspect_ratio=1, rand_augment='no', clips_per_video=1):
        
        # Load video list
        if csv_file != '':
            self.cls_list = None
            csv_file = os.path.join(root_path, csv_file)
            if csv_file.endswith('.csv'):
                import pandas as pd
                self.vid_list = pd.read_csv(csv_file)['path'].tolist()
            elif csv_file.endswith('.js'):
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
                print(cur_cls, flush=True)
                cur_dir = os.path.join(root_path, cur_cls)
                print(len(os.listdir(cur_dir)), flush=True)
                for cur_vid in sorted(os.listdir(cur_dir))[:vid_num]:
                    self.vid_list.append(os.path.join(cur_dir, cur_vid))
        
        self.split, self.frame_num, self.rand_flip = split, frame_num, rand_flip
        self.crop_size, self.scale, self.aspect_ratio = crop_size, scale, aspect_ratio
        
        if rand_augment in ['no', '']:
            self.augment = None
        else:
            num_ops, magnitude, num_magnitude_bins = [int(x) for x in rand_augment.split('_')]
            self.augment = RandAugment(num_ops, magnitude, num_magnitude_bins)

        self.clips_per_video = clips_per_video
        
        # Lazy initialization
        self._video_lengths = {}  # Cache for video lengths
        self._valid_start_frames = {}  # Cache for valid start frames per video
        
    def _get_video_info(self, idx):
        """Lazily get video length and valid start frames"""
        if idx not in self._video_lengths:
            try:
                vr = _make_video_reader(self.vid_list[idx])
                length = len(vr)
                self._video_lengths[idx] = length
                
                # Calculate valid start frames
                num_clips = length // self.frame_num
                valid_starts = list(range(0, num_clips * self.frame_num, self.frame_num))
                self._valid_start_frames[idx] = valid_starts
                
            except Exception as e:
                print(f"Error loading video {self.vid_list[idx]}: {str(e)}")
                # Return dummy values if video is corrupted
                self._video_lengths[idx] = self.frame_num
                self._valid_start_frames[idx] = [0]
                
        return self._video_lengths[idx], self._valid_start_frames[idx]

    def __len__(self):
        return len(self.vid_list) * self.clips_per_video

    def __getitem__(self, idx):
        vid_idx = idx // self.clips_per_video
        vid_idx = vid_idx.item() if torch.is_tensor(vid_idx) else vid_idx
        
        _, valid_starts = self._get_video_info(vid_idx)
        
        # Set random seed based on index to ensure deterministic sampling
        rng = random.Random(hash((vid_idx, idx)) % 10000)
        start_frame = rng.choice(valid_starts)
        
        vr = _make_video_reader(self.vid_list[vid_idx])
        frame_idx = [int(x + start_frame) for x in range(self.frame_num)]
        video = vr.get_batch(frame_idx)
        video = _ensure_video_tensor(video)
        
        vid_name = f'{self.vid_list[vid_idx].split("/")[-1].split(".")[0]}_{start_frame}'
        
        if self.augment is not None:
            video = self.augment(video.permute(0,-1,1,2)).permute(0,2,3,1)
            
        video = video.permute(-1,0,1,2).float() / 255.  # T,H,W,C -> C,T,H,W
        
        if self.split == 'train':
            cur_tfm = VideoTransform(crop_size=self.crop_size, scale=self.scale, 
                ratio=self.aspect_ratio, eval_tfm=False)
        elif self.split == 'test':
            cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
        else:
            raise NotImplementedError
            
        video_data = cur_tfm(video)
        
        return {'inp': video_data, 'gt': video_data, 'name': vid_name}
    
@register('vidrec_dataset_clip_inference_lazy_uvg')
class VideoDataSetClipInferenceLazyUVG(Dataset):
    """
    Lazy loading dataset for UVG video evaluation that processes full clips without patches.
    Each UVG video is a sequence of PNG files in a folder.
    """
    def __init__(self, root_path, frame_num, cls_vid_num, crop_size, rand_flip='no',
        split='test', csv_file='', scale=1, aspect_ratio=1, rand_augment='no'):
        
        self.crop_size = handle_size_param(crop_size)
        
        csv_file = os.path.join(root_path, csv_file)
        if csv_file.endswith('.csv'):
            self.vid_list = pd.read_csv(csv_file)['path'].tolist()
        elif csv_file.endswith('.js'):
            with open(csv_file, 'r') as f:
                vid_dict = json.load(f)
            cls_num, vid_num = [int(x) for x in cls_vid_num.split('_')]
            sorted_keys=sorted(vid_dict, key=lambda k: len(vid_dict[k]), reverse=True)
            vid_list = [vid_dict[cls][:vid_num] for cls in sorted_keys[:cls_num]]
            self.vid_list = sum(vid_list, []) # beauty, bosphore, ...
        
        self.split, self.frame_num, self.rand_flip = split, frame_num, rand_flip
        self.scale, self.aspect_ratio = scale, aspect_ratio
        
        if rand_augment in ['no', '']:
            self.augment = None
        else:
            raise NotImplementedError("Augmentation not supported for UVG evaluation")
            
        # Generate all possible clip indices without loading data
        self.samples = []
        
        for vid_idx, vid_path in enumerate(self.vid_list):
            # Get video name from path
            vid_name = os.path.basename(vid_path)
            
            # Get all image files and sort them
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
                    
                sample_info = {
                    'vid_idx': vid_idx,
                    'vid_path': vid_path,
                    'vid_name': vid_name,
                    'start_frame': start_frame
                }
                self.samples.append(sample_info)
    
    def __len__(self):
        return len(self.samples)
    
    def load_image(self, path, index):
        """Load a single image from the UVG dataset"""
        img_files = sorted([f for f in os.listdir(path) if f.startswith('f') and f.endswith('.png')])
        img_path = os.path.join(path, img_files[index])
        img = Image.open(img_path).convert('RGB')
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.
        return img
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load the clip data on demand
        vid_path = sample['vid_path']
        start_frame = sample['start_frame']
        
        # Load frame sequence
        frames = []
        for i in range(self.frame_num):
            frame_idx = start_frame + i
            frames.append(self.load_image(vid_path, frame_idx))
        
        # Stack frames to create video tensor
        video = torch.stack(frames, dim=1)  # C,T,H,W
        
        # Resize/crop to target size
        cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
        video_data = cur_tfm(video)
        
        name = f"{sample['vid_name']}_{start_frame}"
        
        metadata = {
            'video_name': sample['vid_name'],
            'start_frame': start_frame,
            'frame_size': self.crop_size
        }
        
        return {
            'inp': video_data, 
            'gt': video_data, 
            'name': name, 
            'metadata': metadata
        }
