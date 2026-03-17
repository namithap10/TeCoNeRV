"""Utilities for reconstructing clips from saved x_dict dumps."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import einops
import imageio
import numpy as np
import torch
import torch.nn.functional as F

from models import make as make_model
from utils import make_coord_grid


TensorDict = Dict[str, torch.Tensor | None]


def load_cfg(model_dir: Path) -> dict:
    import yaml

    with (model_dir / "cfg.yaml").open("r") as f:
        return yaml.safe_load(f)


def build_hyponet(model_dir: Path, device: torch.device) -> tuple[dict, torch.nn.Module]:
    cfg = load_cfg(model_dir)
    hyponet = make_model(cfg["model"]["args"]["hyponet"], args={"n_groups": None})
    hyponet = hyponet.to(device)
    hyponet.eval()
    return cfg, hyponet


def load_base_params(pred_dir: Path, device: torch.device) -> TensorDict:
    base_params = torch.load(pred_dir / "base_params.pth", map_location=device)
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in base_params.items()}


def move_x_dict_to_device(x_dict: TensorDict, device: torch.device) -> TensorDict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x_dict.items()}


def clone_x_dict(x_dict: TensorDict) -> TensorDict:
    return {k: v.clone() if isinstance(v, torch.Tensor) else None for k, v in x_dict.items()}


def add_param_dict_residuals(base_dict: TensorDict, residual_dict: TensorDict) -> TensorDict:
    result: TensorDict = {}
    for key in base_dict.keys():
        if base_dict[key] is not None and residual_dict[key] is not None:
            result[key] = base_dict[key] + residual_dict[key]
        else:
            result[key] = None
    return result


def combine_batch_reconstructions(recon_batch_list: Sequence[TensorDict]) -> TensorDict:
    return {
        key: torch.cat([item[key] for item in recon_batch_list], dim=0)
        if recon_batch_list[0][key] is not None
        else None
        for key in recon_batch_list[0].keys()
    }


def convert_x_dict_to_params(
    batch_size: int,
    x_dict: TensorDict,
    base_params: TensorDict,
    hyponet: torch.nn.Module,
) -> TensorDict:
    params: TensorDict = {"embed": None}
    for name in hyponet.param_shapes.keys():
        wb = einops.repeat(base_params[name], "n m -> b n m", b=batch_size)
        init_w, init_b = wb[:, :-1, :], wb[:, -1:, :]
        if x_dict[name] is not None:
            x_val = x_dict[name]
            repeat_num = init_w.nelement() // x_val.nelement()
            x_val = einops.repeat(x_val, "b n m -> b n d m", d=repeat_num).reshape_as(init_w)
            weight = F.normalize(init_w * x_val, dim=1)
        else:
            weight = F.normalize(init_w, dim=1)
        params[name] = torch.cat([weight, init_b], dim=1)
    return params


def reconstruct_from_x_dict(
    batch_size: int,
    num_frames: int,
    x_dict: TensorDict,
    base_params: TensorDict,
    hyponet: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    coord = make_coord_grid((num_frames,), (-1, 1), device=device)
    coord = einops.repeat(coord, "t d -> b t d", b=batch_size)
    params = convert_x_dict_to_params(batch_size, x_dict, base_params, hyponet)
    hyponet.set_params(params)
    with torch.no_grad():
        pred = hyponet(coord)
    return pred


def pred_to_clip_frames(pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 5:
        raise ValueError(f"Expected 5D prediction tensor, got shape {tuple(pred.shape)}")
    if pred.shape[1] == 1:
        pred = pred[:, 0]
    elif pred.shape[0] == 1:
        pred = pred[0]
    else:
        raise ValueError(f"Cannot convert prediction of shape {tuple(pred.shape)} to single clip frames")
    return pred


def clip_to_uint8(clip_frames: torch.Tensor) -> np.ndarray:
    clip_np = clip_frames.detach().cpu().numpy()
    clip_np = np.clip(clip_np, 0.0, 1.0)
    clip_np = (clip_np * 255.0).round().astype(np.uint8)
    return clip_np


def save_clip_as_pngs(clip_frames: torch.Tensor, out_dir: Path, start_frame: int) -> List[np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_np = clip_to_uint8(clip_frames)
    rgb_frames: List[np.ndarray] = []
    for frame_idx in range(clip_np.shape[0]):
        rgb = np.ascontiguousarray(clip_np[frame_idx].transpose(1, 2, 0))
        rgb_frames.append(rgb)
        imageio.imwrite(out_dir / f"{frame_idx + start_frame + 1:03d}.png", rgb)
    return rgb_frames


def write_video(frames: Iterable[np.ndarray], out_path: Path, fps: int) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print(
            f"Warning: could not write MP4 to {out_path} because `ffmpeg` was not found. "
            "PNGs were still written successfully."
        )
        return False

    frame_dir = out_path.parent / "__mp4_frames__"
    try:
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        for frame_count, frame in enumerate(frames, start=1):
            imageio.imwrite(frame_dir / f"{frame_count:03d}.png", frame)

        if frame_count == 0:
            print(f"Warning: no frames available for MP4 export at {out_path}.")
            return False

        cmd = [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "%03d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as exc:
        print(
            f"Warning: could not write MP4 to {out_path} ({exc}). "
            "PNGs were still written successfully."
        )
        return False
    finally:
        if frame_dir.exists():
            shutil.rmtree(frame_dir)


def sample_clip_indices(num_clips: int, all_clips: bool, sampled_clips: int) -> List[int]:
    if num_clips <= 0:
        return []
    if all_clips or num_clips <= sampled_clips:
        return list(range(num_clips))
    return sorted(set(np.linspace(0, num_clips - 1, sampled_clips, dtype=int).tolist()))


def resolve_video_dir(base_dir: Path, video: str) -> Path:
    target = video.lower()
    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    for candidate in candidates:
        if candidate.name.lower() == target:
            return candidate
    for candidate in candidates:
        if target in candidate.name.lower():
            return candidate
    raise FileNotFoundError(
        f"Could not find video '{video}' under {base_dir}. Available: {[p.name for p in candidates]}"
    )


def collect_baseline_clip_files(video_dir: Path) -> List[Tuple[int, int, Path]]:
    pattern = re.compile(r".*_clip_(\d+)_frame_(\d+)\.pth$")
    clip_info: List[Tuple[int, int, Path]] = []
    for file_path in video_dir.iterdir():
        match = pattern.match(file_path.name)
        if match:
            clip_info.append((int(match.group(1)), int(match.group(2)), file_path))
    return sorted(clip_info, key=lambda item: item[0])


def _parse_tubelet_position(tubelet_dir: Path) -> Tuple[int, int]:
    match = re.match(r"tubelet_(\d+)_(\d+)$", tubelet_dir.name)
    if not match:
        raise ValueError(f"Invalid tubelet directory name: {tubelet_dir.name}")
    return int(match.group(1)), int(match.group(2))


def load_tubelet_infos(video_dir: Path, device: torch.device) -> List[dict]:
    tubelet_infos = []
    for tubelet_dir in sorted(video_dir.glob("tubelet_*"), key=_parse_tubelet_position):
        with (tubelet_dir / "metadata.json").open("r") as f:
            metadata = json.load(f)
        initial_x_dict = move_x_dict_to_device(
            torch.load(tubelet_dir / "clip_0_frame_0_full.pth", map_location=device),
            device,
        )
        info = {
            "dir": tubelet_dir,
            "position": tuple(metadata["position"]),
            "patch_size": tuple(metadata["patch_size"]),
            "full_res": tuple(metadata["full_res"]),
            "num_frames": int(metadata.get("num_frames", 8)),
            "overlap_h": int(metadata.get("overlap_h", 0)),
            "overlap_w": int(metadata.get("overlap_w", 0)),
            "initial_x_dict": initial_x_dict,
        }
        tubelet_infos.append(info)
    if not tubelet_infos:
        raise FileNotFoundError(f"No tubelet_* directories found under {video_dir}")
    return tubelet_infos


def collect_patch_clip_files(tubelet_dir: Path) -> List[Tuple[int, int, Path]]:
    pattern = re.compile(r"clip_(\d+)_frame_(\d+)_residual\.pth$")
    residual_files: List[Tuple[int, int, Path]] = []
    for file_path in tubelet_dir.iterdir():
        match = pattern.match(file_path.name)
        if match:
            residual_files.append((int(match.group(1)), int(match.group(2)), file_path))
    return sorted(residual_files, key=lambda item: item[0])


def tile_clip_from_patches(
    recon_patches: torch.Tensor,
    positions: Sequence[Tuple[int, int, int, int]],
    crop_size: Tuple[int, int],
) -> torch.Tensor:
    num_frames, num_patches, channels, _, _ = recon_patches.shape
    crop_h, crop_w = crop_size
    recon_clip = torch.zeros(num_frames, channels, crop_h, crop_w, device=recon_patches.device)
    for patch_idx, (h_start, w_start, h_end, w_end) in enumerate(positions):
        recon_clip[:, :, h_start:h_end, w_start:w_end] = recon_patches[:, patch_idx]
    return recon_clip


def calculate_patch_grid_info(
    positions: Sequence[Tuple[int, int, int, int]],
    overlap_h: int,
    overlap_w: int,
) -> List[dict]:
    h_positions = sorted(set(pos[0] for pos in positions))
    w_positions = sorted(set(pos[1] for pos in positions))
    patch_grid = []
    for h_start, w_start, _, _ in positions:
        h_idx = h_positions.index(h_start)
        w_idx = w_positions.index(w_start)
        has_top = h_idx > 0
        has_bottom = h_idx < len(h_positions) - 1
        has_left = w_idx > 0
        has_right = w_idx < len(w_positions) - 1
        patch_grid.append(
            {
                "crop_top": min((overlap_h // 2) if has_top else 0, overlap_h),
                "crop_bottom": min((overlap_h // 2) if has_bottom else 0, overlap_h),
                "crop_left": min((overlap_w // 2) if has_left else 0, overlap_w),
                "crop_right": min((overlap_w // 2) if has_right else 0, overlap_w),
            }
        )
    return patch_grid


def tile_clip_from_overlapping_patches_with_cropping(
    recon_patches: torch.Tensor,
    positions: Sequence[Tuple[int, int, int, int]],
    crop_size: Tuple[int, int],
    overlap_h: int,
    overlap_w: int,
) -> torch.Tensor:
    num_frames, num_patches, channels, _, _ = recon_patches.shape
    crop_h, crop_w = crop_size
    recon_clip = torch.zeros(num_frames, channels, crop_h, crop_w, device=recon_patches.device)
    patch_grid = calculate_patch_grid_info(positions, overlap_h, overlap_w)
    for patch_idx, (h_start, w_start, h_end, w_end) in enumerate(positions):
        grid_info = patch_grid[patch_idx]
        crop_top = grid_info["crop_top"]
        crop_bottom = grid_info["crop_bottom"]
        crop_left = grid_info["crop_left"]
        crop_right = grid_info["crop_right"]
        patch_h = h_end - h_start
        patch_w = w_end - w_start
        patch_cropped = recon_patches[
            :,
            patch_idx,
            :,
            crop_top:patch_h - crop_bottom,
            crop_left:patch_w - crop_right,
        ]
        out_h_start = h_start + crop_top
        out_h_end = h_end - crop_bottom
        out_w_start = w_start + crop_left
        out_w_end = w_end - crop_right
        recon_clip[:, :, out_h_start:out_h_end, out_w_start:out_w_end] = patch_cropped
    return recon_clip


def tile_clip_from_overlapping_patches_with_blending(
    recon_patches: torch.Tensor,
    positions: Sequence[Tuple[int, int, int, int]],
    crop_size: Tuple[int, int],
    overlap_h: int,
    overlap_w: int,
) -> torch.Tensor:
    num_frames, num_patches, channels, _, _ = recon_patches.shape
    crop_h, crop_w = crop_size
    recon_clip = torch.zeros(num_frames, channels, crop_h, crop_w, device=recon_patches.device)
    weight_map = torch.zeros(num_frames, channels, crop_h, crop_w, device=recon_patches.device)
    for patch_idx, (h_start, w_start, h_end, w_end) in enumerate(positions):
        patch_h = h_end - h_start
        patch_w = w_end - w_start
        weight_h = torch.ones(patch_h, device=recon_patches.device)
        weight_w = torch.ones(patch_w, device=recon_patches.device)
        if overlap_h > 0:
            fade_h = int(min(overlap_h // 2, patch_h // 4))
            if patch_h > 2 * fade_h and fade_h > 0:
                weight_h[:fade_h] = torch.linspace(0.1, 1.0, fade_h, device=recon_patches.device)
                weight_h[-fade_h:] = torch.linspace(1.0, 0.1, fade_h, device=recon_patches.device)
        if overlap_w > 0:
            fade_w = int(min(overlap_w // 2, patch_w // 4))
            if patch_w > 2 * fade_w and fade_w > 0:
                weight_w[:fade_w] = torch.linspace(0.1, 1.0, fade_w, device=recon_patches.device)
                weight_w[-fade_w:] = torch.linspace(1.0, 0.1, fade_w, device=recon_patches.device)
        patch_weight = weight_h[:, None] * weight_w[None, :]
        patch_weight = patch_weight[None, None, :, :].expand(num_frames, channels, -1, -1)
        recon_clip[:, :, h_start:h_end, w_start:w_end] += recon_patches[:, patch_idx] * patch_weight
        weight_map[:, :, h_start:h_end, w_start:w_end] += patch_weight
    return recon_clip / (weight_map + 1e-8)
