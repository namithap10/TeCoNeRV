"""Reconstruct clips or videos from saved quantized x_dict dumps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reconstruct_utils import (
    add_param_dict_residuals,
    build_hyponet,
    collect_baseline_clip_files,
    collect_patch_clip_files,
    combine_batch_reconstructions,
    load_base_params,
    load_tubelet_infos,
    move_x_dict_to_device,
    pred_to_clip_frames,
    reconstruct_from_x_dict,
    resolve_video_dir,
    sample_clip_indices,
    save_clip_as_pngs,
    tile_clip_from_overlapping_patches_with_blending,
    tile_clip_from_overlapping_patches_with_cropping,
    tile_clip_from_patches,
    write_video,
)


def parse_bool(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def reconstruct_baseline(args: argparse.Namespace, device: torch.device) -> None:
    model_dir = Path(args.model_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)

    cfg, hyponet = build_hyponet(model_dir, device)
    base_params = load_base_params(pred_dir, device)
    video_dir = resolve_video_dir(pred_dir / "direct", args.video)
    clip_files = collect_baseline_clip_files(video_dir)
    if not clip_files:
        raise FileNotFoundError(f"No baseline clip dumps found under {video_dir}")

    num_frames = int(cfg["test_dataset"]["args"].get("frame_num", 8))
    selected_indices = sample_clip_indices(len(clip_files), args.all_clips, args.sampled_clips)
    mp4_frames = []

    for clip_list_idx in selected_indices:
        clip_idx, start_frame, clip_path = clip_files[clip_list_idx]
        x_dict = move_x_dict_to_device(torch.load(clip_path, map_location=device), device)
        pred = reconstruct_from_x_dict(1, num_frames, x_dict, base_params, hyponet, device)
        clip_frames = pred_to_clip_frames(pred)
        clip_out_dir = out_dir / f"qual_clip_{clip_idx}_frame_{start_frame}"
        rgb_frames = save_clip_as_pngs(clip_frames, clip_out_dir, start_frame)
        if args.save_mp4:
            mp4_frames.extend(rgb_frames)

    if args.save_mp4:
        write_video(mp4_frames, out_dir / "reconstruction.mp4", args.fps)


def reconstruct_patch(args: argparse.Namespace, device: torch.device) -> None:
    model_dir = Path(args.model_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)

    cfg, hyponet = build_hyponet(model_dir, device)
    base_params = load_base_params(pred_dir, device)
    video_dir = resolve_video_dir(pred_dir / "from_prev", args.video)
    tubelet_infos = load_tubelet_infos(video_dir, device)
    positions = [tuple(info["position"]) for info in tubelet_infos]
    crop_size = tuple(tubelet_infos[0]["full_res"])
    residual_files = collect_patch_clip_files(tubelet_infos[0]["dir"])
    num_clips = len(residual_files) + 1
    num_frames = int(cfg["test_dataset"]["args"].get("frame_num", tubelet_infos[0]["num_frames"]))
    selected_indices = sample_clip_indices(num_clips, args.all_clips, args.sampled_clips)

    prev_x_dicts = {tuple(info["position"]): info["initial_x_dict"] for info in tubelet_infos}
    mp4_frames = []

    for clip_idx in range(num_clips):
        frame_idx = 0 if clip_idx == 0 else residual_files[clip_idx - 1][1]
        x_dict_recon_patches = []
        if clip_idx == 0:
            for info in tubelet_infos:
                x_dict_recon_patches.append(prev_x_dicts[tuple(info["position"])])
        else:
            for info in tubelet_infos:
                pos_key = tuple(info["position"])
                residual_path = info["dir"] / f"clip_{clip_idx}_frame_{frame_idx}_residual.pth"
                residual = move_x_dict_to_device(torch.load(residual_path, map_location=device), device)
                prev_x_dicts[pos_key] = add_param_dict_residuals(prev_x_dicts[pos_key], residual)
                x_dict_recon_patches.append(prev_x_dicts[pos_key])

        if clip_idx not in selected_indices:
            continue

        recon_batch = combine_batch_reconstructions(x_dict_recon_patches)
        recon_patches = reconstruct_from_x_dict(
            len(x_dict_recon_patches), num_frames, recon_batch, base_params, hyponet, device
        )
        full_clip = tile_clip_from_patches(recon_patches, positions, crop_size)
        clip_out_dir = out_dir / f"qual_clip_{clip_idx}_frame_{frame_idx}"
        rgb_frames = save_clip_as_pngs(full_clip, clip_out_dir, frame_idx)
        if args.save_mp4:
            mp4_frames.extend(rgb_frames)

    if args.save_mp4:
        write_video(mp4_frames, out_dir / "reconstruction.mp4", args.fps)


def reconstruct_patch_overlap(args: argparse.Namespace, device: torch.device) -> None:
    model_dir = Path(args.model_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)

    cfg, hyponet = build_hyponet(model_dir, device)
    base_params = load_base_params(pred_dir, device)
    video_dir = resolve_video_dir(pred_dir / "from_prev", args.video)
    tubelet_infos = load_tubelet_infos(video_dir, device)
    positions = [tuple(info["position"]) for info in tubelet_infos]
    crop_size = tuple(tubelet_infos[0]["full_res"])
    overlap_h = int(tubelet_infos[0]["overlap_h"])
    overlap_w = int(tubelet_infos[0]["overlap_w"])
    residual_files = collect_patch_clip_files(tubelet_infos[0]["dir"])
    num_clips = len(residual_files) + 1
    num_frames = int(cfg["test_dataset"]["args"].get("frame_num", tubelet_infos[0]["num_frames"]))
    selected_indices = sample_clip_indices(num_clips, args.all_clips, args.sampled_clips)

    prev_x_dicts = {tuple(info["position"]): info["initial_x_dict"] for info in tubelet_infos}
    mp4_frames = []

    for clip_idx in range(num_clips):
        frame_idx = 0 if clip_idx == 0 else residual_files[clip_idx - 1][1]
        x_dict_recon_patches = []
        if clip_idx == 0:
            for info in tubelet_infos:
                x_dict_recon_patches.append(prev_x_dicts[tuple(info["position"])])
        else:
            for info in tubelet_infos:
                pos_key = tuple(info["position"])
                residual_path = info["dir"] / f"clip_{clip_idx}_frame_{frame_idx}_residual.pth"
                residual = move_x_dict_to_device(torch.load(residual_path, map_location=device), device)
                prev_x_dicts[pos_key] = add_param_dict_residuals(prev_x_dicts[pos_key], residual)
                x_dict_recon_patches.append(prev_x_dicts[pos_key])

        if clip_idx not in selected_indices:
            continue

        recon_batch = combine_batch_reconstructions(x_dict_recon_patches)
        recon_patches = reconstruct_from_x_dict(
            len(x_dict_recon_patches), num_frames, recon_batch, base_params, hyponet, device
        )
        if args.blend_overlap:
            full_clip = tile_clip_from_overlapping_patches_with_blending(
                recon_patches, positions, crop_size, overlap_h, overlap_w
            )
        else:
            full_clip = tile_clip_from_overlapping_patches_with_cropping(
                recon_patches, positions, crop_size, overlap_h, overlap_w
            )
        clip_out_dir = out_dir / f"qual_clip_{clip_idx}_frame_{frame_idx}"
        rgb_frames = save_clip_as_pngs(full_clip, clip_out_dir, frame_idx)
        if args.save_mp4:
            mp4_frames.extend(rgb_frames)

    if args.save_mp4:
        write_video(mp4_frames, out_dir / "reconstruction.mp4", args.fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct qualitative frames from saved quantized x_dict predictions or residuals."
    )
    parser.add_argument("--mode", choices=["baseline", "patch", "patch_overlap"], required=True)
    parser.add_argument("--model_dir", required=True, help="Checkpoint directory containing cfg.yaml")
    parser.add_argument("--pred_dir", required=True, help="Directory created by the weight-saver evaluation scripts")
    parser.add_argument("--video", required=True, help="Video name to reconstruct")
    parser.add_argument("--out_dir", required=True, help="Directory where reconstructed PNGs will be written")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--all_clips", action="store_true", help="Reconstruct all clips instead of sampling")
    parser.add_argument("--sampled_clips", type=int, default=4, help="Number of sampled clips when --all_clips is not used")
    parser.add_argument("--save_mp4", action="store_true", help="Additionally write a stitched MP4")
    parser.add_argument("--fps", type=int, default=30, help="FPS for the stitched MP4 output")
    parser.add_argument("--blend_overlap", type=parse_bool, default=False, help="For patch_overlap mode, blend overlap regions instead of cropping them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.mode == "baseline":
        reconstruct_baseline(args, device)
    elif args.mode == "patch":
        reconstruct_patch(args, device)
    elif args.mode == "patch_overlap":
        reconstruct_patch_overlap(args, device)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
