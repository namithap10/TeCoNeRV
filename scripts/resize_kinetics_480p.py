
"""Resize the Kinetics-400 training subset to 640x480 for faster training."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple

import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = REPO_ROOT / "data/dataset_meta/k400_2023_train_cls400_50_480p.js"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "data/kinetics_2023/train"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/kinetics_480p/train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=DEFAULT_METADATA,
        help="JSON file listing the 480p training subset paths.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root directory of the source Kinetics videos.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for the resized 480p videos.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=640,
        help="Target output width. Default: 640",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=480,
        help="Target output height. Default: 480",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Number of worker processes, default: half of available CPUs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of videos to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos.",
    )
    return parser.parse_args()


def resize_and_center_crop(frame, target_width: int, target_height: int):

    height, width = frame.shape[:2]
    if height >= width:
        return None

    new_height = target_height
    new_width = int(width * (target_height / height))
    resized = cv2.resize(frame, (new_width, new_height))

    crop_x = max(0, new_width // 2 - target_width // 2)
    crop_y = max(0, new_height // 2 - target_height // 2)
    cropped = resized[crop_y : crop_y + target_height, crop_x : crop_x + target_width]

    if cropped.shape[0] != target_height or cropped.shape[1] != target_width:
        return None
    return cropped


def resize_video(
    source_path: Path,
    output_path: Path,
    target_width: int,
    target_height: int,
) -> bool:

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return False

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (target_width, target_height),
    )

    success = True
    while True:
        has_frame, frame = capture.read()
        if not has_frame:
            break

        resized = resize_and_center_crop(frame, target_width, target_height)
        if resized is None:
            success = False
            break
        writer.write(resized)

    capture.release()
    writer.release()

    if not success and output_path.exists():
        output_path.unlink()
    return success


def build_job_list(
    metadata_file: Path, source_root: Path, output_root: Path
) -> List[Tuple[Path, Path]]:
    with metadata_file.open() as handle:
        metadata = json.load(handle)

    jobs = []
    prefix = Path("data/kinetics_480p/train")
    for video_list in metadata.values():
        for output_rel in video_list:
            output_rel_path = Path(output_rel)
            relative_video = output_rel_path.relative_to(prefix)
            source_path = source_root / relative_video
            output_path = output_root / relative_video
            jobs.append((source_path, output_path))
    return jobs


def process_job(job):
    source_path, output_path, target_width, target_height, overwrite = job

    if not source_path.is_file():
        return ("missing", str(source_path))

    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return ("skip", str(output_path))

    if output_path.exists() and overwrite:
        output_path.unlink()

    success = resize_video(source_path, output_path, target_width, target_height)
    return ("ok" if success else "fail", str(source_path))


def main() -> int:
    args = parse_args()

    metadata_file = args.metadata_file.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if not metadata_file.is_file():
        raise SystemExit(f"Metadata file does not exist: {metadata_file}")
    if not source_root.is_dir():
        raise SystemExit(f"Source root does not exist: {source_root}")

    jobs = build_job_list(metadata_file, source_root, output_root)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    worker_jobs = [
        (source_path, output_path, args.target_width, args.target_height, args.overwrite)
        for source_path, output_path in jobs
    ]

    results = {"ok": 0, "skip": 0, "missing": 0, "fail": 0}
    with mp.Pool(processes=args.workers) as pool:
        for status, path in tqdm(
            pool.imap_unordered(process_job, worker_jobs),
            total=len(worker_jobs),
            desc="Resizing videos",
        ):
            results[status] += 1
            if status in {"missing", "fail"}:
                print(f"[{status}] {path}")

    print("Summary:")
    for key in ("ok", "skip", "missing", "fail"):
        print(f"  {key}: {results[key]}")

    return 1 if results["missing"] or results["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
