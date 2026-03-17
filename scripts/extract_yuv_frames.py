"""Extract HEVC, or MCL-JCV YUV sequences into PNG frame folders."""

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_METADATA = {
    "hevc": [
        REPO_ROOT / "data/dataset_meta/hevc_class_b.csv",
        REPO_ROOT / "data/dataset_meta/hevc_class_c.csv",
        REPO_ROOT / "data/dataset_meta/hevc_class_e.csv",
    ],
    "mcl-jcv": [
        REPO_ROOT / "data/dataset_meta/mcl_jcv_24_1080p.csv",
    ],
}

DEFAULT_OUTPUT_ROOT = {
    "hevc": REPO_ROOT / "data/HEVC",
    "mcl-jcv": REPO_ROOT / "data/MCL-JCV",
}

YUV_NAME_RE = re.compile(
    r"^(?P<stem>.+)_(?P<width>\d+)x(?P<height>\d+)_(?P<suffix>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DEFAULT_METADATA),
        required=True,
        help="Dataset type. HEVC uses the trailing filename "
        "field as frame count; MCL-JCV uses it as frame rate.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw .yuv files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for extracted frame folders. Defaults to the repo layout.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        action="append",
        default=None,
        help="Metadata CSV(s) whose path column defines the expected output folders.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="Optional sequence stem(s) to extract. Defaults to all sequences in the metadata.",
    )
    parser.add_argument(
        "--pix-fmt",
        default="yuv420p",
        help="YUV pixel format passed to ffmpeg. Default: yuv420p",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable to use. Default: ffmpeg",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted frame folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the work that would be done without invoking ffmpeg.",
    )
    return parser.parse_args()


def load_expected_sequences(metadata_files: List[Path]) -> List[str]:
    sequences = []
    seen = set()
    for metadata_file in metadata_files:
        with metadata_file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                stem = Path(row["path"]).name
                if stem not in seen:
                    seen.add(stem)
                    sequences.append(stem)
    return sequences


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def parse_yuv_filename(stem: str) -> Dict[str, object]:
    match = YUV_NAME_RE.match(stem)
    if not match:
        raise ValueError(
            f"Unsupported YUV filename format: {stem}. "
            "Expected <name>_<width>x<height>_<frames-or-fps>."
        )
    parsed = match.groupdict()
    parsed["width"] = int(parsed["width"])
    parsed["height"] = int(parsed["height"])
    parsed["suffix"] = int(parsed["suffix"])
    return parsed


def count_pngs(directory: Path) -> int:
    return sum(1 for path in directory.glob("f*.png") if path.is_file())


def run_ffmpeg(
    ffmpeg_bin: str,
    yuv_path: Path,
    output_dir: Path,
    width: int,
    height: int,
    pix_fmt: str,
) -> None:
    output_pattern = output_dir / "f%05d.png"
    command = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-s:v",
        f"{width}x{height}",
        "-i",
        str(yuv_path),
        "-start_number",
        "1",
        str(output_pattern),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()

    ffmpeg_path = shutil.which(args.ffmpeg_bin)
    if ffmpeg_path is None:
        print(f"Could not find ffmpeg executable: {args.ffmpeg_bin}", file=sys.stderr)
        return 1

    metadata_files = args.metadata_csv or DEFAULT_METADATA[args.dataset]
    metadata_files = [path.resolve() for path in metadata_files]
    output_root = (args.output_root or DEFAULT_OUTPUT_ROOT[args.dataset]).resolve()
    input_dir = args.input_dir.resolve()

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    for metadata_file in metadata_files:
        if not metadata_file.is_file():
            print(f"Metadata CSV does not exist: {metadata_file}", file=sys.stderr)
            return 1

    expected_sequences = load_expected_sequences(metadata_files)
    if args.sequence:
        requested = set(args.sequence)
        expected_sequences = [stem for stem in expected_sequences if stem in requested]
        missing_requested = sorted(requested.difference(expected_sequences))
        if missing_requested:
            print(
                "Requested sequences were not found in the metadata: "
                + ", ".join(missing_requested),
                file=sys.stderr,
            )
            return 1

    available_yuvs = {path.stem: path for path in sorted(input_dir.glob("*.yuv"))}
    missing_inputs = [stem for stem in expected_sequences if stem not in available_yuvs]
    if missing_inputs:
        print("Missing input YUV files for:", file=sys.stderr)
        for stem in missing_inputs:
            print(f"  {stem}.yuv", file=sys.stderr)
        return 1
    sequence_count = len(expected_sequences)

    print(f"Dataset:     {args.dataset}")
    print(f"Input dir:   {input_dir}")
    print(f"Output root: {output_root}")
    print(f"Sequences:   {sequence_count}")

    failures = 0
    for item in expected_sequences:
        stem = item
        yuv_path = available_yuvs[stem]
        parsed = parse_yuv_filename(stem)
        output_dir = output_root / stem
        width = parsed["width"]
        height = parsed["height"]
        expected_frames = parsed["suffix"] if args.dataset == "hevc" else None
        display_name = stem

        if output_dir.exists():
            existing_pngs = count_pngs(output_dir)
            if existing_pngs > 0 and not args.overwrite:
                print(f"[skip] {display_name}: found {existing_pngs} existing PNGs")
                continue
            if args.overwrite:
                shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[extract] {display_name} -> {output_dir} "
            f"({width}x{height}, pix_fmt={args.pix_fmt})"
        )

        if args.dry_run:
            continue

        try:
            run_ffmpeg(
                ffmpeg_path,
                yuv_path,
                output_dir,
                width,
                height,
                args.pix_fmt,
            )
        except subprocess.CalledProcessError as exc:
            failures += 1
            print(f"  ffmpeg failed for {display_name}: {exc}", file=sys.stderr)
            continue

        actual_frames = count_pngs(output_dir)
        if expected_frames is not None and actual_frames != expected_frames:
            failures += 1
            print(
                f"  frame count mismatch for {display_name}: "
                f"expected {expected_frames}, extracted {actual_frames}",
                file=sys.stderr,
            )
            continue

        print(f"  extracted {actual_frames} PNG frames")

    if failures:
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 1

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
