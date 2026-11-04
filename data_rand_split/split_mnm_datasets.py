#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split the M&Ms raw dataset into the directory layout expected by FDAS.

Default behavior treats each raw *_sa.nii or *_sa.nii.gz case as one sample.
It standardizes output names to:
    image: <sample_id>.nii.gz
    label: <sample_id>_gt.nii.gz
so FDAS's dataloader can map image -> label by appending '_gt.nii.gz'.

Example for the user's case-level split:
    python split_mnm_for_fdas.py \
        --src /data/WPW/datasets/MnM \
        --dst /data/WPW/datasets/MnM01 \
        --sample-unit case \
        --train-count 242 --valid-count 34 --test-count 69 \
        --subset10-count 24 --subset20-count 48 \
        --seed 2025 --overwrite

If your raw *_sa.nii files are 4D and the original FDAS scripts fail on them,
use volume mode instead. It splits each 4D case into 3D volumes using frames
whose GT has foreground labels:
    python split_mnm_for_fdas.py \
        --src /data/WPW/datasets/MnM \
        --dst /data/WPW/datasets/MnM01 \
        --sample-unit volume \
        --frame-policy nonzero-gt \
        --seed 2025 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class Pair:
    case_id: str
    image_stem: str
    img_path: Path
    lab_path: Path
    source_group: str


@dataclass(frozen=True)
class Sample:
    sample_id: str
    img_path: Path
    lab_path: Path
    source_group: str
    original_case_id: str
    frame_index: Optional[int] = None


def strip_nii_suffix(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    raise ValueError(f"Not a NIfTI filename: {filename}")


def is_nii_file(path: Path) -> bool:
    name = path.name
    return name.endswith(".nii") or name.endswith(".nii.gz")


def is_sa_image(path: Path) -> bool:
    """True for *_sa.nii / *_sa.nii.gz, false for *_sa_gt.nii(.gz)."""
    if not path.is_file() or not is_nii_file(path):
        return False
    stem = strip_nii_suffix(path.name)
    return stem.endswith("_sa") and not stem.endswith("_sa_gt")


def find_label_for_image(img_path: Path) -> Optional[Path]:
    stem = strip_nii_suffix(img_path.name)
    candidates = [
        img_path.with_name(stem + "_gt.nii.gz"),
        img_path.with_name(stem + "_gt.nii"),
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    return None


def collect_pairs(src_root: Path) -> list[Pair]:
    source_dirs = [
        ("Training_Labeled", src_root / "Training" / "Labeled"),
        ("Training_Unlabeled", src_root / "Training" / "Unlabeled"),
        ("Validation", src_root / "Validation"),
        ("Testing", src_root / "Testing"),
    ]

    pairs: list[Pair] = []
    seen_stems: set[str] = set()

    for group_name, group_dir in source_dirs:
        if not group_dir.exists():
            print(f"[WARN] Missing source folder, skipped: {group_dir}", file=sys.stderr)
            continue

        for img_path in sorted(group_dir.rglob("*_sa.nii*")):
            if not is_sa_image(img_path):
                continue

            lab_path = find_label_for_image(img_path)
            if lab_path is None:
                raise FileNotFoundError(
                    f"Cannot find GT for image: {img_path}\n"
                    f"Expected: {strip_nii_suffix(img_path.name)}_gt.nii or .nii.gz"
                )

            image_stem = strip_nii_suffix(img_path.name)  # e.g. A0S9V9_sa
            case_id = image_stem[:-3] if image_stem.endswith("_sa") else image_stem

            if image_stem in seen_stems:
                raise RuntimeError(
                    f"Duplicate image stem found: {image_stem}. Please check duplicated cases."
                )
            seen_stems.add(image_stem)

            pairs.append(
                Pair(
                    case_id=case_id,
                    image_stem=image_stem,
                    img_path=img_path,
                    lab_path=lab_path,
                    source_group=group_name,
                )
            )

    return pairs


def import_sitk_numpy():
    try:
        import SimpleITK as sitk  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "SimpleITK and numpy are required for --sample-unit volume. "
            "Install them in your FDAS environment, e.g. pip install SimpleITK numpy."
        ) from exc
    return sitk, np


def build_case_samples(pairs: list[Pair]) -> list[Sample]:
    samples = []
    for pair in pairs:
        samples.append(
            Sample(
                sample_id=pair.image_stem,  # A0S9V9_sa -> image A0S9V9_sa.nii.gz
                img_path=pair.img_path,
                lab_path=pair.lab_path,
                source_group=pair.source_group,
                original_case_id=pair.case_id,
                frame_index=None,
            )
        )
    return samples


def build_volume_samples(pairs: list[Pair], frame_policy: str) -> list[Sample]:
    """
    Convert raw cases into 3D volume samples.
    For 4D images, frames are selected according to frame_policy.
    For 3D images, the image is used as one sample.
    """
    sitk, np = import_sitk_numpy()
    samples: list[Sample] = []

    for pair in pairs:
        img_obj = sitk.ReadImage(str(pair.img_path))
        lab_obj = sitk.ReadImage(str(pair.lab_path))
        img_dim = img_obj.GetDimension()
        lab_dim = lab_obj.GetDimension()

        if img_dim == 3:
            samples.append(
                Sample(
                    sample_id=pair.image_stem,
                    img_path=pair.img_path,
                    lab_path=pair.lab_path,
                    source_group=pair.source_group,
                    original_case_id=pair.case_id,
                    frame_index=None,
                )
            )
            continue

        if img_dim != 4 or lab_dim != 4:
            raise RuntimeError(
                f"Unsupported dimensions for {pair.img_path}: image dim={img_dim}, label dim={lab_dim}. "
                "Expected both 3D or both 4D."
            )

        lab_arr = sitk.GetArrayFromImage(lab_obj)
        # SimpleITK array order for 4D is normally [t, z, y, x].
        num_frames = lab_arr.shape[0]
        if frame_policy == "all":
            frames = list(range(num_frames))
        elif frame_policy == "nonzero-gt":
            frames = [int(t) for t in range(num_frames) if np.any(lab_arr[t] > 0)]
        else:
            raise ValueError(f"Unknown frame_policy: {frame_policy}")

        if len(frames) == 0:
            print(
                f"[WARN] No foreground GT frame found, skipped case: {pair.img_path}",
                file=sys.stderr,
            )
            continue

        for t in frames:
            samples.append(
                Sample(
                    sample_id=f"{pair.case_id}_{t}",  # e.g. A0S9V9_0
                    img_path=pair.img_path,
                    lab_path=pair.lab_path,
                    source_group=pair.source_group,
                    original_case_id=pair.case_id,
                    frame_index=t,
                )
            )

    return samples


def round_fraction_half_even(total: int, numerator: int, denominator: int) -> int:
    """Exact version of Python round(total * numerator / denominator)."""
    q, r = divmod(total * numerator, denominator)
    twice_r = 2 * r
    if twice_r < denominator:
        return q
    if twice_r > denominator:
        return q + 1
    return q if q % 2 == 0 else q + 1


def resolve_counts(
    total: int,
    train_count: Optional[int],
    valid_count: Optional[int],
    test_count: Optional[int],
) -> tuple[int, int, int]:
    if train_count is None:
        train_count = round_fraction_half_even(total, 7, 10)
    if valid_count is None:
        valid_count = round_fraction_half_even(total, 1, 10)
    if test_count is None:
        test_count = total - train_count - valid_count

    if train_count + valid_count + test_count != total:
        raise ValueError(
            f"Split counts do not sum to total: train={train_count}, "
            f"valid={valid_count}, test={test_count}, total={total}"
        )
    return train_count, valid_count, test_count


def resolve_subset_count(train_total: int, count: Optional[int], ratio_num: int, ratio_den: int) -> int:
    if count is not None:
        return count
    return round_fraction_half_even(train_total, ratio_num, ratio_den)


def ensure_empty_or_overwrite(dst_root: Path, overwrite: bool) -> None:
    if dst_root.exists():
        has_content = any(dst_root.iterdir())
        if has_content and not overwrite:
            raise FileExistsError(
                f"Destination already exists and is not empty: {dst_root}\n"
                "Use --overwrite to remove and rebuild it."
            )
        if has_content and overwrite:
            shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)


def make_output_dirs(dst_root: Path, make_finetune_compatible: bool) -> None:
    dirs = [
        dst_root / "train" / "img" / "full",
        dst_root / "train" / "img" / "subset10",
        dst_root / "train" / "img" / "subset20",
        dst_root / "train" / "lab" / "full",
        dst_root / "train" / "lab" / "subset10",
        dst_root / "train" / "lab" / "subset20",
        dst_root / "valid" / "img",
        dst_root / "valid" / "lab",
        dst_root / "test" / "img",
        dst_root / "test" / "lab",
        dst_root / "splits",
    ]

    if make_finetune_compatible:
        for split in ["valid", "test"]:
            for subset in ["subset10", "subset20"]:
                dirs.extend(
                    [
                        dst_root / split / "img" / subset,
                        dst_root / split / "lab" / subset,
                    ]
                )

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def gzip_nii(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    shutil.copystat(src, dst)


def transfer_nifti_as_niigz(src: Path, dst: Path, mode: str) -> None:
    """Copy/link a 3D or 4D NIfTI file to a .nii.gz destination."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.name.endswith(".nii.gz"):
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "hardlink":
            try:
                os.link(src, dst)
            except OSError:
                print(f"[WARN] Hardlink failed, fallback to copy: {src} -> {dst}", file=sys.stderr)
                shutil.copy2(src, dst)
        elif mode == "symlink":
            os.symlink(src.resolve(), dst)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    elif src.name.endswith(".nii"):
        # A .nii.gz file is simply a gzip-compressed .nii file.
        # For .nii inputs we must actually compress; a symlink renamed to .nii.gz is unsafe.
        gzip_nii(src, dst)
    else:
        raise ValueError(f"Unsupported NIfTI file: {src}")


def sitk_write_3d_or_frame(src: Path, dst: Path, frame_index: Optional[int]) -> None:
    sitk, np = import_sitk_numpy()
    img_obj = sitk.ReadImage(str(src))

    if frame_index is None:
        sitk.WriteImage(img_obj, str(dst), True)
        return

    arr = sitk.GetArrayFromImage(img_obj)
    if arr.ndim != 4:
        raise RuntimeError(f"Expected 4D array for frame extraction, got shape {arr.shape}: {src}")
    if frame_index < 0 or frame_index >= arr.shape[0]:
        raise IndexError(f"Frame index {frame_index} out of range for {src}, shape={arr.shape}")

    arr_3d = arr[frame_index]
    out = sitk.GetImageFromArray(arr_3d)

    # Best-effort copy of 3D metadata from the 4D image.
    try:
        spacing = img_obj.GetSpacing()
        origin = img_obj.GetOrigin()
        direction = img_obj.GetDirection()
        dim = img_obj.GetDimension()

        if len(spacing) >= 3:
            out.SetSpacing(tuple(spacing[:3]))
        if len(origin) >= 3:
            out.SetOrigin(tuple(origin[:3]))
        if dim >= 3 and len(direction) == dim * dim:
            mat = np.array(direction, dtype=float).reshape(dim, dim)
            out.SetDirection(tuple(mat[:3, :3].reshape(-1)))
    except Exception as exc:
        print(f"[WARN] Failed to copy metadata for {src}: {exc}", file=sys.stderr)

    dst.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(dst), True)


def write_sample(sample: Sample, img_dir: Path, lab_dir: Path, mode: str) -> tuple[Path, Path]:
    img_dst = img_dir / f"{sample.sample_id}.nii.gz"
    lab_dst = lab_dir / f"{sample.sample_id}_gt.nii.gz"

    if sample.frame_index is None:
        transfer_nifti_as_niigz(sample.img_path, img_dst, mode)
        transfer_nifti_as_niigz(sample.lab_path, lab_dst, mode)
    else:
        sitk_write_3d_or_frame(sample.img_path, img_dst, sample.frame_index)
        sitk_write_3d_or_frame(sample.lab_path, lab_dst, sample.frame_index)

    return img_dst, lab_dst


def write_list_txt(path: Path, samples: Iterable[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.sample_id + "\n")


def count_files(path: Path) -> int:
    return sum(1 for p in path.iterdir() if p.is_file() or p.is_symlink())


def copy_split(
    samples: list[Sample],
    img_dir: Path,
    lab_dir: Path,
    mode: str,
    split_name: str,
    subset_name: str,
    manifest_rows: list[dict],
) -> None:
    for sample in samples:
        img_dst, lab_dst = write_sample(sample, img_dir, lab_dir, mode)
        manifest_rows.append(
            {
                "split": split_name,
                "subset": subset_name,
                "sample_id": sample.sample_id,
                "image_dst": str(img_dst),
                "label_dst": str(lab_dst),
                "source_group": sample.source_group,
                "original_case_id": sample.original_case_id,
                "frame_index": "" if sample.frame_index is None else sample.frame_index,
                "image_src": str(sample.img_path),
                "label_src": str(sample.lab_path),
            }
        )


def write_manifest(dst_root: Path, rows: list[dict]) -> None:
    out_csv = dst_root / "splits" / "manifest.csv"
    fieldnames = [
        "split",
        "subset",
        "sample_id",
        "image_dst",
        "label_dst",
        "source_group",
        "original_case_id",
        "frame_index",
        "image_src",
        "label_src",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split M&Ms raw dataset into FDAS-style train/valid/test folders."
    )
    parser.add_argument("--src", type=Path, default=Path("/data/WPW/datasets/MnM"))
    parser.add_argument("--dst", type=Path, default=Path("/data/WPW/datasets/MnM01"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--sample-unit",
        choices=["case", "volume"],
        default="case",
        help=(
            "case: each raw *_sa NIfTI is one sample, matching 345 -> 242/34/69. "
            "volume: split 4D cases into 3D frames, usually more compatible with original FDAS scripts."
        ),
    )
    parser.add_argument(
        "--frame-policy",
        choices=["nonzero-gt", "all"],
        default="nonzero-gt",
        help="Only used by --sample-unit volume for 4D NIfTI files.",
    )
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--valid-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--subset10-count", type=int, default=None)
    parser.add_argument("--subset20-count", type=int, default=None)
    parser.add_argument("--expect-total", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="For existing 3D .nii.gz files only. .nii files and 4D extracted frames are written/copied as .nii.gz.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove existing destination and rebuild.")
    parser.add_argument(
        "--make-finetune-compatible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also create valid/img/subset10, valid/img/subset20, test/img/subset10, test/img/subset20 "
            "and corresponding lab folders. This matches the current FDAS fine_tune.py path construction."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print split counts; do not copy files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = args.src.resolve()
    dst_root = args.dst.resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    print(f"[INFO] Source root: {src_root}")
    print(f"[INFO] Destination: {dst_root}")
    print(f"[INFO] Seed: {args.seed}")
    print(f"[INFO] Sample unit: {args.sample_unit}")

    pairs = collect_pairs(src_root)
    print(f"[INFO] Found paired raw cases: {len(pairs)}")
    if len(pairs) == 0:
        raise RuntimeError("No paired *_sa.nii(.gz) and *_sa_gt.nii(.gz) files found.")

    if args.sample_unit == "case":
        samples = build_case_samples(pairs)
    else:
        samples = build_volume_samples(pairs, args.frame_policy)

    if args.expect_total is not None and len(samples) != args.expect_total:
        raise RuntimeError(
            f"Expected total samples={args.expect_total}, but got {len(samples)}. "
            "Check --sample-unit or your source dataset."
        )

    sample_ids = [s.sample_id for s in samples]
    if len(sample_ids) != len(set(sample_ids)):
        dup = sorted({x for x in sample_ids if sample_ids.count(x) > 1})[:10]
        raise RuntimeError(f"Duplicate sample IDs found, examples: {dup}")

    rng = random.Random(args.seed)
    rng.shuffle(samples)

    n_train, n_valid, n_test = resolve_counts(
        len(samples), args.train_count, args.valid_count, args.test_count
    )
    train_samples = samples[:n_train]
    valid_samples = samples[n_train : n_train + n_valid]
    test_samples = samples[n_train + n_valid :]

    n_subset10 = resolve_subset_count(n_train, args.subset10_count, 1, 10)
    n_subset20 = resolve_subset_count(n_train, args.subset20_count, 2, 10)
    if n_subset10 > n_subset20:
        raise ValueError("subset10-count cannot be larger than subset20-count.")
    if n_subset20 > n_train:
        raise ValueError("subset20-count cannot be larger than train-count.")

    # Make subset10 nested in subset20 for fair comparison across data ratios.
    train_for_subsets = train_samples.copy()
    rng_subset = random.Random(args.seed + 17)
    rng_subset.shuffle(train_for_subsets)
    subset20_samples = train_for_subsets[:n_subset20]
    subset10_samples = subset20_samples[:n_subset10]

    print("[INFO] Split counts:")
    print(f"       train/full : {len(train_samples)}")
    print(f"       valid      : {len(valid_samples)}")
    print(f"       test       : {len(test_samples)}")
    print(f"       subset10   : {len(subset10_samples)}")
    print(f"       subset20   : {len(subset20_samples)}")

    if args.dry_run:
        print("[INFO] Dry run finished. No files were copied.")
        return

    ensure_empty_or_overwrite(dst_root, args.overwrite)
    make_output_dirs(dst_root, args.make_finetune_compatible)

    manifest_rows: list[dict] = []
    copy_split(
        train_samples,
        dst_root / "train" / "img" / "full",
        dst_root / "train" / "lab" / "full",
        args.mode,
        "train",
        "full",
        manifest_rows,
    )
    copy_split(
        subset10_samples,
        dst_root / "train" / "img" / "subset10",
        dst_root / "train" / "lab" / "subset10",
        args.mode,
        "train",
        "subset10",
        manifest_rows,
    )
    copy_split(
        subset20_samples,
        dst_root / "train" / "img" / "subset20",
        dst_root / "train" / "lab" / "subset20",
        args.mode,
        "train",
        "subset20",
        manifest_rows,
    )
    copy_split(
        valid_samples,
        dst_root / "valid" / "img",
        dst_root / "valid" / "lab",
        args.mode,
        "valid",
        "full",
        manifest_rows,
    )
    copy_split(
        test_samples,
        dst_root / "test" / "img",
        dst_root / "test" / "lab",
        args.mode,
        "test",
        "full",
        manifest_rows,
    )

    if args.make_finetune_compatible:
        for subset_name in ["subset10", "subset20"]:
            copy_split(
                valid_samples,
                dst_root / "valid" / "img" / subset_name,
                dst_root / "valid" / "lab" / subset_name,
                args.mode,
                "valid",
                subset_name,
                manifest_rows,
            )
            copy_split(
                test_samples,
                dst_root / "test" / "img" / subset_name,
                dst_root / "test" / "lab" / subset_name,
                args.mode,
                "test",
                subset_name,
                manifest_rows,
            )

    split_dir = dst_root / "splits"
    write_list_txt(split_dir / "train_full.txt", train_samples)
    write_list_txt(split_dir / "train_subset10.txt", subset10_samples)
    write_list_txt(split_dir / "train_subset20.txt", subset20_samples)
    write_list_txt(split_dir / "valid.txt", valid_samples)
    write_list_txt(split_dir / "test.txt", test_samples)
    write_manifest(dst_root, manifest_rows)

    print("[INFO] Finished writing FDAS dataset layout.")
    print("[INFO] File counts in main folders:")
    checks = [
        ("train/img/full", dst_root / "train" / "img" / "full"),
        ("train/lab/full", dst_root / "train" / "lab" / "full"),
        ("train/img/subset10", dst_root / "train" / "img" / "subset10"),
        ("train/lab/subset10", dst_root / "train" / "lab" / "subset10"),
        ("train/img/subset20", dst_root / "train" / "img" / "subset20"),
        ("train/lab/subset20", dst_root / "train" / "lab" / "subset20"),
        ("valid/img", dst_root / "valid" / "img"),
        ("valid/lab", dst_root / "valid" / "lab"),
        ("test/img", dst_root / "test" / "img"),
        ("test/lab", dst_root / "test" / "lab"),
    ]
    for name, path in checks:
        print(f"       {name:20s}: {count_files(path)}")

    print(f"[INFO] Split lists saved to: {split_dir}")
    print(f"[INFO] Manifest saved to: {split_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
