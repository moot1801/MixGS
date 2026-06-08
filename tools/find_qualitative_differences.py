#!/usr/bin/env python3
"""Find qualitative crop candidates where two render sets differ.

Edit METHOD_A_DIR, METHOD_B_DIR, GT_DIR, and OUTPUT_DIR below for the
common no-argument workflow, or override them with CLI flags.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


# Default paths for quick repeated runs. Override with CLI flags when needed.
METHOD_A_DIR = Path("/home/yeonchanheum/MixGS/output/rubble_mixgs/lightning_logs/version_1/test_legacy_eval/ours_250000/renders")
METHOD_B_DIR = Path("/home/yeonchanheum/MixGS/output/rubble_projected2/lightning_logs/version_2/test_research_eval/ours_250000/renders")
GT_DIR = Path("/home/yeonchanheum/MixGS/output/rubble_final/lightning_logs/version_0/test_research_eval/ours_250000/gt")
OUTPUT_DIR = Path("output/projected_vs_mixgs2")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageTriplet:
    key: str
    method_a: Path
    method_b: Path
    gt: Path


@dataclass(frozen=True)
class Candidate:
    score: float
    key: str
    crop_index: int
    box: Tuple[int, int, int, int]
    method_a: Path
    method_b: Path
    gt: Path
    image_shape: Tuple[int, int]
    diff_score: float = 0.0
    method_b_detail_score: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and crop regions where two rendered image sets differ most."
    )
    parser.add_argument("--method-a", type=Path, default=METHOD_A_DIR, help="First render image directory")
    parser.add_argument("--method-b", type=Path, default=METHOD_B_DIR, help="Second render image directory")
    parser.add_argument("--gt", type=Path, default=GT_DIR, help="Ground-truth image directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--match",
        choices=("name", "sorted"),
        default="name",
        help="Match images by relative file name or by sorted order",
    )
    parser.add_argument(
        "--score-mode",
        choices=("render_diff", "gt_gap", "b_better"),
        default="b_better",
        help=(
            "render_diff uses abs(A-B); gt_gap uses abs(abs(A-GT)-abs(B-GT)); "
            "b_better favors very low B-GT error plus low A quality or strong A-B structure differences"
        ),
    )
    parser.add_argument(
        "--focus-b-detail",
        action="store_true",
        help="Prefer crops where method B has high detail allocation/proxy and differs from method A",
    )
    parser.add_argument(
        "--method-b-detail-map-dir",
        type=Path,
        default=None,
        help="Optional directory containing method B detail/allocation maps matched by image name or relative path",
    )
    parser.add_argument(
        "--detail-source",
        choices=("auto", "map", "image"),
        default="auto",
        help="Use external detail maps, method B image detail proxy, or auto fallback",
    )
    parser.add_argument(
        "--detail-weight",
        type=float,
        default=0.75,
        help="Extra weight for high method B detail regions when --focus-b-detail is enabled",
    )
    parser.add_argument(
        "--detail-percentile",
        type=float,
        default=70.0,
        help="Downweight regions below this method B detail percentile; use 0 to disable",
    )
    parser.add_argument(
        "--focus-crops-per-image",
        type=int,
        default=3,
        help="Minimum crops per image to inspect when --focus-b-detail is enabled",
    )
    parser.add_argument("--top-n", type=int, default=30, help="Number of crop candidates to save")
    parser.add_argument("--crops-per-image", type=int, default=1, help="Max crop candidates per image triplet")
    parser.add_argument("--crop-size", type=int, default=256, help="Crop size in original image pixels")
    parser.add_argument("--zoom-size", type=int, default=512, help="Saved crop panel size in pixels")
    parser.add_argument(
        "--min-separation",
        type=int,
        default=192,
        help="Minimum center distance suppression between crops from the same image",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only scan the first N matched images; 0 scans all")
    parser.add_argument("--recursive", action="store_true", help="Scan image directories recursively")
    parser.add_argument("--no-overview", action="store_true", help="Do not save full-image bbox overview files")
    return parser.parse_args()


def iter_image_files(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def index_by_relative_path(root: Path, recursive: bool) -> Dict[str, Path]:
    return {p.relative_to(root).as_posix(): p for p in iter_image_files(root, recursive)}


def build_triplets(args: argparse.Namespace) -> List[ImageTriplet]:
    for directory in (args.method_a, args.method_b, args.gt):
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")

    a_files = iter_image_files(args.method_a, args.recursive)
    b_files = iter_image_files(args.method_b, args.recursive)
    gt_files = iter_image_files(args.gt, args.recursive)
    if not a_files or not b_files or not gt_files:
        raise RuntimeError("All three directories must contain at least one supported image file")

    if args.match == "sorted":
        count = min(len(a_files), len(b_files), len(gt_files))
        return [
            ImageTriplet(a_files[i].name, a_files[i], b_files[i], gt_files[i])
            for i in range(count)
        ]

    a_index = index_by_relative_path(args.method_a, args.recursive)
    b_index = index_by_relative_path(args.method_b, args.recursive)
    gt_index = index_by_relative_path(args.gt, args.recursive)
    common = sorted(set(a_index) & set(b_index) & set(gt_index))
    if not common:
        common_by_name = sorted(
            set(p.name for p in a_files) & set(p.name for p in b_files) & set(p.name for p in gt_files)
        )
        if not common_by_name:
            raise RuntimeError("No common image names found. Try --match sorted.")
        a_by_name = {p.name: p for p in a_files}
        b_by_name = {p.name: p for p in b_files}
        gt_by_name = {p.name: p for p in gt_files}
        return [ImageTriplet(name, a_by_name[name], b_by_name[name], gt_by_name[name]) for name in common_by_name]

    return [ImageTriplet(key, a_index[key], b_index[key], gt_index[key]) for key in common]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def resize_to(image: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def normalize_map(values: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    data = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if data.size == 0:
        return data
    lo = float(np.percentile(data, low_percentile))
    hi = float(np.percentile(data, high_percentile))
    if hi <= lo + 1e-8:
        hi = float(data.max())
        lo = float(data.min())
    denom = max(hi - lo, 1e-8)
    return np.clip((data - lo) / denom, 0.0, 1.0)


def detail_proxy_from_image(method_b: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(method_b, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    highpass = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0, sigmaY=3.0))
    detail = 0.7 * gradient + 0.3 * highpass
    sigma = max(1.0, min(method_b.shape[:2]) / 250.0)
    return normalize_map(cv2.GaussianBlur(detail, (0, 0), sigmaX=sigma, sigmaY=sigma))


def _detail_map_candidates(root: Path, key: str, method_b_path: Path) -> Iterable[Path]:
    key_path = Path(key)
    names = [key_path]
    if key_path.suffix:
        names.extend(key_path.with_suffix(ext) for ext in IMAGE_EXTENSIONS)
        names.extend([key_path.with_suffix(".npy"), key_path.with_suffix(".npz")])
    else:
        names.extend(Path(str(key_path) + ext) for ext in IMAGE_EXTENSIONS)
        names.extend([Path(str(key_path) + ".npy"), Path(str(key_path) + ".npz")])

    stem = method_b_path.stem
    names.extend(Path(stem + ext) for ext in IMAGE_EXTENSIONS)
    names.extend([Path(stem + ".npy"), Path(stem + ".npz")])

    seen = set()
    for name in names:
        candidate = root / name
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def load_detail_map(args: argparse.Namespace, key: str, method_b_path: Path, shape_hw: Tuple[int, int]) -> np.ndarray | None:
    if args.method_b_detail_map_dir is None:
        return None
    root = args.method_b_detail_map_dir
    if not root.exists():
        raise FileNotFoundError(f"Method B detail map directory does not exist: {root}")

    for path in _detail_map_candidates(root, key, method_b_path):
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() == ".npy":
            detail = np.load(path).astype(np.float32)
        elif path.suffix.lower() == ".npz":
            archive = np.load(path)
            first_key = archive.files[0]
            detail = archive[first_key].astype(np.float32)
        else:
            detail_image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if detail_image is None:
                continue
            if detail_image.ndim == 3:
                detail_image = cv2.cvtColor(detail_image, cv2.COLOR_BGR2GRAY)
            detail = detail_image.astype(np.float32)
        if detail.shape[:2] != shape_hw:
            detail = cv2.resize(detail, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_AREA)
        return normalize_map(detail)
    return None


def method_b_detail_map(method_b: np.ndarray, args: argparse.Namespace, key: str, method_b_path: Path) -> np.ndarray:
    if args.detail_source in ("auto", "map"):
        detail = load_detail_map(args, key, method_b_path, method_b.shape[:2])
        if detail is not None:
            return detail
        if args.detail_source == "map":
            raise FileNotFoundError(f"No method B detail map found for {key} in {args.method_b_detail_map_dir}")
    return detail_proxy_from_image(method_b)


def focused_score_map(
        method_a: np.ndarray,
        method_b: np.ndarray,
        gt: np.ndarray,
        args: argparse.Namespace,
        key: str,
        method_b_path: Path,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    base = score_map(method_a, method_b, gt, args.score_mode)
    if not args.focus_b_detail:
        return base, None, base

    detail = method_b_detail_map(method_b, args, key, method_b_path)
    base_norm = normalize_map(base)
    detail_norm = normalize_map(detail)
    detail_weight = max(0.0, float(args.detail_weight))
    combined = base_norm * (1.0 + detail_weight * detail_norm)

    detail_percentile = float(args.detail_percentile or 0.0)
    if detail_percentile > 0.0:
        threshold = float(np.percentile(detail_norm, np.clip(detail_percentile, 0.0, 100.0)))
        combined = np.where(detail_norm >= threshold, combined, combined * 0.25)

    sigma = max(1.0, min(method_a.shape[:2]) / 250.0)
    combined = cv2.GaussianBlur(combined.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return combined, detail_norm, base


def box_mean(values: np.ndarray | None, box: Tuple[int, int, int, int]) -> float:
    if values is None:
        return 0.0
    x0, y0, x1, y1 = box
    crop = values[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    return float(np.mean(crop))


def score_map(method_a: np.ndarray, method_b: np.ndarray, gt: np.ndarray, mode: str) -> np.ndarray:
    a = method_a.astype(np.float32) / 255.0
    b = method_b.astype(np.float32) / 255.0
    g = gt.astype(np.float32) / 255.0
    if mode == "gt_gap":
        err_a = np.mean(np.abs(a - g), axis=2)
        err_b = np.mean(np.abs(b - g), axis=2)
        diff = np.abs(err_a - err_b)
    elif mode == "b_better":
        err_a = np.mean(np.abs(a - g), axis=2)
        err_b = np.mean(np.abs(b - g), axis=2)
        render_gap = np.mean(np.abs(a - b), axis=2)

        gray_a = cv2.cvtColor(method_a, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_b = cv2.cvtColor(method_b, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        grad_a_x = cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3)
        grad_a_y = cv2.Sobel(gray_a, cv2.CV_32F, 0, 1, ksize=3)
        grad_b_x = cv2.Sobel(gray_b, cv2.CV_32F, 1, 0, ksize=3)
        grad_b_y = cv2.Sobel(gray_b, cv2.CV_32F, 0, 1, ksize=3)
        structure_gap = np.sqrt((grad_a_x - grad_b_x) ** 2 + (grad_a_y - grad_b_y) ** 2)

        b_quality = np.clip((0.20 - err_b) / 0.20, 0.0, 1.0) ** 2
        relative_quality_gap = np.clip(err_a - err_b, 0.0, None)
        structural_difference = 0.5 * render_gap + 0.5 * np.clip(structure_gap, 0.0, 1.0)
        diff = b_quality * (relative_quality_gap + 0.5 * structural_difference)
    else:
        diff = np.mean(np.abs(a - b), axis=2)
    sigma = max(1.0, min(method_a.shape[:2]) / 200.0)
    return cv2.GaussianBlur(diff, (0, 0), sigmaX=sigma, sigmaY=sigma)


def crop_box(center_y: int, center_x: int, crop_size: int, height: int, width: int) -> Tuple[int, int, int, int]:
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    y0 = int(np.clip(center_y - crop_h // 2, 0, height - crop_h))
    x0 = int(np.clip(center_x - crop_w // 2, 0, width - crop_w))
    return x0, y0, x0 + crop_w, y0 + crop_h


def find_boxes(diff: np.ndarray, crop_size: int, count: int, min_separation: int) -> List[Tuple[float, Tuple[int, int, int, int]]]:
    height, width = diff.shape[:2]
    win_h = min(crop_size, height)
    win_w = min(crop_size, width)
    score = cv2.boxFilter(diff, ddepth=-1, ksize=(win_w, win_h), normalize=True)
    work = score.copy()
    boxes: List[Tuple[float, Tuple[int, int, int, int]]] = []
    radius = max(1, int(min_separation))

    for _ in range(max(1, count)):
        _, max_value, _, max_loc = cv2.minMaxLoc(work)
        if not np.isfinite(max_value) or max_value <= 0:
            break
        center_x, center_y = max_loc
        box = crop_box(center_y, center_x, crop_size, height, width)
        boxes.append((float(max_value), box))

        y0 = max(0, center_y - radius)
        y1 = min(height, center_y + radius + 1)
        x0 = max(0, center_x - radius)
        x1 = min(width, center_x + radius + 1)
        work[y0:y1, x0:x1] = -1.0
    return boxes


def sanitize_name(value: str, max_len: int = 90) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    safe = safe.strip("._") or "image"
    return safe[:max_len]


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    bar_h = 34
    out = np.full((image.shape[0] + bar_h, image.shape[1], 3), 255, dtype=np.uint8)
    out[bar_h:, :, :] = image
    cv2.putText(out, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def crop_and_resize(image: np.ndarray, box: Tuple[int, int, int, int], size: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)


def heatmap_crop(diff: np.ndarray, box: Tuple[int, int, int, int], size: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = diff[y0:y1, x0:x1]
    denom = max(float(crop.max()), 1e-8)
    normalized = np.clip(crop / denom * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return cv2.resize(colored, (size, size), interpolation=cv2.INTER_CUBIC)


def draw_overview(image: np.ndarray, diff: np.ndarray, box: Tuple[int, int, int, int], label: str) -> np.ndarray:
    denom = max(float(diff.max()), 1e-8)
    heat = np.clip(diff / denom * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(image, 0.65, heat, 0.35, 0.0)
    x0, y0, x1, y1 = box
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 3)
    return add_label(overlay, label)


def save_candidate(candidate: Candidate, rank: int, output_dir: Path, args: argparse.Namespace, zoom_size: int, save_overview: bool) -> None:
    method_a = read_image(candidate.method_a)
    method_b = resize_to(read_image(candidate.method_b), method_a.shape[:2])
    gt = resize_to(read_image(candidate.gt), method_a.shape[:2])
    diff, detail, base_diff = focused_score_map(method_a, method_b, gt, args, candidate.key, candidate.method_b)

    box = candidate.box
    stem = f"{rank:03d}_{sanitize_name(candidate.key)}_crop{candidate.crop_index:02d}_score{candidate.score:.5f}"
    crop_panels = [
        add_label(crop_and_resize(method_a, box, zoom_size), "method_a"),
        add_label(crop_and_resize(method_b, box, zoom_size), "method_b"),
        add_label(crop_and_resize(gt, box, zoom_size), "gt"),
        add_label(heatmap_crop(diff, box, zoom_size), "focused score" if args.focus_b_detail else f"{args.score_mode} score"),
    ]
    if detail is not None:
        crop_panels.append(add_label(heatmap_crop(detail, box, zoom_size), "method_b detail"))
    if args.focus_b_detail:
        crop_panels.append(add_label(heatmap_crop(base_diff, box, zoom_size), f"{args.score_mode} diff"))
    side_by_side = np.concatenate(crop_panels, axis=1)
    cv2.imwrite(str(output_dir / f"{stem}_crop.png"), side_by_side)

    if save_overview:
        overview_panels = [
            draw_overview(method_a, diff, box, "method_a overview"),
            draw_overview(method_b, diff, box, "method_b overview"),
            draw_overview(gt, diff, box, "gt overview"),
        ]
        min_h = min(panel.shape[0] for panel in overview_panels)
        resized = [cv2.resize(panel, (int(panel.shape[1] * min_h / panel.shape[0]), min_h)) for panel in overview_panels]
        overview = np.concatenate(resized, axis=1)
        cv2.imwrite(str(output_dir / f"{stem}_overview.png"), overview)


def scan_candidates(triplets: Sequence[ImageTriplet], args: argparse.Namespace) -> List[Candidate]:
    candidates: List[Candidate] = []
    limit = int(args.limit or 0)
    scan_triplets = triplets[:limit] if limit > 0 else triplets

    for index, triplet in enumerate(scan_triplets, start=1):
        method_a = read_image(triplet.method_a)
        method_b = resize_to(read_image(triplet.method_b), method_a.shape[:2])
        gt = resize_to(read_image(triplet.gt), method_a.shape[:2])
        diff, detail, base_diff = focused_score_map(method_a, method_b, gt, args, triplet.key, triplet.method_b)
        crops_per_image = int(args.crops_per_image)
        if args.focus_b_detail:
            crops_per_image = max(crops_per_image, int(args.focus_crops_per_image))
        for crop_index, (score, box) in enumerate(
            find_boxes(diff, args.crop_size, crops_per_image, args.min_separation),
            start=1,
        ):
            candidates.append(
                Candidate(
                    score=score,
                    key=triplet.key,
                    crop_index=crop_index,
                    box=box,
                    method_a=triplet.method_a,
                    method_b=triplet.method_b,
                    gt=triplet.gt,
                    image_shape=method_a.shape[:2],
                    diff_score=box_mean(base_diff, box),
                    method_b_detail_score=box_mean(detail, box),
                )
            )
        if index % 50 == 0:
            print(f"Scanned {index}/{len(scan_triplets)} image triplets")

    return sorted(candidates, key=lambda item: item.score, reverse=True)[:max(1, int(args.top_n))]


def write_manifest(candidates: Sequence[Candidate], output_dir: Path) -> None:
    with (output_dir / "manifest.csv").open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "rank", "score", "diff_score", "method_b_detail_score", "key", "crop_index",
            "x0", "y0", "x1", "y1", "height", "width", "method_a", "method_b", "gt",
        ])
        for rank, candidate in enumerate(candidates, start=1):
            x0, y0, x1, y1 = candidate.box
            height, width = candidate.image_shape
            writer.writerow([
                rank,
                f"{candidate.score:.8f}",
                f"{candidate.diff_score:.8f}",
                f"{candidate.method_b_detail_score:.8f}",
                candidate.key,
                candidate.crop_index,
                x0,
                y0,
                x1,
                y1,
                height,
                width,
                candidate.method_a,
                candidate.method_b,
                candidate.gt,
            ])


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    triplets = build_triplets(args)
    print(f"Matched {len(triplets)} image triplets")
    candidates = scan_candidates(triplets, args)
    for rank, candidate in enumerate(candidates, start=1):
        save_candidate(
            candidate,
            rank,
            args.output,
            args,
            max(32, int(args.zoom_size)),
            save_overview=not args.no_overview,
        )
    write_manifest(candidates, args.output)
    print(f"Saved {len(candidates)} candidates to {args.output}")


if __name__ == "__main__":
    main()
