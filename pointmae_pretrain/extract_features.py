"""Extract 768-d geometric features from pretrained Point-MAE encoder.

Usage (from project root):
    python -m pointmae_pretrain.extract_features \
        --checkpoint outputs/pretrain_custom/best.pth \
        --data_dir  data/all_parts/ \
        --output    outputs/features/
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml

from .model import PointMAEPretrain
from .transforms import pc_normalize, pca_align, random_sample


def parse_args():
    parser = argparse.ArgumentParser("Point-MAE feature extraction")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to pretrained .pth"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing .npy point cloud files",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/features/",
        help="Directory to save features",
    )
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pca_align", action="store_true", default=True)
    parser.add_argument("--no_pca_align", dest="pca_align", action="store_false")
    parser.add_argument(
        "--size_file", type=str, default="",
        help="Optional JSON mapping filename -> [L, W, H] for size-aware fusion",
    )
    parser.add_argument("--size_lambda", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> PointMAEPretrain:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    model = PointMAEPretrain(model_cfg=cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def load_point_cloud(
    path: str, npoints: int, do_pca: bool = True
) -> np.ndarray:
    points = np.load(path).astype(np.float32)[:, :3]
    points = random_sample(points, npoints)
    if do_pca:
        points = pca_align(points)
    points = pc_normalize(points)
    return points


def main():
    args = parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, device)

    npy_files = sorted(
        f for f in os.listdir(args.data_dir) if f.endswith(".npy")
    )
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {args.data_dir}")
    print(f"Found {len(npy_files)} point cloud files.")

    all_features = []
    batch_buf = []
    batch_idx = []

    for i, fname in enumerate(npy_files):
        pc = load_point_cloud(
            os.path.join(args.data_dir, fname),
            npoints=args.npoints,
            do_pca=args.pca_align,
        )
        batch_buf.append(pc)
        batch_idx.append(i)

        if len(batch_buf) == args.batch_size or i == len(npy_files) - 1:
            pts = torch.from_numpy(np.stack(batch_buf)).float().to(device)
            feat = model.extract_feature(pts)  # (B, 768)
            all_features.append(feat.cpu().numpy())
            batch_buf.clear()
            batch_idx.clear()
            if (i + 1) % (args.batch_size * 10) == 0 or i == len(npy_files) - 1:
                print(f"  processed {i + 1}/{len(npy_files)}")

    features = np.concatenate(all_features, axis=0)  # (N, 768)

    # ── optional size-aware fusion ──
    if args.size_file and os.path.exists(args.size_file):
        print(f"Applying size-aware fusion (lambda={args.size_lambda}) ...")
        with open(args.size_file, "r", encoding="utf-8") as f:
            size_map = json.load(f)
        size_vectors = []
        for fname in npy_files:
            lwh = size_map.get(fname, size_map.get(os.path.splitext(fname)[0], [1, 1, 1]))
            size_vectors.append(np.log1p(lwh).astype(np.float32))
        size_arr = np.stack(size_vectors)  # (N, 3)
        size_arr = size_arr * args.size_lambda
        features = np.concatenate([features, size_arr], axis=1)  # (N, 771)
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms

    # ── save ──
    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, "features.npy"), features)
    with open(os.path.join(args.output, "filenames.json"), "w", encoding="utf-8") as f:
        json.dump(npy_files, f, ensure_ascii=False, indent=2)

    print(f"Saved features shape={features.shape} to {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
