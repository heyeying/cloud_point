import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import pc_normalize, pca_align, random_sample, scale_and_translate


class PointCloudDataset(Dataset):
    def __init__(
        self,
        list_file: str,
        npoints: int = 1024,
        root: Optional[str] = None,
        normalize: bool = True,
        pca_align_flag: bool = True,
        train: bool = True,
    ) -> None:
        super().__init__()
        self.npoints = npoints
        self.root = root
        self.normalize = normalize
        self.pca_align_flag = pca_align_flag
        self.train = train

        with open(list_file, "r", encoding="utf-8") as f:
            lines = [x.strip() for x in f if x.strip()]
        self.paths = [self._resolve_path(p, list_file) for p in lines]

    def _resolve_path(self, path: str, list_file: str) -> str:
        if os.path.isabs(path):
            return path
        if self.root is not None:
            return os.path.join(self.root, path)
        return os.path.join(os.path.dirname(list_file), path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            points = np.load(path)
        elif ext == ".txt":
            points = np.loadtxt(path)
        else:
            raise ValueError(f"Unsupported point cloud format: {ext}")

        points = points[:, :3].astype(np.float32)
        points = random_sample(points, self.npoints)
        if self.pca_align_flag:
            points = pca_align(points)
        if self.normalize:
            points = pc_normalize(points)
        if self.train:
            points = scale_and_translate(points)
        return torch.from_numpy(points).float()


def build_dataloader(
    list_file: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    npoints: int = 1024,
    root: Optional[str] = None,
    normalize: bool = True,
    pca_align_flag: bool = True,
    train: bool = True,
) -> DataLoader:
    dataset = PointCloudDataset(
        list_file=list_file,
        npoints=npoints,
        root=root,
        normalize=normalize,
        pca_align_flag=pca_align_flag,
        train=train,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=train,
        pin_memory=True,
    )
