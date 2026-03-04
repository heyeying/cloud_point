from typing import Tuple

import numpy as np


def random_sample(points: np.ndarray, npoints: int) -> np.ndarray:
    n = points.shape[0]
    if n >= npoints:
        idx = np.random.choice(n, npoints, replace=False)
    else:
        idx = np.random.choice(n, npoints, replace=True)
    return points[idx]


def pc_normalize(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
    return points


def pca_align(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return centered @ basis


def scale_and_translate(
    points: np.ndarray,
    scale_low: float = 2.0 / 3.0,
    scale_high: float = 3.0 / 2.0,
    translate_range: float = 0.2,
) -> np.ndarray:
    scale = np.random.uniform(scale_low, scale_high, size=(1, 3))
    shift = np.random.uniform(-translate_range, translate_range, size=(1, 3))
    return points * scale + shift
