"""Point-MAE self-supervised pretraining — standalone script.

Usage (from project root):
    python -m pointmae_pretrain.train --config configs/pretrain_custom.yaml
"""

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
import yaml
from timm.scheduler import CosineLRScheduler

from .dataset import build_dataloader
from .model import PointMAEPretrain

# ───────────────────────── helpers ─────────────────────────


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def create_logger(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("pointmae_pretrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(
        os.path.join(save_dir, "train.log"), encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def save_checkpoint(path, model, optimizer, epoch, best_metric, cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", -1)) + 1


# ───────────────────────── training loops ─────────────────────────


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    for points in loader:
        points = points.to(device, non_blocking=True)
        loss = model(points)
        try:
            loss.backward()
        except Exception:
            loss = loss.mean()
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_meter.update(loss.item(), n=points.shape[0])
    return loss_meter.avg


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    loss_meter = AverageMeter()
    for points in loader:
        points = points.to(device, non_blocking=True)
        loss = model(points)
        loss_meter.update(loss.item(), n=points.shape[0])
    return loss_meter.avg


# ───────────────────────── main ─────────────────────────


def parse_args():
    parser = argparse.ArgumentParser("Point-MAE pretrain")
    parser.add_argument("--config", type=str, default="configs/pretrain_custom.yaml")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--train_list", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.save_dir is not None:
        cfg["save"]["dir"] = args.save_dir
    if args.train_list is not None:
        cfg["data"]["train_list"] = args.train_list
    if args.val_list is not None:
        cfg["data"]["val_list"] = args.val_list

    seed_everything(cfg.get("seed", 42))

    save_dir = cfg["save"]["dir"]
    os.makedirs(save_dir, exist_ok=True)
    logger = create_logger(save_dir)

    with open(os.path.join(save_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    device = get_device(cfg.get("device", "auto"))
    logger.info("device=%s", device)

    # ── data ──
    data_cfg = cfg["data"]
    train_loader = build_dataloader(
        list_file=data_cfg["train_list"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        shuffle=True,
        npoints=data_cfg["npoints"],
        root=data_cfg.get("root"),
        normalize=data_cfg.get("normalize", True),
        pca_align_flag=data_cfg.get("pca_align", True),
        train=True,
    )

    val_loader = None
    val_list = data_cfg.get("val_list", "")
    if val_list and os.path.exists(val_list):
        val_loader = build_dataloader(
            list_file=val_list,
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["train"]["num_workers"],
            shuffle=False,
            npoints=data_cfg["npoints"],
            root=data_cfg.get("root"),
            normalize=data_cfg.get("normalize", True),
            pca_align_flag=data_cfg.get("pca_align", True),
            train=False,
        )

    # ── model ──
    model = PointMAEPretrain(model_cfg=cfg["model"]).to(device)

    # ── optimizer + scheduler ──
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=train_cfg["epochs"],
        t_mul=1,
        lr_min=train_cfg.get("min_lr", 1e-6),
        decay_rate=0.1,
        warmup_lr_init=1e-6,
        warmup_t=train_cfg.get("warmup_epochs", 10),
        cycle_limit=1,
        t_in_epochs=True,
    )

    # ── resume ──
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer)
        logger.info("resumed from %s at epoch=%d", args.resume, start_epoch)

    # ── training ──
    for epoch in range(start_epoch, train_cfg["epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step(epoch)
        elapsed = time.time() - t0
        logger.info(
            "epoch=%d  train_loss=%.6f  lr=%.2e  time=%.1fs",
            epoch,
            train_loss,
            optimizer.param_groups[0]["lr"],
            elapsed,
        )

        metric = train_loss
        eval_every = cfg["save"].get("eval_every", 1)
        if val_loader is not None and (epoch + 1) % eval_every == 0:
            val_loss = validate_one_epoch(model, val_loader, device)
            metric = val_loss
            logger.info("epoch=%d  val_loss=%.6f", epoch, val_loss)

        save_checkpoint(
            os.path.join(save_dir, "last.pth"),
            model, optimizer, epoch, best_val, cfg,
        )
        if metric < best_val:
            best_val = metric
            save_checkpoint(
                os.path.join(save_dir, "best.pth"),
                model, optimizer, epoch, best_val, cfg,
            )
            logger.info("new best checkpoint — metric=%.6f", best_val)

    logger.info("training finished. best_metric=%.6f", best_val)


if __name__ == "__main__":
    main()
