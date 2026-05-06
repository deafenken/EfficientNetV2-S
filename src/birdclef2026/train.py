import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config, resolve_data_root
from .dataset import BirdCLEFDataset
from .metadata import build_train_metadata, get_target_columns, split_train_val
from .metrics import labelwise_auc
from .model import create_model
from .utils import ensure_dir, get_device, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--debug", action="store_true", help="Run a tiny, fast smoke training job.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    data_root = resolve_data_root(config)
    output_dir = ensure_dir(config.get("paths", {}).get("output_dir", "outputs/baseline"))
    target_columns = get_target_columns(data_root, config)
    metadata = build_train_metadata(data_root, config, target_columns)
    if metadata.empty:
        raise RuntimeError(
            "No training rows found. Check data_root, train.csv, taxonomy.csv, and train_audio files."
        )

    if args.debug:
        metadata = metadata.sample(min(len(metadata), 256), random_state=int(config.get("seed", 42)))
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0
        config["data"]["max_train_samples"] = 192
        config["data"]["max_val_samples"] = 64

    train_df, val_df = split_train_val(
        metadata,
        val_fraction=float(config.get("data", {}).get("val_fraction", 0.2)),
        seed=int(config.get("seed", 42)),
    )

    max_train = config.get("data", {}).get("max_train_samples")
    max_val = config.get("data", {}).get("max_val_samples")
    if max_train:
        train_df = train_df.sample(min(len(train_df), int(max_train)), random_state=int(config.get("seed", 42)))
    if max_val:
        val_df = val_df.sample(min(len(val_df), int(max_val)), random_state=int(config.get("seed", 42)))

    train_ds = BirdCLEFDataset(train_df, target_columns, config["audio"], random_crop=True)
    val_ds = BirdCLEFDataset(val_df, target_columns, config["audio"], random_crop=False)
    train_cfg = config.get("train", {})
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )

    device = get_device()
    model_cfg = config.get("model", {})
    model = create_model(
        num_classes=len(target_columns),
        name=model_cfg.get("name", "resnet18"),
        pretrained=bool(model_cfg.get("pretrained", False)),
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    best_auc = -np.inf
    metrics_path = output_dir / "metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_auc"])
        writer.writeheader()

    epochs = int(train_cfg.get("epochs", 3))
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            grad_clip_norm=float(train_cfg.get("grad_clip_norm", 0) or 0),
            amp=bool(train_cfg.get("amp", True)),
            epoch=epoch,
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        print(
            f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_auc={val_auc:.5f}"
        )
        score_for_selection = val_auc if not np.isnan(val_auc) else -val_loss
        if score_for_selection > best_auc:
            best_auc = score_for_selection
            save_checkpoint(output_dir / "best.pt", model, target_columns, config, epoch, row)
            print(f"saved {output_dir / 'best.pt'}")

    save_checkpoint(output_dir / "last.pt", model, target_columns, config, epochs, row)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip_norm, amp, epoch):
    model.train()
    total_loss = 0.0
    total_examples = 0
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for features, targets in progress:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(features)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = features.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=total_loss / max(total_examples, 1))
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    ys = []
    ps = []
    for features, targets in tqdm(loader, desc="valid", leave=False):
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, targets)
        probs = torch.sigmoid(logits)
        batch_size = features.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        ys.append(targets.detach().cpu().numpy())
        ps.append(probs.detach().cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_score = np.concatenate(ps, axis=0)
    return total_loss / max(total_examples, 1), labelwise_auc(y_true, y_score)


def save_checkpoint(path, model, target_columns, config, epoch, metrics):
    path = Path(path)
    payload = {
        "model_state": model.state_dict(),
        "target_columns": list(target_columns),
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


if __name__ == "__main__":
    main()

