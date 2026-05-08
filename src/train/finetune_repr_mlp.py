"""Train a small 2-layer MLP on frozen representation CSVs (regression / binary / multiclass)."""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


def _activation(name: str) -> nn.Module:
    n = name.lower()
    if n == "gelu":
        return nn.GELU()
    if n == "silu":
        return nn.SiLU()
    if n == "mish":
        return nn.Mish()
    raise ValueError(f"Unknown activation {name!r}; use gelu, silu, or mish")


class RepresentationMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = _activation(activation)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


class ReprCSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feat_cols: Sequence[str], label_col: str, task: str, class_to_idx=None):
        self.x = df[list(feat_cols)].values.astype(np.float32)
        self.task = task
        self.class_to_idx = class_to_idx
        if task == "regression":
            self.y = df[label_col].values.astype(np.float32)
        elif task == "binary":
            self.y = df[label_col].values.astype(np.float32)
        else:
            labs = df[label_col].values
            if class_to_idx is None:
                raise ValueError("multiclass requires class_to_idx")
            self.y = np.array([class_to_idx[str(v)] for v in labs], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i])


def _infer_feature_columns(df: pd.DataFrame, repr_prefix: str, repr_cols: List[str] | None) -> List[str]:
    if repr_cols:
        return repr_cols
    cols = [c for c in df.columns if c.startswith(repr_prefix) or c.startswith("dim_")]
    if not cols:
        cols = [c for c in df.columns if c.startswith("repr_dim_")]
    cols = sorted(cols, key=lambda c: int("".join(filter(str.isdigit, c)) or 0))
    if not cols:
        raise ValueError(
            "No representation columns found. Pass --repr_cols or use columns named dim_*, repr_dim_*, "
            f"or prefix {repr_prefix!r}."
        )
    return cols


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("repr_csv", help="CSV with representation columns and a label column")
    p.add_argument("--label_column", required=True)
    p.add_argument("--task", choices=("regression", "binary", "multiclass"), required=True)
    p.add_argument("--repr_prefix", default="repr_dim_", help="Prefix for wide repr columns if not using --repr_cols")
    p.add_argument(
        "--repr_cols",
        nargs="*",
        default=None,
        help="Explicit list of feature columns (overrides prefix inference)",
    )
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--activation", default="gelu", choices=("gelu", "silu", "mish"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_path", default="repr_mlp.pt")
    p.add_argument("--meta_json", default="repr_mlp_meta.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.repr_csv)
    feat_cols = _infer_feature_columns(df, args.repr_prefix, args.repr_cols)
    in_dim = len(feat_cols)

    class_to_idx = None
    num_classes = 1
    if args.task == "multiclass":
        raw_labs = df[args.label_column].values
        uniq = sorted({str(v) for v in raw_labs})
        class_to_idx = {lab: i for i, lab in enumerate(uniq)}
        num_classes = len(class_to_idx)

    ds_full = ReprCSVDataset(df, feat_cols, args.label_column, args.task, class_to_idx=class_to_idx)
    n_val = max(1, int(len(ds_full) * args.val_frac))
    n_train = len(ds_full) - n_val
    train_ds, val_ds = random_split(
        ds_full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    out_dim = 1 if args.task in ("regression", "binary") else num_classes
    model = RepresentationMLP(in_dim, args.hidden_dim, out_dim, activation=args.activation)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.task == "regression":
        loss_fn = nn.MSELoss()
    elif args.task == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    def step_batch(xb, yb):
        logits = model(xb)
        if args.task == "binary":
            logits = logits.squeeze(-1)
            return loss_fn(logits, yb.float())
        if args.task == "multiclass":
            return loss_fn(logits, yb.long())
        return loss_fn(logits.squeeze(-1), yb.float())

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = step_batch(xb, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                va_loss += step_batch(xb, yb).item()
        tr_loss /= max(1, len(train_loader))
        va_loss /= max(1, len(val_loader))
        print(f"epoch {epoch+1}/{args.epochs} train_loss={tr_loss:.6f} val_loss={va_loss:.6f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), args.save_path)

    meta = {
        "task": args.task,
        "label_column": args.label_column,
        "feature_columns": feat_cols,
        "in_dim": in_dim,
        "hidden_dim": args.hidden_dim,
        "activation": args.activation,
        "class_to_idx": class_to_idx,
    }
    with open(args.meta_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved best checkpoint to {args.save_path} (val_loss={best_val:.6f}); metadata → {args.meta_json}")


if __name__ == "__main__":
    main()
