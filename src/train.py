import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import TabularReadmissionNet
from preprocess import build_preprocessor, prepare_features


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def detect_target_column(df: pd.DataFrame) -> str:
    candidates = [
        "readmitted_30d",
        "readmitted_30days",
        "readmitted_30_days",
        "readmitted",
        "target",
        "label",
        "y",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    # Fallback: pick a column name that looks like a readmission label.
    lowered = {c: c.lower() for c in df.columns}
    for col, name in lowered.items():
        if "readmit" in name and ("30" in name or "day" in name):
            return col

    raise ValueError(
        "Could not detect target column. Expected one of: "
        f"{', '.join(candidates)}. Please rename your label column."
    )


def to_binary_targets(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.astype(int).to_numpy()
    if np.issubdtype(series.dtype, np.number):
        return (series.astype(float) > 0).astype(int).to_numpy()

    cleaned = series.astype(str).str.strip().str.lower()
    positive = {"1", "true", "yes", "y", "readmitted", "positive"}
    return cleaned.isin(positive).astype(int).to_numpy()


def to_tensor(x) -> torch.Tensor:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    return metrics


def tune_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.linspace(0.1, 0.9, 161)
    best_threshold = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        score = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(thr)
    return best_threshold


def train_model(
    train_path: Path,
    artifacts_dir: Path,
    target_col: Optional[str] = None,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
) -> Tuple[Path, Dict]:
    df = pd.read_csv(train_path)
    if target_col is None:
        target_col = detect_target_column(df)

    y = to_binary_targets(df[target_col])
    x_df = prepare_features(df.drop(columns=[target_col]))

    x_train_df, x_val_df, y_train, y_val = train_test_split(
        x_df, y, test_size=0.2, random_state=SEED, stratify=y
    )

    fitted = build_preprocessor(x_train_df)
    x_train = to_tensor(fitted.transformer.transform(x_train_df))
    x_val = to_tensor(fitted.transformer.transform(x_val_df))
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val_t), batch_size=batch_size, shuffle=False)

    input_dim = x_train.shape[1]
    model = TabularReadmissionNet(input_dim=input_dim)

    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        val_probs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                val_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

        mean_val_loss = float(np.mean(val_losses))
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(x_val)).cpu().numpy()
    best_threshold = tune_threshold_for_f1(y_val, val_prob)
    metrics = compute_metrics(y_val, val_prob, threshold=best_threshold)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.pt"
    preprocessor_path = artifacts_dir / "preprocessor.joblib"
    config_path = artifacts_dir / "config.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(input_dim),
            "hidden_dims": [128, 64],
            "dropout": 0.2,
        },
        model_path,
    )
    joblib.dump(fitted, preprocessor_path)

    config = {
        "target_col": target_col,
        "threshold": float(best_threshold),
        "features_in_train": x_df.columns.tolist(),
        "metrics": metrics,
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return model_path, config


def main():
    parser = argparse.ArgumentParser(description="Train tabular readmission model.")
    parser.add_argument("--train-path", type=str, default="data/train.csv")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--target-col", type=str, default=None)
    args = parser.parse_args()

    model_path, config = train_model(
        train_path=Path(args.train_path),
        artifacts_dir=Path(args.artifacts_dir),
        target_col=args.target_col,
    )
    print(f"Model saved to: {model_path}")
    print(f"Validation metrics: {config['metrics']}")


if __name__ == "__main__":
    main()
