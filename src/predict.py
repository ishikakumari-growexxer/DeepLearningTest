import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from model import TabularReadmissionNet
from preprocess import prepare_features


def load_artifacts(artifacts_dir: Path):
    model_ckpt = torch.load(artifacts_dir / "model.pt", map_location="cpu")
    preprocessor = joblib.load(artifacts_dir / "preprocessor.joblib")
    with (artifacts_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    model = TabularReadmissionNet(
        input_dim=model_ckpt["input_dim"],
        hidden_dims=tuple(model_ckpt.get("hidden_dims", [128, 64])),
        dropout=float(model_ckpt.get("dropout", 0.2)),
    )
    model.load_state_dict(model_ckpt["model_state_dict"])
    model.eval()
    return model, preprocessor, config


def maybe_autotrain(repo_root: Path, artifacts_dir: Path):
    required = ["model.pt", "preprocessor.joblib", "config.json"]
    if all((artifacts_dir / x).exists() for x in required):
        return

    train_csv = repo_root / "data" / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            "Artifacts missing and no data/train.csv found. "
            "Run training first: python src/train.py --train-path data/train.csv"
        )

    print("Artifacts not found. Training a model from data/train.csv ...")
    from train import train_model

    train_model(train_path=train_csv, artifacts_dir=artifacts_dir)


def infer(input_csv: Path, output_csv: Path, artifacts_dir: Path):
    repo_root = Path(__file__).resolve().parents[1]
    maybe_autotrain(repo_root=repo_root, artifacts_dir=artifacts_dir)
    model, fitted_preprocessor, config = load_artifacts(artifacts_dir)

    df = pd.read_csv(input_csv)
    df = prepare_features(df)
    features = config.get("features_in_train", list(df.columns))
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
    x_df = df[features]

    x = fitted_preprocessor.transformer.transform(x_df)
    if hasattr(x, "toarray"):
        x = x.toarray()
    x_t = torch.tensor(np.asarray(x), dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(x_t)).cpu().numpy()
    threshold = float(config.get("threshold", 0.5))
    pred = (prob >= threshold).astype(int)

    out = pd.DataFrame(
        {
            "prediction": pred,
            "probability": prob,
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run 30-day readmission prediction.")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV for inference")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV path")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = repo_root / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = repo_root / artifacts_dir

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        raise SystemExit(1)

    infer(input_csv=input_path, output_csv=output_path, artifacts_dir=artifacts_dir)


if __name__ == "__main__":
    main()
