from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FittedPreprocessor:
    transformer: ColumnTransformer
    feature_names: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # Remove high-cardinality identifier columns that typically hurt generalization.
    id_like_cols = [c for c in x.columns if c.lower().endswith("_id") or c.lower() == "id"]
    x = x.drop(columns=id_like_cols, errors="ignore")

    # Expand admission date into cyclic-like categorical/time features.
    if "admission_date" in x.columns:
        dt = pd.to_datetime(x["admission_date"], errors="coerce")
        x["admission_month"] = dt.dt.month
        x["admission_day"] = dt.dt.day
        x["admission_dow"] = dt.dt.day_name()
        x = x.drop(columns=["admission_date"])

    return x


def split_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(df_features: pd.DataFrame) -> FittedPreprocessor:
    numeric_cols, categorical_cols = split_feature_types(df_features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    transformer.fit(df_features)
    feature_names = transformer.get_feature_names_out().tolist()

    return FittedPreprocessor(
        transformer=transformer,
        feature_names=feature_names,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
