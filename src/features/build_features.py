from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger("features")


@dataclass(frozen=True)
class FeatureSpec:
    """
    Defines how we build features for this dataset.
    Keeping this explicit makes it easy to reuse the same pattern in other projects.
    """
    table_name: str = "churn_dataset"
    id_col: str = "customer_id"
    target_col: str = "churn"
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42


@dataclass(frozen=True)
class SplitData:
    """
    Holds train/val/test splits and metadata.
    """
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    categorical_cols: List[str]
    numeric_cols: List[str]


def load_from_postgres(table_name: str) -> pd.DataFrame:
    """
    Load a table from Postgres into a pandas DataFrame.
    For legacy firms, reading from SQL is very realistic.
    """
    if not settings.database_url:
        raise ValueError("DATABASE_URL is empty. Check your .env file.")

    engine = create_engine(settings.database_url, future=True)

    logger.info(f"Loading table '{table_name}' from Postgres...")
    df = pd.read_sql_table(table_name, con=engine)

    # Convert column names to plain Python strings (important for sklearn)
    df.columns = [str(c) for c in df.columns]

    # Also ensure column names are stripped/lowercased if needed (optional)
    # df.columns = [c.strip() for c in df.columns]

    logger.info(f"Loaded dataframe: {df.shape[0]} rows x {df.shape[1]} cols")
    logger.info(f"Columns (post-normalize): {list(df.columns)}")
    return df


def infer_column_types(df: pd.DataFrame, id_col: str, target_col: str) -> Tuple[List[str], List[str]]:
    """
    Decide which columns are categorical vs numeric based on pandas dtypes.

    This version is robust: it will silently ignore the id_col or target_col
    if they are not present in the provided DataFrame (useful because we often
    drop identifiers before building feature-only DataFrames).
    """
    # Drop id/target if present, but don't raise if they are missing
    cols_to_drop = [c for c in (id_col, target_col) if c in df.columns]
    feature_df = df.drop(columns=cols_to_drop)

    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()

    # Safety: if something is neither object nor number (rare), treat it as categorical
    other_cols = [c for c in feature_df.columns if c not in categorical_cols and c not in numeric_cols]
    categorical_cols.extend(other_cols)

    return categorical_cols, numeric_cols


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    """
    Build a preprocessing transformer that:
    - Imputes missing numeric values with median, then scales
    - Imputes missing categoricals with most_frequent, then one-hot encodes

    Even though our ETL filled most values already, we keep this for robustness:
    in real systems, new data often has unexpected missingness.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def make_splits(spec: FeatureSpec = FeatureSpec()) -> SplitData:
    """
    Load data, split into train/val/test, infer column types.
    We split with stratification so churn class balance is maintained.
    """
    df = load_from_postgres(spec.table_name)

    # Basic checks
    for col in [spec.id_col, spec.target_col]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' missing from table '{spec.table_name}'.")

    y = df[spec.target_col].astype(int)
    X = df.drop(columns=[spec.target_col])

    # First split: train vs temp (val+test)
    temp_size = spec.val_size + spec.test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=spec.random_state,
        stratify=y,
    )

    # Second split: temp -> val and test
    # Compute test proportion inside the temp chunk
    test_prop_in_temp = spec.test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_prop_in_temp,
        random_state=spec.random_state,
        stratify=y_temp,
    )

    # Drop identifier from model features, but keep it available for later (scoring/automation)
    # For training, the model shouldn't learn from customer_id.
    if spec.id_col in X_train.columns:
        X_train_model = X_train.drop(columns=[spec.id_col])
        X_val_model = X_val.drop(columns=[spec.id_col])
        X_test_model = X_test.drop(columns=[spec.id_col])
    else:
        X_train_model, X_val_model, X_test_model = X_train, X_val, X_test

    categorical_cols, numeric_cols = infer_column_types(
        df=pd.concat([X_train_model, y_train], axis=1).assign(**{spec.target_col: y_train}),
        id_col=spec.id_col if spec.id_col in X_train.columns else "__missing__",
        target_col=spec.target_col,
    )

    logger.info(f"Split sizes: train={len(X_train_model)}, val={len(X_val_model)}, test={len(X_test_model)}")
    logger.info(f"Inferred columns: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

    return SplitData(
        X_train=X_train_model,
        X_val=X_val_model,
        X_test=X_test_model,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )
