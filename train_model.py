"""
train_model.py
================
Trains a regression model that predicts how long a traffic incident will
take to resolve (minutes from start_datetime to closure), based on the
Astram event dataset.

This corresponds to "Module B: AI Impact & Duration Predictor" in the
Astram Guardian concept note.

USAGE
-----
    python train_model.py --csv /path/to/Astram_event_data.csv

Outputs (written to --outdir, default "./output"):
    - resolution_time_model.joblib   (trained sklearn pipeline)
    - feature_columns.json           (the exact feature schema the model expects)
    - metrics.json                   (test-set performance + feature importances)
    - predicted_vs_actual.png        (diagnostic plot)

WHY ONLY SOME ROWS ARE USED FOR TRAINING
-----------------------------------------
The raw dataset has ~8,100 events, but only a subset have a logged
closure/resolution timestamp (most "closed" rows are missing
`closed_datetime`, likely a data-entry gap rather than a true open ticket).
We can only learn duration from rows where we KNOW the true duration, so the
script filters down to rows with a valid, positive, and "reasonable"
(<= 48 hours) resolution time. Everything else is excluded from training,
but the resulting model can still produce a duration prediction for ANY new
incident (planned or unplanned) once enough events have been resolved to
provide good training signal for that combination of features.
"""

import argparse
import glob
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration: which columns the model is allowed to use, and how the
# target is derived. Edit these constants if your dataset schema differs.
# --------------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "event_type",       # planned / unplanned
    "event_cause",      # vehicle_breakdown, pot_holes, accident, ...
    "veh_type",         # bmtc_bus, heavy_vehicle, lcv, ...
    "priority",         # High / Low
    "corridor",          # named road corridor or "Non-corridor"
    "zone",             # policing zone
    "police_station",
    "requires_road_closure",  # True/False, cast to string
    "authenticated",    # yes/no — citizen-reported vs verified
]

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

MAX_REASONABLE_DURATION_MIN = 48 * 60  # cap: 48 hours
MIN_DURATION_MIN = 0.5                 # drop near-zero/negative timestamp errors


def load_and_engineer(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and engineer the target + feature columns."""
    df = pd.read_csv(csv_path, low_memory=False)

    # --- Target: minutes between start and resolution -------------------
    # closed_datetime is the primary resolution timestamp; a few rows only
    # have resolved_datetime populated instead, so we fall back to that.
    df["resolution_dt_raw"] = df["closed_datetime"].fillna(df["resolved_datetime"])

    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
    df["resolution_dt_raw"] = pd.to_datetime(df["resolution_dt_raw"], utc=True, errors="coerce")

    df["duration_minutes"] = (
        df["resolution_dt_raw"] - df["start_datetime"]
    ).dt.total_seconds() / 60.0

    # --- Time-derived features from the moment the incident started -----
    df["hour_of_day"] = df["start_datetime"].dt.hour
    df["day_of_week"] = df["start_datetime"].dt.dayofweek  # 0=Mon
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # requires_road_closure comes in as a real boolean; stringify for OHE
    df["requires_road_closure"] = df["requires_road_closure"].astype(str)

    # --- Keep only rows with a trustworthy target -------------------------
    valid = (
        df["start_datetime"].notna()
        & df["duration_minutes"].notna()
        & (df["duration_minutes"] >= MIN_DURATION_MIN)
        & (df["duration_minutes"] <= MAX_REASONABLE_DURATION_MIN)
    )
    clean = df.loc[valid].copy()

    print(f"Loaded {len(df):,} total events.")
    print(f"Retained {len(clean):,} events with a usable resolution duration "
          f"({MIN_DURATION_MIN} - {MAX_REASONABLE_DURATION_MIN} min).")

    return clean


def build_pipeline() -> Pipeline:
    """Build the preprocessing + model pipeline."""
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ("num", numeric_pipe, NUMERIC_FEATURES),
    ])

    # Duration is heavily right-skewed (most incidents clear in under an
    # hour, a handful take most of a day) so the model is trained on
    # log1p(duration) and predictions are converted back automatically.
    regressor = TransformedTargetRegressor(
        regressor=RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", regressor),
    ])
    return pipeline


def try_alternate_model(X_train, y_train) -> str:
    """
    Compare RandomForest vs GradientBoosting with cross-validation and
    report which one generalizes better. Returns the name of the winner
    (the main pipeline above already uses RandomForest by default; this
    function is informational so you can swap regressors if GBM wins).
    """
    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ("num", numeric_pipe, NUMERIC_FEATURES),
    ])

    print("\nComparing candidate models with 5-fold cross-validation "
          "(scoring = negative MAE on log1p-duration)...")
    best_name, best_score = None, -np.inf
    for name, reg in candidates.items():
        model = TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        scores = cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        mean_score = scores.mean()
        print(f"  {name:18s} CV MAE (log-scale): {-mean_score:.4f}")
        if mean_score > best_score:
            best_name, best_score = name, mean_score

    print(f"Best candidate: {best_name}\n")
    return best_name


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    metrics = {
        "test_set_size": len(y_test),
        "mean_absolute_error_minutes": round(mae, 2),
        "root_mean_squared_error_minutes": round(rmse, 2),
        "r2_score": round(r2, 4),
    }
    print("Test set performance:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics, preds


def get_feature_importances(pipeline: Pipeline, top_n: int = 15) -> list:
    """Map the model's feature_importances_ back to human-readable names."""
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    model = pipeline.named_steps["model"].regressor_  # fitted inner regressor
    importances = model.feature_importances_

    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    top = [{"feature": name, "importance": round(float(score), 4)} for name, score in ranked[:top_n]]

    print(f"\nTop {top_n} most predictive features:")
    for item in top:
        print(f"  {item['feature']:40s} {item['importance']:.4f}")
    return top


def plot_predicted_vs_actual(y_test, preds, outpath: str):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, preds, alpha=0.4, s=18, edgecolor="none")
    max_val = max(y_test.max(), preds.max())
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Perfect prediction")
    plt.xlabel("Actual resolution time (minutes)")
    plt.ylabel("Predicted resolution time (minutes)")
    plt.title("Predicted vs Actual Resolution Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved diagnostic plot to {outpath}")


def build_location_lookup(raw_df: pd.DataFrame, min_count: int = 3) -> dict:
    """
    For each corridor/zone/police_station seen in training, compute the
    centroid (mean lat/lon) of events reported under that label. The
    serving API uses this to map a live GPS click to the nearest known
    category, so the model receives values from the vocabulary it was
    actually trained on instead of arbitrary live place names.
    """
    geo = raw_df.copy()
    geo["latitude"] = pd.to_numeric(geo["latitude"], errors="coerce")
    geo["longitude"] = pd.to_numeric(geo["longitude"], errors="coerce")
    geo = geo.dropna(subset=["latitude", "longitude"])

    lookup = {}
    for col in ["corridor", "zone", "police_station"]:
        sub = geo.dropna(subset=[col])
        grp = sub.groupby(col).agg(lat=("latitude", "mean"), lon=("longitude", "mean"), n=("latitude", "count"))
        grp = grp[grp["n"] >= min_count]
        lookup[col] = {
            idx: {"lat": round(row.lat, 6), "lon": round(row.lon, 6), "n": int(row.n)}
            for idx, row in grp.iterrows()
        }
    return lookup


def build_category_vocab(raw_df: pd.DataFrame) -> dict:
    """
    The exact category strings the model's encoder was fit on, plus a
    data-driven default priority per event_cause (the most common
    observed priority for that cause). Used by the API both to validate
    incoming requests and to populate frontend dropdowns that can't
    drift from what the model actually knows.
    """
    default_priority = (
        raw_df.groupby("event_cause")["priority"]
        .agg(lambda s: s.value_counts().idxmax() if len(s.value_counts()) else "Low")
        .to_dict()
    )
    return {
        "event_cause": sorted(raw_df["event_cause"].dropna().unique().tolist()),
        "veh_type": sorted(raw_df["veh_type"].dropna().unique().tolist()),
        "priority": sorted(raw_df["priority"].dropna().unique().tolist()),
        "event_type": sorted(raw_df["event_type"].dropna().unique().tolist()),
        "default_priority_by_cause": default_priority,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the Astram Guardian duration-prediction model.")
    parser.add_argument(
        "--csv", default=None,
        help="Path to the Astram event CSV. If omitted, the script looks for a "
             "single .csv file in the same folder as this script and uses that."
    )
    parser.add_argument("--outdir", default="./output", help="Directory to write model + reports to.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data held out for testing.")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = glob.glob(os.path.join(script_dir, "*.csv"))
        if len(candidates) == 1:
            csv_path = candidates[0]
            print(f"No --csv given; auto-detected CSV in script folder: {csv_path}")
        elif len(candidates) == 0:
            parser.error(
                "No --csv path given, and no .csv file found next to train_model.py. "
                "Either pass --csv \"path\\to\\file.csv\" or place the CSV in this same folder."
            )
        else:
            parser.error(
                f"No --csv path given, and {len(candidates)} CSV files found in this folder "
                f"(ambiguous: {candidates}). Pass --csv explicitly to pick one."
            )

    os.makedirs(args.outdir, exist_ok=True)

    # 1. Load + engineer features/target
    df = load_and_engineer(csv_path)

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[feature_cols]
    y = df["duration_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

    # 2. Quickly compare candidate regressors (informational)
    try_alternate_model(X_train, y_train)

    # 3. Fit the production pipeline (RandomForest, log-target)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 4. Evaluate
    metrics, preds = evaluate(pipeline, X_test, y_test)
    importances = get_feature_importances(pipeline)

    # 5. Diagnostic plot
    plot_path = os.path.join(args.outdir, "predicted_vs_actual.png")
    plot_predicted_vs_actual(y_test.values, preds, plot_path)

    # 6. Save model + metadata
    model_path = os.path.join(args.outdir, "resolution_time_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nSaved trained model to {model_path}")

    schema_path = os.path.join(args.outdir, "feature_columns.json")
    with open(schema_path, "w") as f:
        json.dump({
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "target": "duration_minutes (predicted, log1p-transformed internally)",
        }, f, indent=2)
    print(f"Saved feature schema to {schema_path}")

    metrics_path = os.path.join(args.outdir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"metrics": metrics, "top_feature_importances": importances}, f, indent=2)
    print(f"Saved metrics report to {metrics_path}")

    # 7. Build and save serving-time assets (location lookup + category vocab).
    # These use the FULL raw dataset (not just the 2,611 rows with a usable
    # duration) since centroids and valid category strings should reflect
    # every event ever logged, not only the resolved ones.
    raw_df = pd.read_csv(csv_path, low_memory=False)

    location_lookup = build_location_lookup(raw_df)
    lookup_path = os.path.join(args.outdir, "location_lookup.json")
    with open(lookup_path, "w") as f:
        json.dump(location_lookup, f, indent=2)
    print(f"Saved location lookup ({sum(len(v) for v in location_lookup.values())} "
          f"known corridor/zone/station centroids) to {lookup_path}")

    category_vocab = build_category_vocab(raw_df)
    vocab_path = os.path.join(args.outdir, "category_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(category_vocab, f, indent=2)
    print(f"Saved category vocabulary to {vocab_path}")


if __name__ == "__main__":
    main()