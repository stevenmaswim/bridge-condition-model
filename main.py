import sys
import os
import pandas as pd
import numpy as np

from src.data_loader import load_config, load_raw_data, clean_data, save_processed
from src.features import prepare_model_data
from src.model import train_all_targets, save_models, get_feature_importance
from src.predict import predict_all_targets, summarize_by_group, save_predictions


def generate_sample_data(config, n_rows=200):
    """Generate synthetic bridge data for testing the pipeline."""
    rng = np.random.default_rng(42)

    districts = [
        "Austin", "Dallas", "Houston", "San Antonio", "El Paso",
        "Lubbock", "Odessa", "Amarillo", "Waco", "Bryan",
    ]
    climate_zones = ["Humid Subtropical", "Semi-Arid", "Arid", "Subtropical Steppe"]

    data = {
        "structure_length": rng.uniform(20, 2000, n_rows),
        "max_span_length": rng.uniform(10, 500, n_rows),
        "deck_width": rng.uniform(8, 40, n_rows),
        "roadway_width": rng.uniform(6, 36, n_rows),
        "adt": rng.integers(100, 150000, n_rows),
        "adt_year": rng.integers(2015, 2024, n_rows),
        "year_built": rng.integers(1940, 2020, n_rows),
        "year_reconstructed": rng.choice([0, *range(1990, 2024)], n_rows),
        "skew_angle": rng.integers(0, 60, n_rows),
        "num_spans_main": rng.integers(1, 15, n_rows),
        "num_spans_approach": rng.integers(0, 10, n_rows),
        "owner": rng.choice(["State", "County", "City", "Federal"], n_rows),
        "maintenance_resp": rng.choice(["State", "County", "City"], n_rows),
        "structure_kind": rng.choice(["Concrete", "Steel", "Prestressed", "Wood", "Masonry"], n_rows),
        "structure_type": rng.choice(["Slab", "Stringer", "Girder", "Truss", "Arch", "Box Beam"], n_rows),
        "deck_type": rng.choice(["Concrete Cast-in-Place", "Concrete Precast", "Open Grating"], n_rows),
        "wearing_surface": rng.choice(["Monolithic Concrete", "Latex Concrete", "Bituminous"], n_rows),
        "membrane_type": rng.choice(["None", "Built-up", "Preformed Fabric"], n_rows),
        "deck_protection": rng.choice(["None", "Epoxy Coated", "Galvanized"], n_rows),
        "design_load": rng.choice(["HS 20", "HS 25", "HL 93"], n_rows),
        "approach_roadway_width": rng.uniform(6, 40, n_rows),
        "functional_class": rng.choice(["Interstate", "Arterial", "Collector", "Local"], n_rows),
        "facility_carried": rng.choice(["Highway", "Railroad", "Pedestrian"], n_rows),
        "features_intersected": rng.choice(["Highway", "Stream", "Railroad", "Valley"], n_rows),
        "txdot_district": rng.choice(districts, n_rows),
        "climate_zone": rng.choice(climate_zones, n_rows),
    }

    ages = 2026 - data["year_built"]
    base_condition = 9 - (ages / 20.0)
    noise = rng.normal(0, 0.5, n_rows)

    data["deck_cond_rating"] = np.clip(base_condition + noise, 0, 9).round(0)
    data["superstructure_cond_rating"] = np.clip(base_condition + rng.normal(0, 0.6, n_rows), 0, 9).round(0)
    data["substructure_cond_rating"] = np.clip(base_condition + rng.normal(0, 0.7, n_rows), 0, 9).round(0)
    culvert_mask = rng.random(n_rows) > 0.7
    culvert_ratings = np.clip(base_condition + rng.normal(0, 0.5, n_rows), 0, 9).round(0)
    culvert_ratings[~culvert_mask] = np.nan
    data["culvert_cond_rating"] = culvert_ratings

    return pd.DataFrame(data)


def run_pipeline(config_path="config.yaml", dry_run=False):
    print("=" * 60)
    print("Bridge Condition Prediction Model - TxDOT")
    print("=" * 60)

    config = load_config(config_path)

    if dry_run:
        print("\n[DRY RUN] Using synthetic sample data")
        df = generate_sample_data(config)
    else:
        print("\n[1/5] Loading data...")
        df = load_raw_data(config)

    print("\n[2/5] Cleaning data...")
    df = clean_data(df, config)

    print("\n[3/5] Engineering features...")
    df, feature_cols = prepare_model_data(df, config)

    print("\n[4/5] Training models...")
    trained_models, metrics_df = train_all_targets(df, feature_cols, config)

    if not trained_models:
        print("No models were trained. Check your data and config.")
        sys.exit(1)

    save_models(trained_models, config)

    print("\n[5/5] Generating predictions...")
    predictions = predict_all_targets(df, feature_cols, trained_models)
    fi_df = get_feature_importance(trained_models, feature_cols)
    summary = summarize_by_group(predictions, config, trained_models)
    save_predictions(predictions, summary, metrics_df, fi_df, config)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Models trained: {len(trained_models)}")
    print(f"  Predictions: {len(predictions)} rows")
    if not metrics_df.empty:
        avg_r2 = metrics_df["r2"].mean()
        avg_mae = metrics_df["mae"].mean()
        print(f"  Avg R2: {avg_r2:.3f}, Avg MAE: {avg_mae:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    config_path = "config.yaml"
    for arg in sys.argv[1:]:
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
    run_pipeline(config_path=config_path, dry_run=dry_run)
