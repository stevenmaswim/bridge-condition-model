import argparse
import os

from src.data_loader import load_config, load_raw_data, rename_raw_columns, clean_data
from src.deterioration import build_watchlist, load_deterioration_model


def main():
    ap = argparse.ArgumentParser(
        description="Build a budget watch-list of bridges forecast to reach poor condition, ranked "
                    "worst-first. Uses the conservative (plan-for-this) forecast.")
    ap.add_argument("--district", default=None, help="District code to filter to (default: all districts)")
    ap.add_argument("--target", default="deck_cond_rating", help="Which condition rating to screen")
    ap.add_argument("--horizon", type=int, default=None, help="Years ahead (default: from config)")
    ap.add_argument("--all-system", action="store_true",
                    help="Include off-system bridges too (default: on-system / state-maintained only)")
    ap.add_argument("--out", default=None, help="Output CSV path (default: data/outputs/...)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    wcfg = (cfg.get("deterioration", {}) or {}).get("watchlist", {})
    horizon = args.horizon or wcfg.get("horizon_years", 10)
    on_system_only = wcfg.get("on_system_only", True) and not args.all_system

    df = clean_data(rename_raw_columns(load_raw_data(cfg), cfg), cfg)
    models = {}
    for t in cfg["targets"]:
        try:
            models[t] = load_deterioration_model(t, cfg)
        except FileNotFoundError:
            pass
    if args.target not in models:
        raise SystemExit(f"No deterioration model found for {args.target}. Train the pipeline first.")

    wl = build_watchlist(
        df, cfg, models, target=args.target, horizon=horizon, district=args.district,
        current_min=wcfg.get("current_min", 5), current_max=wcfg.get("current_max", 7),
        poor_threshold=wcfg.get("poor_threshold", 5.0), on_system_only=on_system_only)

    scope = args.district or "ALL districts"
    sysscope = "on-system only" if on_system_only else "all systems"
    print(f"\nWatch-list: {len(wl)} bridges  (target={args.target}, horizon={horizon}y, "
          f"district={scope}, {sysscope})")
    print("Bridges now rated 5-7 whose conservative forecast reaches the poor threshold, worst first:\n")
    print(wl.head(25).to_string(index=False))

    out = args.out or os.path.join(cfg["data"]["output_dir"], f"watchlist_{args.target}_{scope}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    wl.to_csv(out, index=False)
    print(f"\nSaved {len(wl)} bridges to {out}")


if __name__ == "__main__":
    main()
