import argparse

from src.forecast import predict_future_all


def main():
    parser = argparse.ArgumentParser(
        description="Estimate a bridge's condition ratings in a future year, given its NBI/bridge code."
    )
    parser.add_argument("--nbi", required=True, help="NBI/bridge number to look up (e.g. 010600013603019)")
    parser.add_argument("--year", required=True, type=int, help="Future year to project condition to")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    predict_future_all(args.nbi, args.year, config_path=args.config)


if __name__ == "__main__":
    main()
