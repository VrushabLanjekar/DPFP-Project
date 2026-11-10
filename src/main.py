"""API-only entry point for one daily pipeline run."""
import argparse
from src.pipeline.runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the Pune Open-Meteo air-quality pipeline for one date.")
    parser.add_argument("--run-date", required=True, help="Date to fetch, e.g. 2026-10-01")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration YAML path")
    args = parser.parse_args()
    result = run_pipeline(args.run_date, config_path=args.config)
    print("\n========== PIPELINE RESULT ==========")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
