#!/usr/bin/env python
"""
perform cleaning on the data and save the results in weights & bias
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning", name='data-cleaning')
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("Download input artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("read dataset with pandas")
    df = pd.read_csv(artifact_local_path)

    logger.info("drop outliers and remove null values")
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()
    df=df.dropna()

    logger.info("Convert last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    logger.info("save cleaning data locally")
    df.to_csv("clean_sample.csv", index=False)

    logger.info("create artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")

    logger.info("log artifact output to W&B")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="this step cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name for the download artifact file from W&B",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact file",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="the type for the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="a description for the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price", type=float, help="the minimum price to consider", required=True
    )

    parser.add_argument(
        "--max_price", type=float, help="the maximum price to consider", required=True
    )

    args = parser.parse_args()

    go(args)
