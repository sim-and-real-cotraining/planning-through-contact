import argparse
import os
import time
from datetime import datetime


def run_data_generation_script(
    config_dir, config_name, plans_dir, suppress_output=False
):
    seed = datetime.now().timestamp()
    command = (
        f"python scripts/planar_pushing/run_data_generation.py "
        f"--config-dir={config_dir} "
        f"--config-name {config_name} "
        f"data_collection_config.plans_dir={plans_dir} "
        f"data_collection_config.plan_config.seed={int(seed) % 1000} "
        f"multi_run_config.seed={int(seed) % 1000} "
    )

    if suppress_output:
        command += " > /dev/null 2>&1"

    os.system(command)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render batched trajectories with configurable parameters"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        required=True,
        help="Starting index (inclusive) for the runs",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        required=True,
        help="Ending index (inclusive) for the runs",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        help="Directory containing the configuration files",
    )
    parser.add_argument(
        "--config-name", type=str, required=True, help="Name of the configuration file"
    )
    parser.add_argument(
        "--plans-root",
        type=str,
        required=True,
        help="Root directory for storing the plans",
    )
    parser.add_argument(
        "--suppress-output",
        action="store_true",
        default=False,
        help="Suppress output from the data generation script",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Loop through the indices and execute the command
    for i in range(args.start_index, args.end_index + 1):
        plans_dir = f"{args.plans_root}/run_{i}"
        print(plans_dir)
        start = time.time()
        run_data_generation_script(
            args.config_dir, args.config_name, plans_dir, args.suppress_output
        )
        print(
            f"Finished rendering plans for run {i} in {time.time() - start:.2f} seconds.\n"
        )
