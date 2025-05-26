import argparse
import copy
import csv
import os
import pickle
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from queue import Queue

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.stats import beta

# Common arguments
CONFIG_DIR = "config/sim_config/sim_sim"
CONFIG_NAME = "gamepad_teleop.yaml"
BASE_COMMAND = [
    "python",
    "scripts/planar_pushing/run_sim_sim_eval.py",
    f"--config-dir={CONFIG_DIR}",
]

# ---------------------------------------------------------
# Example Usage:
# python launch_evals.py --csv-path /path/to/jobs.csv --max-concurrent-jobs 8
#
# CSV file format:
# checkpoint_path,run_dir,config_name (optional)
# /path/to/checkpoint1.ckpt, data/test1, custom_config.yaml
# /path/to/checkpoint2.ckpt, data/test2
# ---------------------------------------------------------


@dataclass
class JobConfig:
    checkpoint_path: str
    run_dir: str
    config_name: str
    num_trials: int = -1
    seed: int = 0
    continue_flag: bool = False

    def __str__(self):
        return f"checkpoint_path={self.checkpoint_path}, run_dir={self.run_dir}, config_name={self.config_name}, num_trials={self.num_trials}, seed={self.seed}, continue_flag={self.continue_flag}"

    def __repr__(self):
        return str(self)


@dataclass
class JobResult:
    num_successful_trials: int
    num_trials: int
    job_config: JobConfig = None

    def __post_init__(self):
        self.success_rate = self.num_successful_trials / self.num_trials

    def __str__(self):
        return f"num_successful_trials={self.num_successful_trials}, num_trials={self.num_trials}, success_rate={self.success_rate}"

    def __repr__(self):
        return str(self)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch multiple Hydra simulation commands concurrently."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to the CSV file containing checkpoint paths, run directories, and optional config names.",
    )
    parser.add_argument(
        "--max-concurrent-jobs",
        type=int,
        default=8,
        help="Maximum number of concurrent jobs (default: 8).",
    )
    parser.add_argument(
        "--num-trials-per-round",
        type=int,
        nargs="+",
        default=[50, 50, 100],
        help="List of number of trials per round (default: [50, 50, 100]).",
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=0.05,
        help="Threshold for dropping checkpoints (default: 0.05).",
    )
    return parser.parse_args()


def get_checkpoint_root_and_name(checkpoint_path):
    """Get the root directory and name of the checkpoint."""
    checkpoint_root = checkpoint_path.split("/checkpoints")[0]
    checkpoint_name = checkpoint_path.split("/")[-1]
    return checkpoint_root, checkpoint_name


def load_jobs_from_csv(csv_file):
    """Load checkpoint groups, where each group consists of one or more checkpoints."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.")

    job_groups = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            checkpoint_path = row.get("checkpoint_path", "").strip()
            run_dir = row.get("run_dir", "").strip()
            config_name = row.get("config_name", CONFIG_NAME).strip()

            # If evaluating a single checkpoint, create a single-element group
            if checkpoint_path.endswith(".ckpt"):
                assert os.path.exists(
                    checkpoint_path
                ), f"Checkpoint file '{checkpoint_path}' does not exist."
                checkpoint_root, checkpoint_file = get_checkpoint_root_and_name(
                    checkpoint_path
                )

                job_config = JobConfig(
                    checkpoint_path=checkpoint_path,
                    run_dir=f"{run_dir}/{checkpoint_file}",
                    config_name=config_name,
                    seed=0,
                    continue_flag=False,
                )
                # assert checkpoint_root not in job_groups
                job_groups[checkpoint_root] = {checkpoint_file: job_config}

            # If evaluating all checkpoints from a training run, create a group
            else:
                checkpoint_group = {}
                checkpoints_dir = os.path.join(checkpoint_path, "checkpoints")
                for checkpoint_file in os.listdir(checkpoints_dir):
                    if checkpoint_file.endswith(".ckpt"):
                        full_checkpoint_path = os.path.join(
                            checkpoints_dir, checkpoint_file
                        )
                        (
                            checkpoints_root,
                            checkpoint_file,
                        ) = get_checkpoint_root_and_name(full_checkpoint_path)
                        job_config = JobConfig(
                            checkpoint_path=full_checkpoint_path,
                            run_dir=os.path.join(run_dir, checkpoint_file),
                            config_name=config_name,
                            seed=0,
                            continue_flag=False,
                        )
                        checkpoint_group[checkpoint_file] = job_config
                # assert checkpoints_root not in job_groups
                job_groups[checkpoints_root] = checkpoint_group

    return job_groups


def run_simulation(job_config, job_number, total_jobs, round_number, total_rounds):
    """Run a single simulation with specified checkpoint, run directory, and config name."""
    checkpoint_path = job_config.checkpoint_path
    run_dir = job_config.run_dir
    config_name = job_config.config_name
    num_trials = job_config.num_trials
    seed = job_config.seed
    continue_flag = job_config.continue_flag
    assert num_trials > 0, "num_trials must be greater than 0"

    command = BASE_COMMAND + [
        f"--config-name={config_name}",
        f'diffusion_policy_config.checkpoint="{checkpoint_path}"',
        f'hydra.run.dir="{run_dir}"',
        f"multi_run_config.seed={seed}",
        f"multi_run_config.num_runs={num_trials}",
        f"++continue_eval={continue_flag}",
    ]
    command_str = " ".join(command)

    print("\n" + "=" * 50)
    print(
        f"=== Round ({round_number} of {total_rounds}): JOB {job_number} of {total_jobs} ==="
    )
    print(f"=== JOB START: {run_dir} ===")
    print(command_str)
    print("=" * 50 + "\n")

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"\n✅ Completed: {run_dir}")

        # Compute success rate
        summary_file = os.path.join(run_dir, "summary.pkl")
        with open(summary_file, "rb") as f:
            summary = pickle.load(f)
        num_successful_trials = len(summary["successful_trials"])
        num_trials = len(summary["trial_times"])
        success_rate = num_successful_trials / num_trials
    else:
        print(f"\n❌ Failed: {run_dir}\nError: {result.stderr}")
        success_rate = None

    print("\n" + "=" * 50)
    print(f"=== JOB END: {run_dir} ===")
    if success_rate is not None:
        print(
            f"Success Rate: {success_rate:.6f} ({num_successful_trials}/{num_trials})"
        )
    else:
        print(f"Success Rate: None")
    print("=" * 50 + "\n")

    if success_rate is None:
        return None
    else:
        return JobResult(
            num_successful_trials=len(summary["successful_trials"]),
            num_trials=len(summary["trial_times"]),
            job_config=job_config,
        )


def validate_job_groups(job_groups):
    if not job_groups:
        print("No valid jobs found in the CSV file. Please check the file.")
        return False

    # Sure there are no duplicate logging directories in the jobs list
    logging_dirs = []
    for _, group in job_groups.items():
        for _, job in group.items():
            logging_dirs.append(job.run_dir)
    if len(logging_dirs) != len(set(logging_dirs)):
        print("Duplicate logging directories found in the jobs list.")
        return False

    # Double check if output directories already exist
    for _, group in job_groups.items():
        for _, job in group.items():
            output_dir = job.run_dir
            if os.path.exists(output_dir):
                print(
                    f"Output directory '{output_dir}' already exists. Running this job will delete the existing contents."
                )
                resp = input("Run job anyways? [y/n]: ")
                if resp.lower() == "y":
                    print("Deleting output directory...\n")
                    shutil.rmtree(output_dir)
                else:
                    print("Exiting...")
                    return False

    return True


def print_diagnostic_info(job_groups, max_concurrent_jobs, num_trials, drop_threshold):
    num_jobs = sum([len(group) for group in job_groups.values()])

    print("\nDiagnostic Information:")
    print("=======================")
    print(
        f"Evaluating {len(job_groups)} training runs, consistenting of {num_jobs} checkpoints"
    )
    print(f"Running with {max_concurrent_jobs} jobs")
    print(f"Checkpoints will be compared at {num_trials} trials.")
    print(
        f"During each comparison, if the probability that a checkpoint "
        f"is better than the current best checkpoint is less than {drop_threshold}, "
        f"the checkpoint will be dropped."
    )
    print(f"The best checkpoints will be evaluated for {sum(num_trials)} trials.")
    print("\nTraining run details:")

    for training_dir, group in job_groups.items():
        print("------------------------------")
        print(f"Training Run: {training_dir}")
        print("Checkpoints:")
        for i, job_item in enumerate(group.items()):
            _, job = job_item
            print(f"  {i+1}. {os.path.basename(job.checkpoint_path)}")
        print(f"Eval directory: {job.run_dir}")
        print(f"Config Name: {job.config_name}")
        print()
    print()


def prob_p1_greater_p2(n1, N1, n2, N2):
    """
    Computes P(p1 > p2) where:
    - n1, N1: Successes and trials for p1
    - n2, N2: Successes and trials for p2

    Returns:
    - Probability that p1 > p2
    """

    # Numerical integration
    alpha1, beta1 = n1 + 1, N1 - n1 + 1
    alpha2, beta2 = n2 + 1, N2 - n2 + 1

    def cdf_p1(x):
        return beta.cdf(x, alpha1, beta1)

    def pdf_p2(x):
        return beta.pdf(x, alpha2, beta2)

    # p(p1 > p2) = int_0^1 cdf_p1(x) * pdf_p2(x) dx
    integral, _ = quad(lambda x: (1 - cdf_p1(x)) * pdf_p2(x), 0, 1)
    return integral


def determine_new_jobs_to_run(success_rates, drop_threshold):
    jobs_to_run = []
    for group, completed_jobs in success_rates.items():
        # Check for Nones
        has_none = False
        for checkpoint, result in completed_jobs.items():
            if result is None:
                has_none = True
                break
        if has_none:
            print(f"Skipping group {group} due to None values.")
            continue

        # Check for only one job
        if len(completed_jobs) == 1:
            result = list(completed_jobs.values())[0]
            jobs_to_run.append(result.job_config)
            continue

        # Find the best job
        best_job_success_rate = 0
        best_job_num_successful_trials = 0
        best_job_num_trials = 0
        for checkpoint, result in completed_jobs.items():
            if result.success_rate > best_job_success_rate:
                best_job_success_rate = result.success_rate
                best_job_num_successful_trials = result.num_successful_trials
                best_job_num_trials = result.num_trials

        # Compare all other jobs to the best job
        for checkpoint, result in completed_jobs.items():
            if result.success_rate == best_job_success_rate:
                jobs_to_run.append(result.job_config)
                continue

            # Compare the job to the best job
            prob = prob_p1_greater_p2(
                result.num_successful_trials,
                result.num_trials,
                best_job_num_successful_trials,
                best_job_num_trials,
            )
            if prob < drop_threshold:
                print(
                    f"Dropping {checkpoint} with success rate {result.success_rate} from group {group}. (p(ckpt > best) = {prob:.6f})"
                )
            else:
                jobs_to_run.append(result.job_config)

    return jobs_to_run


def print_best_checkpoints(success_rates, job_groups):
    print("Final Results (Best Checkpoints):")
    print("=======================")
    for group in job_groups.keys():
        if len(success_rates[group]) == 0:
            print(f"{group}:\n  error (please rerun)\n")
            continue

        # if group has None result, a job has failed along the way
        has_none = False
        for checkpoint, result in success_rates[group].items():
            if result is None:
                has_none = True
                break
        if has_none:
            print(f"{group}:\n  error (please rerun)\n")
            continue

        # find the best job
        best_result = JobResult(0, 1)  # success rate of 0
        for checkpoint, result in success_rates[group].items():
            if result.success_rate > best_result.success_rate:
                best_result = result
            elif result.success_rate == best_result.success_rate:
                _, checkpoint_file = get_checkpoint_root_and_name(
                    result.job_config.checkpoint_path
                )
        print(f"{group}:")
        for checkpoint, result in success_rates[group].items():
            if result.success_rate == best_result.success_rate:
                _, checkpoint_file = get_checkpoint_root_and_name(
                    result.job_config.checkpoint_path
                )
                print(
                    f"  {checkpoint_file}: {result.success_rate:.6f} ({result.num_successful_trials}/{result.num_trials})"
                )
        print()


def main():
    args = parse_arguments()
    csv_file = args.csv_path
    max_concurrent_jobs = args.max_concurrent_jobs
    num_trials = args.num_trials_per_round  # default: [50, 50, 100]
    drop_threshold = args.drop_threshold  # default: 0.05

    job_groups = load_jobs_from_csv(csv_file)
    if not validate_job_groups(job_groups):
        return
    print_diagnostic_info(job_groups, max_concurrent_jobs, num_trials, drop_threshold)

    jobs_to_run = [
        job_config for group in job_groups.values() for job_config in group.values()
    ]
    for i, trial in enumerate(num_trials):
        round_number = i + 1
        success_rates = {group: {} for group in job_groups.keys()}
        num_jobs_per_group = {group: 0 for group in job_groups.keys()}
        for job in jobs_to_run:
            group, checkpoint_file = get_checkpoint_root_and_name(job.checkpoint_path)
            num_jobs_per_group[group] += 1

        print(f"\nRound {i+1} of {len(num_trials)}: Running {len(jobs_to_run)} jobs")
        for group, num_jobs in num_jobs_per_group.items():
            print(f"  {group}: {num_jobs} job(s)")

        # Overwrite job configs
        for job_config in jobs_to_run:
            job_config.num_trials = trial
            job_config.seed = i
            job_config.continue_flag = i != 0

        # Run jobs
        with ThreadPoolExecutor(max_workers=max_concurrent_jobs) as executor:
            # Submit jobs
            futures = {}
            for job_number, job in enumerate(jobs_to_run):
                future = executor.submit(
                    run_simulation,
                    job,
                    job_number + 1,
                    len(jobs_to_run),
                    round_number,
                    len(num_trials),
                )
                futures[future] = job
                time.sleep(1)  # prevent syncing issues with arbitrary_shape.sdf

            # Wait for jobs to finish
            for future in as_completed(futures):
                job_result = future.result()
                if job_result is not None:
                    group, checkpoint_file = get_checkpoint_root_and_name(
                        job_result.job_config.checkpoint_path
                    )
                else:
                    job_config = futures[future]
                    group, checkpoint_file = get_checkpoint_root_and_name(
                        job_config.checkpoint_path
                    )
                assert (
                    checkpoint_file not in success_rates[group]
                ), f"Duplicate checkpoint {checkpoint_file} in group {group}"
                success_rates[group][checkpoint_file] = job_result

        # Determine new jobs to run
        if i != len(num_trials) - 1:
            jobs_to_run = determine_new_jobs_to_run(success_rates, drop_threshold)

    print("\n✅ All jobs finished.\n")
    print_best_checkpoints(success_rates, job_groups)


def create_probability_grid(N1, N2, threshold=0.05):
    grid = np.zeros((N1 + 1, N2 + 1), dtype=bool)

    for n1 in range(N1 + 1):
        for n2 in range(N2 + 1):
            prob = prob_p1_greater_p2(n1, N1, n2, N2)
            grid[n1, n2] = prob < threshold

    return grid


def visualize_grid(grid, N1, N2, threshold):
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap="coolwarm", origin="lower", extent=(0, N2, 0, N1))
    plt.colorbar(label=f"Probability < {threshold}")
    plt.title(f"Grid of P(p1 > p2) < {threshold}")
    plt.xlabel("n2 (better policy)")
    plt.ylabel("n1 (worse policyt)")
    plt.xticks(np.arange(0, N2 + 1, step=max(1, N2 // 10)))
    plt.yticks(np.arange(0, N1 + 1, step=max(1, N1 // 10)))

    # Annotate the grid with 'T' and 'F'
    for i in range(N1 + 1):
        for j in range(N2 + 1):
            text = "T" if grid[i, j] else "F"
            plt.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig("probability_grid.png")


# if __name__ == "__main__":
#     # Parameters
#     N1 = 50
#     N2 = 50
#     threshold = 0.03

#     # Create and visualize the grid
#     grid = create_probability_grid(N1, N2, threshold)
#     visualize_grid(grid, N1, N2, threshold)

if __name__ == "__main__":
    main()
