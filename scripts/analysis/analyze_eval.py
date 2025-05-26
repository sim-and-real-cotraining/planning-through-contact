import argparse
import os
import pickle

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from planning_through_contact.geometry.planar.planar_pose import PlanarPose

# Analysis


def compute_average_errors(summary):
    """
    Compute and display the average successful translation and rotation errors
    from the given summary file.

    Args:
        summary (dictionary): summary dictionary.
    """
    # Check if there are successful trials
    if len(summary["successful_trials"]) == 0:
        average_successful_trans_error = float("nan")
        average_successful_rot_error = float("nan")
    else:
        successful_translation_errors = []
        successful_rotation_errors = []

        # Compute errors for each successful trial
        for trial_idx in summary["successful_trials"]:
            successful_translation_errors.append(
                np.linalg.norm(summary["final_error"][trial_idx]["slider_error"][:2])
            )
            successful_rotation_errors.append(
                np.abs(summary["final_error"][trial_idx]["slider_error"][2])
            )

        # Compute averages
        average_successful_trans_error = np.mean(successful_translation_errors)
        average_successful_rot_error = np.mean(successful_rotation_errors)

    return average_successful_trans_error, average_successful_rot_error


def compute_average_speed(analysis_dir):
    pusher_speeds = []
    for i, combined_pkl_file in enumerate(os.listdir(analysis_dir)):
        if not combined_pkl_file.endswith(".pkl"):
            continue
        combined_pkl_path = os.path.join(analysis_dir, combined_pkl_file)
        with open(combined_pkl_path, "rb") as f:
            data = pickle.load(f)
        pusher_data = data.pusher_actual
        t = pusher_data.t
        DT = t[1] - t[0]
        pusher_traj = np.column_stack((pusher_data.x, pusher_data.y))

        # Compute average pusher speed
        pusher_speed = np.linalg.norm(np.diff(pusher_traj, axis=0), axis=1).mean() / DT
        pusher_speeds.append(pusher_speed)

        if i > 500:
            break

    return np.mean(pusher_speeds), np.std(pusher_speeds)


def compute_average_successful_trial_time(summary):
    """
    Compute the average time taken for successful trials.

    Args:
        summary (dictionary): summary dictionary.
    """
    if len(summary["successful_trials"]) == 0:
        return float("nan")

    successful_trial_times = []
    for trial_idx in summary["successful_trials"]:
        trial_time = summary["trial_times"][trial_idx]
        successful_trial_times.append(trial_time)

    average_successful_trial_time = np.mean(successful_trial_times)
    return average_successful_trial_time


def is_success(final_slider_error, trans_tol, rot_tol):
    """
    Check if the final slider error is within the specified tolerances.

    Args:
        final_slider_error (np.ndarray): Final slider error (x, y, theta).
        trans_tol (float): Translation tolerance.
        rot_tol (float): Rotation tolerance.
    """
    position_error = np.linalg.norm(final_slider_error[:2])
    rotation_error = np.abs(final_slider_error[2])
    return position_error <= trans_tol and rotation_error <= rot_tol


def success_rate(summary, trans_tol, rot_tol):
    """
    Compute the success rate from a summary dictionary.

    Args:
        summary (dictionary): summary dictionary
        trans_tol (float): Translation tolerance
        rot_tol (float): Rotation tolerance
    """

    final_errors = summary["final_error"]
    successful_trials = 0
    for final_error in final_errors:
        slider_error = final_error["slider_error"]
        if is_success(slider_error, trans_tol, rot_tol):
            successful_trials += 1
    return successful_trials / len(final_errors)


# Plotting


def plot_success_rate(summary):
    """
    Plot the success rate vs the number of trials from an eval summary.

    Args:
        summary (dictionary): summary dictionary.
    """
    # Compute success rate for each trial
    max_trial = len(summary["trial_result"])
    success_rate = []
    total_success = 0

    for trial, result in enumerate(summary["trial_result"]):
        if result == "success":
            total_success += 1
        success_rate.append(total_success / (trial + 1))

    # Plot success rate vs number of trials
    trials = np.arange(1, max_trial + 1)
    final_success_rate = success_rate[-1] if success_rate else 0

    plt.figure(figsize=(10, 6))
    plt.plot(trials, success_rate)
    plt.axhline(
        y=final_success_rate,
        color="r",
        linestyle="--",
        label=f"Final Success Rate: {final_success_rate:.2f}",
    )
    plt.title("Success Rate vs. Number of Trials")
    plt.xlabel("Number of Trials")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def load_successful_trajectories_from_eval(summary, analysis_directory):
    """
    Load slider and pusher trajectories from a directory of pickle files.

    Parameters:
        analysis_directory (str): Path to the directory containing combined_logs_{i}.pkl files.

    Returns:
        trajectories (list of np.ndarray): List of LxD arrays for each slider trajectory.
    """
    trajectories = []
    pusher_trajectories = []

    # Load summary pickle
    successful_indices = summary["successful_trials"]
    for i in successful_indices:
        file = f"combined_logs_{i}.pkl"
        file_path = os.path.join(analysis_directory, file)

        # Load pickle file
        with open(file_path, "rb") as f:
            combined_logs = pickle.load(f)

        # Extract slider_actual (x, y, theta)
        slider_actual = combined_logs.slider_actual
        slider_traj = np.stack(
            [slider_actual.x, slider_actual.y, slider_actual.theta], axis=1
        )

        # Extract pusher_actual (x, y)
        pusher_actual = combined_logs.pusher_actual
        pusher_traj = np.stack([pusher_actual.x, pusher_actual.y], axis=1)

        trajectories.append(slider_traj)
        pusher_trajectories.append(pusher_traj)

    return trajectories, pusher_trajectories


def load_unsuccessful_trajectories_from_eval(summary, analysis_directory):
    """
    Load slider and pusher trajectories from a directory of pickle files.

    Parameters:
        analysis directory (str): Path to the directory containing combined_logs_{i}.pkl files.

    Returns:
        trajectories (list of np.ndarray): List of LxD arrays for each slider trajectory.
    """
    trajectories = []
    pusher_trajectories = []

    # Load summary pickle
    successful_indices = summary["successful_trials"]
    for i in range(len(summary["trial_times"])):
        if i in successful_indices:
            continue

        file = f"combined_logs_{i}.pkl"
        file_path = os.path.join(analysis_directory, file)

        # Load pickle file
        with open(file_path, "rb") as f:
            combined_logs = pickle.load(f)

        # Extract slider_actual (x, y, theta)
        slider_actual = combined_logs.slider_actual
        slider_traj = np.stack(
            [slider_actual.x, slider_actual.y, slider_actual.theta], axis=1
        )

        # Extract pusher_actual (x, y)
        pusher_actual = combined_logs.pusher_actual
        pusher_traj = np.stack([pusher_actual.x, pusher_actual.y], axis=1)

        trajectories.append(slider_traj)
        pusher_trajectories.append(pusher_traj)

    return trajectories, pusher_trajectories


def subsample_trajectory(trajectory, M):
    """
    Subsample a trajectory to M points after detecting significant movement.
    """
    L = trajectory.shape[0]
    if L < M:
        raise ValueError(
            f"Trajectory length ({L}) is shorter than subsample count ({M})"
        )

    indices = np.linspace(0, L - 1, M).astype(int)
    return trajectory[indices]


def compute_average_absolute_error(trajectories, targets, M):
    """
    Compute average absolute errors for subsampled trajectories.
    """
    abs_errors = []
    for traj, target in zip(trajectories, targets):
        subsampled_traj = subsample_trajectory(traj, M)
        abs_error = np.abs(subsampled_traj - target)
        abs_errors.append(abs_error)

    abs_errors = np.array(abs_errors)
    avg_abs_error = np.mean(abs_errors, axis=0)
    return avg_abs_error


def compute_M(trajectories, M):
    """
    Compute the subsample count M based on the shortest trajectory.
    """
    min_length = min([traj.shape[0] for traj in trajectories])
    return min(M, min_length)


def plot_average_absolute_error(succ_slider_traj, succ_targets, M=100):
    """
    Plot the average absolute error for each dimension as a Dx1 subplot.
    """
    avg_abs_error = compute_average_absolute_error(succ_slider_traj, succ_targets, M)
    D = avg_abs_error.shape[1]
    time_steps = np.linspace(0, 1, avg_abs_error.shape[0])

    fig, axs = plt.subplots(D, 1, figsize=(8, 2 * D), sharex=True)

    if D == 1:
        axs = [axs]  # Ensure axs is iterable when D == 1

    titles = ["X", "Y", "theta"]
    units = ["m", "m", "rad"]
    for i in range(D):
        axs[i].plot(
            time_steps, avg_abs_error[:, i], label=f"Dimension {i+1}", color=f"C{i}"
        )
        axs[i].set_title(f"Average Absolute Error - {titles[i]}")
        axs[i].set_xlabel("Normalized Time")
        axs[i].set_ylabel(f"Error ({units[i]})")
        axs[i].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

    axs[-1].set_xlabel("Time Step (Subsampled)")
    plt.tight_layout()
    plt.show()


def plot_success_rate_vs_tolerance(summary, trans_tol_range, rot_tol_range):
    """
    Plot the success rate vs translation and rotation tolerances.

    Args:
        summary (dictionary): summary dictionary
        trans_tol_range (list): Translation tolerance range
        rot_tol_range (list): Rotation tolerance range
    """

    trans_tols, rot_tols = np.meshgrid(
        np.linspace(trans_tol_range[0], trans_tol_range[1], 20),
        np.linspace(rot_tol_range[0], rot_tol_range[1], 20),
    )
    m, n = trans_tols.shape

    success_rates = np.zeros_like(trans_tols)
    for i in range(m):
        for j in range(n):
            success_rates[i, j] = success_rate(
                summary, trans_tols[i, j], rot_tols[i, j]
            )

    plt.figure(figsize=(8, 6))
    heatmap = plt.pcolormesh(
        trans_tols, rot_tols, success_rates, shading="auto", cmap="viridis"
    )
    plt.colorbar(heatmap, label="Success Rate")
    plt.xlabel("Translation Tolerance")
    plt.ylabel("Rotation Tolerance")
    plt.title("Success Rate vs. Translation and Rotation Tolerances")

    # Annotate each cell with its value
    for i in range(m - 1):
        for j in range(n - 1):
            cell_x = trans_tols[i, j] + (trans_tols[i, j + 1] - trans_tols[i, j]) / 2
            cell_y = rot_tols[i, j] + (rot_tols[i + 1, j] - rot_tols[i, j]) / 2
            plt.text(
                cell_x,
                cell_y,
                f"{success_rates[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=6,
            )

    plt.show()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Compute average successful errors from a summary.pkl file."
    )
    parser.add_argument("--eval-path", type=str, help="Path to the eval directory.")
    args = parser.parse_args()

    analysis_dir = os.path.join(args.eval_path, "analysis")
    summary_path = os.path.join(args.eval_path, "summary.pkl")
    with open(summary_path, "rb") as f:
        summary = pickle.load(f)
    cfg_path = os.path.join(args.eval_path, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    pusher_goal = hydra.utils.instantiate(cfg.pusher_start_pose)
    slider_goal = hydra.utils.instantiate(cfg.slider_goal_pose)

    succ_slider_traj, succ_pusher_traj = load_successful_trajectories_from_eval(
        summary, analysis_dir
    )
    fail_slider_traj, fail_pusher_traj = load_unsuccessful_trajectories_from_eval(
        summary, analysis_dir
    )
    succ_targets = [slider_goal.vector().flatten() for traj in succ_slider_traj]
    fail_targets = [slider_goal.vector().flatten() for traj in fail_slider_traj]

    avg_speed, std_speed = compute_average_speed(analysis_dir)
    avg_trans_err, avg_rot_err = compute_average_errors(summary)
    avg_succ_trial_time = compute_average_successful_trial_time(summary)

    # Display results
    print("\n" + "=" * 50)
    print("=== Average Successful Errors ===")
    print(f"Number of Successful Trials: {len(summary['successful_trials'])}")
    print(
        f"Success Rate: {len(summary['successful_trials']) / len(summary['trial_times'])}"
    )
    print(f"Average Pusher Speed: {100*avg_speed:.2f} cm/s")
    print(f"Standard Deviation of Pusher Speed: {100*std_speed:.2f} cm/s")
    print(f"Average Successful Translation Error: {100*avg_trans_err:.2f} cm")
    print(f"Average Successful Rotation Error: {np.rad2deg(avg_rot_err):.2f}°")
    print(f"Average Successful Trial Time: {avg_succ_trial_time:.2f} s")
    print("=" * 50 + "\n")

    # Plots
    plot_success_rate_vs_tolerance(summary, [0.0, 0.02], [0.0, 0.1])
    plot_success_rate(summary)
    plot_average_absolute_error(succ_slider_traj, succ_targets, 100)


if __name__ == "__main__":
    main()
