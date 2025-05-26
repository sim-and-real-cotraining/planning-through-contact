import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import zarr


def load_trajectories_from_eval(directory, movement_threshold=0.01):
    """
    Load slider and pusher trajectories from a directory of pickle files.
    """
    trajectories = []

    summary_file = os.path.join(directory, "summary.pkl")
    with open(summary_file, "rb") as f:
        summary = pickle.load(f)
    successful_indices = summary["successful_trials"]
    analysis_directory = os.path.join(directory, "analysis")
    for i in successful_indices:
        file_path = os.path.join(analysis_directory, f"combined_logs_{i}.pkl")
        with open(file_path, "rb") as f:
            combined_logs = pickle.load(f)
        slider_actual = combined_logs.slider_actual
        slider_traj = np.stack(
            [slider_actual.x, slider_actual.y, slider_actual.theta], axis=1
        )
        trajectories.append(slider_traj)
    return trajectories


def load_trajectory_from_zarr(zarr_path, movement_threshold=0.001):
    root = zarr.open(zarr_path)
    slider_state = root["data/slider_state"]
    episode_ends = root["meta/episode_ends"]
    trajectories = []
    episode_start = 0
    for episode_end in episode_ends:
        slider_traj = slider_state[episode_start:episode_end]
        trajectories.append(slider_traj)
        episode_start = episode_end
    return trajectories


def subsample_trajectory(trajectory, M):
    L = trajectory.shape[0]
    indices = np.linspace(0, L - 1, M).astype(int)
    return trajectory[indices]


def compute_average_absolute_error(trajectories, targets, M):
    abs_errors = []
    for traj, target in zip(trajectories, targets):
        subsampled_traj = subsample_trajectory(traj, M)
        abs_error = np.abs(subsampled_traj - target)
        abs_errors.append(abs_error)
    return np.mean(np.array(abs_errors), axis=0)


def compute_M(trajectories, M):
    return min(M, min([traj.shape[0] for traj in trajectories]))


def plot_comparison(avg_abs_errors, directory_labels, save_path=None):
    """
    Plot normalized average absolute errors for multiple datasets in a 1x3 grid,
    with the legend placed below the subplots.

    Args:
        avg_abs_errors (list of np.ndarray): List of error arrays for each dataset.
        directory_labels (list of str): Labels corresponding to each dataset.
    """
    D = avg_abs_errors[0].shape[1]  # Assuming 3 (X, Y, Theta)
    time_steps = np.linspace(0, 1, avg_abs_errors[0].shape[0])

    fig, axs = plt.subplots(1, D, figsize=(12, 4), sharex=True)  # 1x3 layout

    titles = [
        r"Normalized X Error",
        r"Normalized Y Error",
        r"Normalized $\theta$ Error",
    ]
    legend_labels = [
        r"$\mathcal{D}_S$",
        r"$\mathcal{D}_T$",
        r"Closed Loop Policy Rollout in $Target$ Sim",
    ]
    colors = ["#4C72B0", "#FF6961", "#FFC20A"]

    def normalize_data(data):
        """Normalize data between 0 and 1 for each dimension."""
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val)

    # Normalize all errors
    normalized_errors = [normalize_data(error) for error in avg_abs_errors]

    for i in range(D):
        for idx, error in enumerate(normalized_errors):
            axs[i].plot(
                time_steps,
                error[:, i],
                label=legend_labels[idx],
                color=colors[idx],
                linewidth=3,
            )

        axs[i].set_title(titles[i], fontsize=18)
        axs[i].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        axs[i].set_xticks([0, 1])  # Keep only 0 and 1 on x-axis
        axs[i].set_yticks([0])  # Add a y-tick at 0
        axs[i].tick_params(axis="both", labelsize=16)  # Increase tick label size
        axs[i].set_xlabel("Normalized Time", fontsize=16)

        # Remove top and right spines
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)

    # Add a figure-wide legend below all subplots
    fig.legend(
        legend_labels,
        loc="lower center",
        fontsize=16,
        ncol=len(legend_labels),
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space at the bottom for legend
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--directories", type=str, nargs='+', required=True, help="List of directories to analyze.")
    parser.add_argument("--movement_threshold", type=float, default=0.01)
    parser.add_argument("--M", type=int, default=100)
    args = parser.parse_args()

    # directories = args.directories
    directories = [
        "/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/sim2sim_cotrain_data.zarr",
        "/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/gamepad_teleop_visual_gap.zarr",  # for some reason, the carbon one is empty
        # "eval/sim_sim/sim2sim_cotrain/cotrain_50_2000_1:3/epoch=0020-val_loss_0=0.0201-val_ddim_mse_0=0.0002.ckpt/",
        "eval/sim_sim/sim2sim_cotrain/cotrain_50_2000/latest.ckpt/",
    ]
    movement_threshold = args.movement_threshold
    M = args.M

    avg_abs_errors = []
    directory_labels = []

    for directory in directories:
        if ".zarr" in directory:
            trajectories = load_trajectory_from_zarr(directory, movement_threshold)
        else:
            trajectories = load_trajectories_from_eval(directory, movement_threshold)
        M = compute_M(trajectories, M)
        if "sim2sim_cotrain_data" in directory:
            targets = [np.array([0.587, -0.0655, 0.0]) for _ in trajectories]
        else:
            targets = [np.array([0.587, -0.0355, 0.0]) for _ in trajectories]
        avg_abs_error = compute_average_absolute_error(trajectories, targets, M)
        avg_abs_errors.append(avg_abs_error)
        directory_labels.append(os.path.basename(directory))

    plot_comparison(avg_abs_errors, directory_labels, "error_plots.pdf")
