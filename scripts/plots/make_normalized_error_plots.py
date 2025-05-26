import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import zarr


def load_trajectories_from_eval(directory, movement_threshold=0.01):
    """
    Load slider and pusher trajectories from a directory of pickle files.

    Parameters:
        directory (str): Path to the directory containing combined_logs_{i}.pkl files.
        movement_threshold (float): Threshold for detecting significant movement in pusher trajectory.

    Returns:
        trajectories (list of np.ndarray): List of LxD arrays for each slider trajectory.
    """
    trajectories = []
    pusher_trajectories = []

    # Load summary pickle
    summary_file = os.path.join(directory, "summary.pkl")
    with open(summary_file, "rb") as f:
        summary = pickle.load(f)
    successful_indices = summary["successful_trials"]
    analysis_directory = os.path.join(directory, "analysis")
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

        # Determine the starting index for significant movement
        # start_idx = detect_movement_start(pusher_traj, threshold=movement_threshold)
        start_idx = 0

        # Trim both trajectories based on the detected start index
        if start_idx >= 0:
            slider_traj = slider_traj[start_idx:]
            pusher_traj = pusher_traj[start_idx:]

            trajectories.append(slider_traj)
            pusher_trajectories.append(pusher_traj)

    return trajectories


def load_trajectory_from_zarr(zarr_path, movement_threshold=0.001):
    root = zarr.open(zarr_path)
    pusher_state = root["data/state"]
    slider_state = root["data/slider_state"]
    episode_ends = root["meta/episode_ends"]

    trajectories = []
    episode_start = 0
    for i, episode_end in enumerate(episode_ends):
        pusher_traj = pusher_state[episode_start:episode_end]
        slider_traj = slider_state[episode_start:episode_end]

        # Determine the starting index for significant movement
        start_idx = detect_movement_start(pusher_traj, threshold=movement_threshold)
        if start_idx >= 0:
            slider_traj = slider_traj[start_idx:]
            trajectories.append(slider_traj)

        episode_start = episode_end

    return trajectories


def detect_movement_start(pusher_traj, threshold=0.01):
    """
    Detect the index where significant movement starts in the pusher trajectory.
    """
    diffs = np.linalg.norm(np.diff(pusher_traj, axis=0), axis=1)
    start_idx = 0
    for diff in diffs:
        if diff > threshold:
            break
        start_idx += 1

    if start_idx == len(diffs):
        return -1
    return start_idx


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


def plot_average_absolute_error(avg_abs_error):
    """
    Plot the average absolute error for each dimension as a Dx1 subplot.
    """
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


# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str)
    parser.add_argument("--movement_threshold", type=float, default=0.01)
    parser.add_argument("--M", type=int, default=100)
    args = parser.parse_args()

    directory = args.directory
    movement_threshold = args.movement_threshold
    M = args.M

    if ".zarr" in directory:
        trajectories = load_trajectory_from_zarr(directory, movement_threshold)
    else:
        trajectories = load_trajectories_from_eval(directory, movement_threshold)
    M = compute_M(trajectories, M)

    # Load trajectories
    print(f"Loaded {len(trajectories)} trajectories.")

    # Create target points (for example purposes, using the mean endpoint of each trajectory)
    targets = [np.array([0.587, -0.0355, 0.0]) for traj in trajectories]

    # Compute average absolute error
    avg_abs_error = compute_average_absolute_error(trajectories, targets, M)
    print("Computed average absolute error.")

    # Plot results
    plot_average_absolute_error(avg_abs_error)
