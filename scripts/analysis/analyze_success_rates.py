import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np


def plot_alpha_sweep(
    alphas, data, num_real, num_sim, real_only_baseline=None, legend=None, title=None
):
    """
    Plot success rate against alpha.

    Args:
        alphas (list or list of lists): List of alpha values or list of lists of alpha values.
        data (np.ndarray): Array of success rates for multiple data streams (rows).
        num_real (int): Number of real data points.
        num_sim (list): Number of simulated data points per stream.
        real_only_baseline (list, optional): Baseline data for alpha=1.
        legend (list, optional): Legend labels for the data streams.
        title (str, optional): Main title for the plot.
    """
    num_streams = data.shape[0]

    # Ensure alphas is a list of lists, one for each data stream
    if not isinstance(alphas[0], list):
        alphas = [alphas] * num_streams

    # Replace None in alphas with num_real / (num_real + num_sim) for each data stream
    for i in range(num_streams):
        alphas[i] = [
            num_real / (num_real + num_sim[i]) if alpha is None else alpha
            for alpha in alphas[i]
        ]

    # Handle real_only_baseline
    if real_only_baseline is not None:
        for i in range(num_streams):
            alphas[i].append(1)
    data = np.hstack(
        (data, np.array([real_only_baseline] * num_streams).reshape(-1, 1))
    )

    # Sort alphas and data together for each stream
    sorted_data = []
    sorted_alphas = []
    for i in range(num_streams):
        sorted_indices = np.argsort(alphas[i])
        sorted_alphas.append(np.array(alphas[i])[sorted_indices])
        sorted_data.append(data[i][sorted_indices])

    # Plot
    plt.figure(figsize=(8, 6))
    if real_only_baseline is not None:
        plt.axhline(y=np.mean(real_only_baseline), color="red", linestyle="--")

    for i in range(num_streams):
        plt.plot(
            sorted_alphas[i],
            sorted_data[i],
            marker="o",
            label=legend[i] if legend else None,
        )

    plt.xlabel(r"$\alpha$")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.title(title or r"Success Rate vs. $\alpha$")
    if legend:
        plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def plot_alpha_sweep_subplot(
    ax,
    alphas,
    data,
    num_real,
    num_sim,
    real_only_baseline=None,
    legend=None,
    title=None,
):
    """
    Plot success rate against alpha.

    Args:
        ax (matplotlib.axes.Axes): Axes object to plot on.
        alphas (list or list of lists): List of alpha values or list of lists of alpha values.
        data (np.ndarray): Array of success rates for multiple data streams (rows).
        num_real (int): Number of real data points.
        num_sim (list): Number of simulated data points per stream.
        real_only_baseline (list, optional): Baseline data for alpha=1.
        legend (list, optional): Legend labels for the data streams.
        title (str, optional): Main title for the plot.
    """
    num_streams = data.shape[0]

    # Ensure alphas is a list of lists, one for each data stream
    if not isinstance(alphas[0], list):
        alphas = [alphas] * num_streams

    # Replace None in alphas with num_real / (num_real + num_sim) for each data stream
    for i in range(num_streams):
        alphas[i] = [
            num_real / (num_real + num_sim[i]) if alpha is None else alpha
            for alpha in alphas[i]
        ]

    # Handle real_only_baseline
    if real_only_baseline is not None:
        for i in range(num_streams):
            alphas[i].append(1)
    data = np.hstack(
        (data, np.array([real_only_baseline] * num_streams).reshape(-1, 1))
    )

    # Sort alphas and data together for each stream
    sorted_data = []
    sorted_alphas = []
    for i in range(num_streams):
        sorted_indices = np.argsort(alphas[i])
        sorted_alphas.append(np.array(alphas[i])[sorted_indices])
        sorted_data.append(data[i][sorted_indices])

    # Plot
    if real_only_baseline is not None:
        ax.axhline(y=np.mean(real_only_baseline), color="red", linestyle="--")

    for i in range(num_streams):
        ax.plot(
            sorted_alphas[i],
            sorted_data[i],
            marker="o",
            label=legend[i] if legend else None,
        )

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.5, 1)
    ax.set_title(title or r"Success Rate vs. $\alpha$")
    if legend:
        ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)


def compute_success_rate(summary, trans_tol, rot_tol):
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


def extract_ratio(filename):
    """
    Extract the ratio from a filename and compute the value.

    Args:
        filename (str): The input filename.

    Returns:
        float or None: The computed ratio (e.g., 1/(1+3)) or None if ':' is not found.
    """
    match = re.search(r"(\d+):(\d+)", filename)
    if match:
        numerator, denominator = map(int, match.groups())
        return numerator / (numerator + denominator)
    return None


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


def compute_results(eval_dir, data_mixtures, trans_tol, rot_tol):
    results = {}
    for data_mixture in data_mixtures:
        # loop through directories in eval_dir
        mixture_results = {}
        for experiment in os.listdir(eval_dir):
            if data_mixture not in experiment:
                continue
            # Loop through all checkpoints
            best_success_rate = -1
            for checkpoint in os.listdir(os.path.join(eval_dir, experiment)):
                # Load summary file
                summary_file = os.path.join(
                    eval_dir, experiment, checkpoint, "summary.pkl"
                )
                if not os.path.exists(summary_file):
                    continue
                with open(summary_file, "rb") as f:
                    summary = pickle.load(f)
                success_rate = compute_success_rate(summary, trans_tol, rot_tol)
                best_success_rate = max(best_success_rate, success_rate)
            ratio = extract_ratio(experiment)
            if best_success_rate == -1:
                best_success_rate = float("nan")
            # if ratio is None:
            #     ratio = float("nan")
            mixture_results[ratio] = best_success_rate
        results[data_mixture] = mixture_results
    return results


def compute_baseline_results(baseline_dir, trans_tol, rot_tol):
    best_success_rate = -1
    for checkpoint in os.listdir(baseline_dir):
        summary_file = os.path.join(baseline_dir, checkpoint, "summary.pkl")
        if not os.path.exists(summary_file):
            continue
        with open(summary_file, "rb") as f:
            summary = pickle.load(f)
        success_rate = compute_success_rate(summary, trans_tol, rot_tol)
        best_success_rate = max(best_success_rate, success_rate)
    return best_success_rate


if __name__ == "__main__":
    num_real = 50
    num_sim = [500, 2000, 4000, 8000]
    eval_dir = "eval/sim_sim/cotrain_carbon"
    baseline_dir = f"eval/sim_sim/baseline_carbon/{num_real}"
    data_mixtures = [f"{num_real}_{num}" for num in num_sim]

    trans_tols = np.linspace(0.0125, 0.0175, 3)
    rot_tols = np.linspace(np.deg2rad(3), np.deg2rad(4), 3)
    x, y = np.meshgrid(trans_tols, rot_tols)
    m, n = x.shape

    fig, axs = plt.subplots(m, n, figsize=(20, 20))
    for i in range(m):
        for j in range(n):
            trans_tol = x[i, j]
            rot_tol = y[i, j]
            baseline_real = compute_baseline_results(baseline_dir, trans_tol, rot_tol)
            baseline_real = 0.59
            results = compute_results(eval_dir, data_mixtures, trans_tol, rot_tol)
            alphas = [None, 0.25, 0.5, 0.75]
            data = [
                [results[mixture].get(alpha, float("nan")) for alpha in alphas]
                for mixture in data_mixtures
            ]
            legend = [f"{num_real} real, {num_sim} sim" for num_sim in num_sim]
            title = f"trans = {100*trans_tol:.2f}cm, rot = {np.rad2deg(rot_tol):.2f}"
            plot_alpha_sweep_subplot(
                axs[i][j],
                alphas,
                np.array(data),
                num_real,
                num_sim,
                legend=legend,
                real_only_baseline=[baseline_real],
                title=title,
            )

    plt.tight_layout()
    plt.show()
