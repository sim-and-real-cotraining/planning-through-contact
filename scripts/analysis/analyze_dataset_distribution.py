import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize


def create_video(
    zarr_paths,
    titles,
    output_file,
    com_y_shift=None,
    theta_bins=50,
    x_bins=50,
    y_bins=50,
):
    """
    Create a video comparing the x-y probability density at different theta values across Zarr datasets.

    Args:
        zarr_paths (list): List of paths to Zarr datasets.
        output_file (str): Path to save the output MP4 file.
        theta_bins (int): Number of bins for theta.
        x_bins (int): Number of bins for x.
        y_bins (int): Number of bins for y.
    """
    for i, path in enumerate(zarr_paths):
        assert os.path.exists(path), f"Zarr dataset {i + 1} does not exist at {path}"

    # Load slider states from all datasets into memory
    slider_states = [
        zarr.open(path, mode="r")["data"]["slider_state"][:] for path in zarr_paths
    ]
    if com_y_shift is not None:
        for i, shift in enumerate(com_y_shift):
            slider_theta = slider_states[i][:, 2]
            slider_states[i][:, 0] += shift * np.sin(slider_theta)
            slider_states[i][:, 1] -= shift * np.cos(slider_theta)

    theta_values = np.linspace(-np.pi, np.pi, theta_bins)

    # Find global x, y axis limits
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")

    for state in slider_states:
        x_data, y_data = state[:, 0], state[:, 1]
        x_min, x_max = min(x_min, x_data.min()), max(x_max, x_data.max())
        y_min, y_max = min(y_min, y_data.min()), max(y_max, y_data.max())

    # Create a video writer
    metadata = dict(
        title="Slider State Distribution",
        artist="Matplotlib",
        comment="Theta-based density visualization",
    )
    writer = FFMpegWriter(fps=5, metadata=metadata)  # Slower animation with fps=5

    fig, axes = plt.subplots(1, len(slider_states), figsize=(6 * len(slider_states), 6))
    if len(slider_states) == 1:
        axes = [axes]

    # Prepare the plot
    with writer.saving(fig, output_file, dpi=100):
        for theta in theta_values:
            for i, state in enumerate(slider_states):
                ax = axes[i]
                ax.clear()

                # Extract data for the current theta slice
                mask = np.abs(state[:, 2] - theta) < (2 * np.pi / theta_bins)
                x_data, y_data = state[mask, 0], state[mask, 1]

                # Plot histogram
                hist, x_edges, y_edges = np.histogram2d(
                    x_data,
                    y_data,
                    bins=[x_bins, y_bins],
                    range=[[x_min, x_max], [y_min, y_max]],
                )
                pcm = ax.pcolormesh(
                    x_edges, y_edges, hist.T, cmap="viridis", shading="auto"
                )
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_title(titles[i])
                ax.set_xlabel("X")
                ax.set_ylabel("Y")

            # Add or update the color bar for the current frame
            # if 'cbar' in locals():
            #     breakpoint()
            #     cbar.remove()
            # cbar = fig.colorbar(pcm, ax=axes, orientation="vertical", shrink=0.8, pad=0.02, aspect=30)
            # cbar.set_label("Density")

            # Add theta label
            fig.suptitle(f"Theta = {theta:.2f} rad", fontsize=16)

            # Write the frame
            writer.grab_frame()

    print(f"Video saved to {output_file}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Visualize slider state distribution over theta.")
    # parser.add_argument("--zarr-paths", nargs='+', required=True, help="Paths to the Zarr datasets.")
    # parser.add_argument("--output-file", type=str, required=True, help="Path to save the output MP4 file.")
    # parser.add_argument("--theta-bins", type=int, default=50, help="Number of bins for theta.")
    # parser.add_argument("--x-bins", type=int, default=50, help="Number of bins for x.")
    # parser.add_argument("--y-bins", type=int, default=50, help="Number of bins for y.")

    # args = parser.parse_args()

    # create_video(
    #     zarr_paths=args.zarr_paths,
    #     output_file=args.output_file,
    #     theta_bins=args.theta_bins,
    #     x_bins=args.x_bins,
    #     y_bins=args.y_bins
    # )

    zarr_paths = [
        "/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/sim_tee_data.zarr",
        "/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/physics_shift/physics_shift_level_1.zarr",
        "/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/physics_shift/physics_shift_level_2.zarr",
        "/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/physics_shift/physics_shift_level_3.zarr",
    ]
    titles = [
        "Physics Shift Level 0",
        "Physics Shift Level 1",
        "Physics Shift Level 2",
        "Physics Shift Level 3",
    ]

    com_y_shift = [0.0, 0.03, -0.03, -0.06]

    output_file = "slider_state_distribution.mp4"
    theta_bins = 50
    x_bins = 50
    y_bins = 50

    create_video(
        zarr_paths=zarr_paths,
        titles=titles,
        output_file=output_file,
        com_y_shift=com_y_shift,
        theta_bins=theta_bins,
        x_bins=x_bins,
        y_bins=y_bins,
    )
