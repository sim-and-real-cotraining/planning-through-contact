import argparse

import numpy as np
import zarr


def override_targets(zarr_path, new_value):
    """
    Override all 3D vectors in the 'target' field of the Zarr dataset with a common value.

    Args:
        zarr_path (str): Path to the Zarr dataset.
        new_value (list or tuple): 3D vector to override all targets with.
    """
    # Load Zarr dataset in write mode
    dataset = zarr.open(zarr_path, mode="r+")

    # Access the 'target' field
    targets = dataset["data"]["target"]
    breakpoint()

    # Ensure the new value is a 3D vector
    if len(new_value) != 3:
        raise ValueError("new_value must be a 3D vector.")

    # Override all targets
    new_value_array = np.array(new_value, dtype=targets.dtype)
    targets[:] = np.tile(new_value_array, (targets.shape[0], 1))

    print(f"Overridden all targets with: {new_value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Override target values in a Zarr dataset."
    )
    parser.add_argument(
        "--zarr-path", type=str, required=True, help="Path to the Zarr dataset."
    )
    parser.add_argument(
        "--override-target",
        type=float,
        nargs=3,
        required=True,
        help="Override all target values with a 3D vector.",
    )

    args = parser.parse_args()

    override_targets(args.zarr_path, args.override_target)

    # Example usage:
    # python script.py --zarr-path /path/to/dataset.zarr --override-target 1.0 2.0 3.0
