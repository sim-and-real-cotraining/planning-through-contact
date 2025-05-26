import os
import shutil

import pytest

PYTHON_SCRIPT = "scripts/planar_pushing/run_data_generation.py"


@pytest.fixture
def config_path() -> str:
    return "test_data_generation_sim_config.yaml"


@pytest.fixture
def config_dir() -> str:
    return "tests/scripts/diffusion_policy"


def test_run_data_generation(config_path: str, config_dir: str):
    # Run a command on the command line using os
    command = (
        f"python {PYTHON_SCRIPT} --config-dir {config_dir} --config-name {config_path}"
    )
    os.system(command)
    passed = _passed_test()

    # Clean up test
    shutil.rmtree("tests/scripts/diffusion_policy/plans")
    shutil.rmtree("tests/scripts/diffusion_policy/rendered_plans")
    assert passed


def _passed_test() -> bool:
    # check if required paths exist in generate plans
    trajectory_path = "tests/scripts/diffusion_policy/plans"
    if not os.path.exists(
        f"{trajectory_path}/traj_0_0/analysis/rounded_traj_trajectory.pdf"
    ):
        print("rounded_traj_trajectory.pdf does not exist")
        return False
    if not os.path.exists(f"{trajectory_path}/traj_0_0/trajectory/traj_rounded.pkl"):
        print("traj_rounded.pkl does not exist")
        return False
    if not os.path.exists(f"{trajectory_path}/config.yaml"):
        print("config.yaml does not exist")
        return False

    # check if required paths exist in render plans
    rendered_path = "tests/scripts/diffusion_policy/rendered_plans"
    if not os.path.exists(f"{rendered_path}/0/overhead_camera/0.png"):
        print("0.png does not exist")
        return False
    if not os.path.exists(f"{rendered_path}/0/combined_logs.pkl"):
        print("combined_logs.pkl does not exist")
        return False
    if not os.path.exists(f"{rendered_path}/0/log.txt"):
        print("log.txt does not exist")
        return False
    if not os.path.exists(f"{rendered_path}/config.yaml"):
        print("config.yaml does not exist")
        return False

    # check if required paths exist in convert to zarr
    zarr_path = "tests/scripts/diffusion_policy/rendered_plans/data.zarr"
    data_paths = [
        "data/action",
        "data/overhead_camera",
        "data/slider_state",
        "data/state",
        "data/target",
    ]
    for data_path in data_paths:
        if not os.path.exists(f"{zarr_path}/{data_path}"):
            print(f"{data_path} does not exist")
            return False
    if not os.path.exists(f"{zarr_path}/meta/episode_ends"):
        print("episode_ends does not exist")
        return False

    return True
