import importlib
import logging
import math
import os
import pathlib
import pickle
import shutil
from typing import List, Optional, Tuple

import cv2
import hydra
import numpy as np
import zarr
from omegaconf import OmegaConf
from PIL import Image
from pydrake.all import Meshcat, StartMeshcat
from tqdm import tqdm

from planning_through_contact.experiments.utils import (
    get_default_plan_config,
    get_default_solver_params,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarPushingWorkspace,
    PlanarSolverParams,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.simulation.controllers.replay_position_source import (
    ReplayPositionSource,
)
from planning_through_contact.simulation.controllers.robot_system_base import (
    RobotSystemBase,
)
from planning_through_contact.simulation.environments.data_collection_table_environment import (
    DataCollectionConfig,
    DataCollectionTableEnvironment,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_utils import (
    create_arbitrary_shape_sdf_file,
    get_slider_pose_within_workspace,
    get_slider_sdf_path,
    models_folder,
)
from planning_through_contact.visualize.analysis import (
    CombinedPlanarPushingLogs,
    PlanarPushingLog,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.planar_pushing import make_traj_figure

logging.getLogger("matplotlib").setLevel(logging.ERROR)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[2].joinpath("config", "sim_config")),
)
def main(cfg: OmegaConf):
    """
    Performs the data collection. Configure the data collection steps in the config file.
    The available steps are:
    - Generate plans
    - Render plans
    - Convert rendered plans to zarr format
    """

    ## Configure logs
    logging.getLogger(
        "planning_through_contact.simulation.environments.data_collection_table_environment"
    ).setLevel(logging.WARNING)
    logging.getLogger("drake").setLevel(logging.WARNING)

    ## Parse Configs
    sim_config: PlanarPushingSimConfig = PlanarPushingSimConfig.from_yaml(cfg)
    _print_sim_config_info(sim_config)

    data_collection_config: DataCollectionConfig = hydra.utils.instantiate(
        cfg.data_collection_config
    )
    _print_data_collection_config_info(data_collection_config)

    ## Generate plans
    if data_collection_config.generate_plans:
        generate_plans(data_collection_config, cfg)
        save_omegaconf(cfg, data_collection_config.plans_dir, config_name="config.yaml")

    ## Render plans
    if data_collection_config.render_plans:
        render_plans(sim_config, data_collection_config, cfg, save_recordings=False)
        save_omegaconf(
            cfg, data_collection_config.rendered_plans_dir, config_name="config.yaml"
        )

    ## Convert data to zarr
    if (
        data_collection_config.convert_to_zarr
        or data_collection_config.convert_to_zarr_reduce
    ):
        convert_to_zarr(sim_config, data_collection_config, debug=False)


def save_omegaconf(cfg: OmegaConf, dir: str, config_name: str = "config.yaml"):
    with open(f"{dir}/{config_name}", "w") as f:
        OmegaConf.save(cfg, f)


def generate_plans(data_collection_config: DataCollectionConfig, cfg: OmegaConf):
    """Generates plans according to the data collection config."""

    print("\nGenerating plans...")

    _create_directory(data_collection_config.plans_dir)

    ## Set up configs
    _plan_config = data_collection_config.plan_config
    config = get_default_plan_config(
        slider_type=_plan_config.slider_type,
        arbitrary_shape_pickle_path=_plan_config.arbitrary_shape_pickle_path,
        pusher_radius=_plan_config.pusher_radius,
        hardware=False,
        slider_physical_properties=hydra.utils.instantiate(cfg.physical_properties),
    )
    solver_params = get_default_solver_params(debug=False, clarabel=False)
    config.contact_config.lam_min = _plan_config.contact_lam_min
    config.contact_config.lam_max = _plan_config.contact_lam_max
    if _plan_config.ang_velocity_regularization is not None:
        config.contact_config.cost.ang_velocity_regularization = (
            _plan_config.ang_velocity_regularization
        )
    config.non_collision_cost.distance_to_object_socp = (
        _plan_config.distance_to_object_socp
    )

    ## Set up workspace
    workspace = PlanarPushingWorkspace(
        slider=BoxWorkspace(
            width=_plan_config.width,
            height=_plan_config.height,
            center=np.array(_plan_config.center),
            buffer=_plan_config.buffer,
        ),
    )

    ## Get starts and goals
    plan_starts_and_goals = _get_plan_start_and_goals_to_point(
        seed=_plan_config.seed,
        num_plans=int(
            1.1 * _plan_config.num_plans
        ),  # Add extra plans in case some fail
        workspace=workspace,
        config=config,
        point=_plan_config.center,
        init_pusher_pose=_plan_config.pusher_start_pose,
        limit_rotations=_plan_config.limit_rotations,
        rotation_limit=_plan_config.rotation_limit,
        noise_final_pose=_plan_config.noise_final_pose,
    )
    print(f"Finished generating start and goal pairs.")

    ## Generate plans
    pbar = tqdm(total=_plan_config.num_plans, desc="Generating plans")
    plan_idx = 0
    num_plans = 0
    while num_plans < _plan_config.num_plans and plan_idx < len(plan_starts_and_goals):
        plan = plan_starts_and_goals[plan_idx]

        success = create_plan(
            plan_spec=plan,
            config=config,
            solver_params=solver_params,
            num_unique_plans=_plan_config.num_unique_plans,
            sort_plans=_plan_config.sort_plans,
            output_dir=data_collection_config.plans_dir,
            traj_name=f"traj_{plan_idx}",
            do_rounding=True,
            save_traj=True,
        )

        if success:
            pbar.update(1)
            num_plans += 1
        else:
            print("Failed to generate plan. Retrying...")
        plan_idx += 1
    print(f"Finished generating {plan_idx} plans.")
    if num_plans < _plan_config.num_plans:
        print(f"Failed to generate all plans since the solver can fail.")


def create_plan(
    plan_spec: PlanarPushingStartAndGoal,
    config: PlanarPlanConfig,
    solver_params: PlanarSolverParams,
    num_unique_plans: int = 1,
    sort_plans: bool = True,
    output_dir: str = "",
    traj_name: str = "Untitled_traj",
    do_rounding: bool = True,
    save_traj: bool = False,
) -> bool:
    """
    Create plans according to plan_spec and other config params.
    This function is largely inspired by the 'create_plan' function in
    'scripts/planar_pushing/create_plan.py'
    """

    planner = PlanarPushingPlanner(config)
    planner.config.start_and_goal = plan_spec
    planner.formulate_problem()
    paths = planner.plan_multiple_paths(solver_params)

    if paths is None:
        print("Failed to generate plan.")
        return False
    if len(paths) < num_unique_plans:
        print("Failed to generate enough unique plans.")
        return False

    # Perform top k sorting if required
    if sort_plans:
        paths = planner.pick_top_k_paths(paths, num_unique_plans)
    else:
        paths = paths[:num_unique_plans]

    for i in range(num_unique_plans):
        path = paths[i]

        # Set up folders
        folder_name = f"{output_dir}/{traj_name}_{i}"
        os.makedirs(folder_name, exist_ok=True)
        trajectory_folder = f"{folder_name}/trajectory"
        os.makedirs(trajectory_folder, exist_ok=True)
        analysis_folder = f"{folder_name}/analysis"
        os.makedirs(analysis_folder, exist_ok=True)

        traj_relaxed = path.to_traj()
        traj_rounded = path.to_traj(rounded=True) if do_rounding else None

        if save_traj:
            if traj_rounded:
                traj_rounded.save(f"{trajectory_folder}/traj_rounded.pkl")
            else:
                traj_relaxed.save(f"{trajectory_folder}/traj_relaxed.pkl")  # type: ignore

        if traj_rounded is not None:
            slider_color = COLORS["aquamarine4"].diffuse()
            make_traj_figure(
                traj_rounded,
                filename=f"{analysis_folder}/rounded_traj",
                slider_color=slider_color,
                split_on_mode_type=True,
                show_workspace=False,
            )

    return True


def render_plans(
    sim_config: PlanarPushingSimConfig,
    data_collection_config: DataCollectionConfig,
    cfg: OmegaConf,
    save_recordings: bool = False,
):
    """Renders plans according to the configs."""

    print("\nRendering plans...")

    if cfg.slider_type == "arbitrary":
        create_arbitrary_shape_sdf_file(cfg, sim_config)

    plans = []
    plan_dirs = list(os.listdir(data_collection_config.plans_dir))
    for plan_dir in plan_dirs:
        if os.path.isdir(f"{data_collection_config.plans_dir}/{plan_dir}"):
            plan_path = f"{data_collection_config.plans_dir}/{plan_dir}/trajectory/traj_rounded.pkl"
            with open(plan_path, "rb") as f:
                plan = pickle.load(f)
            if isinstance(plan, CombinedPlanarPushingLogs):
                plans.append(plan)
            else:
                plans.append(PlanarPushingTrajectory.load(plan_path))

    meshcat = StartMeshcat()
    for plan in tqdm(plans):
        simulate_plan(
            traj=plan,
            sim_config=sim_config,
            data_collection_config=data_collection_config,
            cfg=cfg,
            meshcat=meshcat,
            save_recording=save_recordings,
        )
        meshcat.Delete()
        meshcat.DeleteAddedControls()

    if cfg.slider_type == "arbitrary":
        # Remove the sdf file.
        sdf_path = get_slider_sdf_path(sim_config, models_folder)
        if os.path.exists(sdf_path):
            os.remove(sdf_path)


def simulate_plan(
    traj: PlanarPushingTrajectory,
    sim_config: PlanarPushingSimConfig,
    data_collection_config: DataCollectionConfig,
    cfg: OmegaConf,
    meshcat: Meshcat,
    save_recording: bool = False,
):
    """Simulate a single plan to render the images."""

    position_source = ReplayPositionSource(
        traj=traj, dt=0.025, delay=sim_config.delay_before_execution
    )

    ## Set up position controller
    # TODO: load with hydra instead (currently giving camera config errors)
    module_name, class_name = cfg.robot_station._target_.rsplit(".", 1)
    robot_system_class = getattr(importlib.import_module(module_name), class_name)
    position_controller: RobotSystemBase = robot_system_class(
        sim_config=sim_config, meshcat=meshcat
    )

    environment = DataCollectionTableEnvironment(
        desired_position_source=position_source,
        robot_system=position_controller,
        sim_config=sim_config,
        data_collection_config=data_collection_config,
        state_estimator_meshcat=meshcat,
    )

    recording_name = f"recording.html" if save_recording else None
    environment.export_diagram("data_collection_table_environment.pdf")

    end_time = 0.5
    if isinstance(traj, PlanarPushingTrajectory):
        end_time += traj.end_time + sim_config.delay_before_execution
    elif isinstance(traj, CombinedPlanarPushingLogs):
        end_time += traj.pusher_desired.t[-1]
    environment.simulate(end_time, recording_file=recording_name)
    environment.resize_saved_images()


def convert_to_zarr(
    sim_config: PlanarPushingSimConfig,
    data_collection_config: DataCollectionConfig,
    debug: bool = False,
):
    """
    Converts the rendered plans to zarr format.

    This function has 2 modes (regular or reduce).
    Regular mode: Assume the rendered plans trajectory has the following structure

    rendered_plans_dir
    ├── 0
    ├──├──images
    ├──├──log.txt
    ├──├──planar_position_command.pkl
    ├── 1
    ...
    In regular mode, this function loops through all trajectories and saves the data to zarr format.

    Reduce mode: Assume the rendered plans trajectory has the following structure

    rendered_plans_dir
    ├── run_0
    ├──├── 0
    ├──├──├──images
    ├──├──├──log.txt
    ├──├──├──planar_position_command.pkl
    ├──├── 1
    ...
    ├── run_1
    ...

    In reduce mode, this function loops through all the runs. Each run contains trajectories.
    This mode is most likely used for MIT Supercloud where data generation is parallelized
    over multiple runs.
    """

    print("\nConverting data to zarr...")

    rendered_plans_dir = pathlib.Path(data_collection_config.rendered_plans_dir)
    zarr_path = data_collection_config.zarr_path

    # Collect all the data paths to compress into zarr format
    traj_dir_list = []
    if data_collection_config.convert_to_zarr_reduce:
        for run in os.listdir(rendered_plans_dir):
            run_path = rendered_plans_dir.joinpath(run)
            if not os.path.isdir(run_path):
                continue

            for plan in os.listdir(run_path):
                traj_dir = run_path.joinpath(plan)
                if not os.path.isdir(traj_dir):
                    continue
                traj_dir_list.append(traj_dir)
    else:
        for plan in os.listdir(rendered_plans_dir):
            traj_dir = rendered_plans_dir.joinpath(plan)
            if not os.path.isdir(traj_dir):
                continue
            traj_dir_list.append(traj_dir)

    concatenated_states = []
    concatenated_slider_states = []
    concatenated_actions = []
    concatenated_targets = []
    episode_ends = []
    current_end = 0

    freq = data_collection_config.policy_freq
    dt = 1 / freq

    num_ik_fails = 0
    num_angular_speed_violations = 0

    for traj_dir in tqdm(traj_dir_list):
        traj_log_path = traj_dir.joinpath("combined_logs.pkl")
        log_path = traj_dir.joinpath("log.txt")

        # If too many IK fails, skip this rollout
        if _is_ik_fail(log_path):
            num_ik_fails += 1
            continue

        # load pickle file and timing variables
        combined_logs = pickle.load(open(traj_log_path, "rb"))
        pusher_desired = combined_logs.pusher_desired
        slider_desired = combined_logs.slider_desired

        if _has_high_angular_speed(
            slider_desired,
            data_collection_config.angular_speed_threshold,
            data_collection_config.angular_speed_window_size,
        ):
            num_angular_speed_violations += 1
            continue

        t = pusher_desired.t
        total_time = math.floor(t[-1] * freq) / freq

        # get start time
        start_idx = _get_start_idx(pusher_desired)
        start_time = math.ceil(t[start_idx] * freq) / freq

        # get state, action, images
        current_time = start_time
        idx = start_idx
        state = []
        slider_state = []

        while current_time < total_time:
            # state and action
            idx = _get_closest_index(t, current_time, idx)
            current_state = np.array(
                [
                    pusher_desired.x[idx],
                    pusher_desired.y[idx],
                    pusher_desired.theta[idx],
                ]
            )
            current_slider_state = np.array(
                [
                    slider_desired.x[idx],
                    slider_desired.y[idx],
                    slider_desired.theta[idx],
                ]
            )
            state.append(current_state)
            slider_state.append(current_slider_state)

            # update current time
            current_time = round((current_time + dt) * freq) / freq

        state = np.array(state)  # T x 3
        slider_state = np.array(slider_state)  # T x 3
        action = np.array(state)[:, :2]  # T x 2
        action = np.concatenate([action[1:, :], action[-1:, :]], axis=0)  # shift action

        # get target
        target = np.array([slider_state[-1] for _ in range(len(state))])

        # update concatenated arrays
        concatenated_states.append(state)
        concatenated_slider_states.append(slider_state)
        concatenated_actions.append(action)
        concatenated_targets.append(target)
        episode_ends.append(current_end + len(state))
        current_end += len(state)

    assert num_ik_fails + num_angular_speed_violations + len(episode_ends) == len(
        traj_dir_list
    )
    print(
        f"{num_ik_fails} of {len(traj_dir_list)} rollouts were skipped due to IK fails."
    )
    print(
        f"{num_angular_speed_violations} of {len(traj_dir_list)} rollouts were skipped due to high angular speeds."
    )
    print(f"Total number of converted rollouts: {len(episode_ends)}\n")

    # save to zarr
    zarr_path = data_collection_config.zarr_path
    root = zarr.open_group(zarr_path, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    # Chunk sizes optimized for read (not for supercloud storage, sorry admins)
    state_chunk_size = (data_collection_config.state_chunk_length, state.shape[1])
    slider_state_chunk_size = (
        data_collection_config.state_chunk_length,
        state.shape[1],
    )
    action_chunk_size = (data_collection_config.action_chunk_length, action.shape[1])
    target_chunk_size = (data_collection_config.target_chunk_length, target.shape[1])

    # convert to numpy
    concatenated_states = np.concatenate(concatenated_states, axis=0)
    concatenated_slider_states = np.concatenate(concatenated_slider_states, axis=0)
    concatenated_actions = np.concatenate(concatenated_actions, axis=0)
    concatenated_targets = np.concatenate(concatenated_targets, axis=0)
    episode_ends = np.array(episode_ends)
    last_episode_end = episode_ends[-1]

    assert last_episode_end == concatenated_states.shape[0]
    assert concatenated_states.shape[0] == concatenated_slider_states.shape[0]
    assert concatenated_states.shape[0] == concatenated_actions.shape[0]
    assert concatenated_states.shape[0] == concatenated_targets.shape[0]

    data_group.create_dataset(
        "state", data=concatenated_states, chunks=state_chunk_size
    )
    data_group.create_dataset(
        "slider_state", data=concatenated_slider_states, chunks=slider_state_chunk_size
    )
    data_group.create_dataset(
        "action", data=concatenated_actions, chunks=action_chunk_size
    )
    data_group.create_dataset(
        "target", data=concatenated_targets, chunks=target_chunk_size
    )
    meta_group.create_dataset("episode_ends", data=episode_ends)

    # Delete arrays to save memory
    del concatenated_states
    del concatenated_slider_states
    del concatenated_actions
    del concatenated_targets
    del episode_ends

    # Save images separately and one at a time to save RAM
    camera_names = [camera_config.name for camera_config in sim_config.camera_configs]
    desired_image_shape = np.array(
        [data_collection_config.image_height, data_collection_config.image_width, 3]
    )
    image_chunk_size = [
        data_collection_config.image_chunk_length,
        *desired_image_shape,
    ]

    for camera_name in camera_names:
        print(f"Converting images from {camera_name} to zarr...")
        concatenated_images = zarr.zeros(
            (last_episode_end, *desired_image_shape),
            chunks=image_chunk_size,
            dtype="u1",
        )
        sequence_idx = 0

        for traj_dir in tqdm(traj_dir_list):
            traj_log_path = traj_dir.joinpath("combined_logs.pkl")
            log_path = traj_dir.joinpath("log.txt")

            # If too many IK fails, skip this rollout
            if _is_ik_fail(log_path):
                continue

            # load pickle file and timing variables
            combined_logs = pickle.load(open(traj_log_path, "rb"))
            pusher_desired = combined_logs.pusher_desired
            total_time = pusher_desired.t[-1]
            total_time = math.floor(total_time * freq) / freq

            if _has_high_angular_speed(
                combined_logs.slider_desired,
                data_collection_config.angular_speed_threshold,
                data_collection_config.angular_speed_window_size,
            ):
                continue

            # get start time
            start_idx = _get_start_idx(pusher_desired)
            start_time = math.ceil(t[start_idx] * freq) / freq
            del pusher_desired

            # Get timestamp of initial image
            image_dir = traj_dir.joinpath(camera_name)
            timestamps = [
                int(f.split(".")[0])
                for f in os.listdir(image_dir)
                if f.endswith(".png")
            ]
            first_image_time = min(timestamps) / 1000
            assert first_image_time >= 0.0

            # get state, action, images
            current_time = start_time
            idx = start_idx

            while current_time < total_time:
                idx = _get_closest_index(t, current_time, idx)

                # Image names are "{time in ms}" rounded to the nearest 100th
                image_name = (
                    round(((current_time - first_image_time) * 1000) / 100) * 100
                    + first_image_time * 1000
                )
                image_path = traj_dir.joinpath(camera_name, f"{int(image_name)}.png")
                img = Image.open(image_path).convert("RGB")
                img = np.asarray(img)
                if not np.allclose(img.shape, desired_image_shape):
                    # Image size for cv2 is (width, height) instead of (height, width)
                    img = cv2.resize(
                        img, (desired_image_shape[1], desired_image_shape[0])
                    )

                concatenated_images[sequence_idx] = img
                sequence_idx += 1

                if debug:
                    from matplotlib import pyplot as plt

                    print(f"\nCurrent time: {current_time}")
                    print(f"Current index: {idx}")
                    print(f"Image path: {image_path}")
                    plt.imshow(img[6:-6, 6:-6, :])
                    plt.show()

                current_time = round((current_time + dt) * freq) / freq
            # End episode time step loop
        # End episode loop

        # Save images to zarr
        assert len(concatenated_images) == last_episode_end
        assert sequence_idx == last_episode_end
        data_group.create_dataset(
            camera_name,
            data=concatenated_images,
            chunks=image_chunk_size,
        )

    # End camera loop


def _get_start_idx(pusher_desired):
    """
    Finds the index of the first "non-stationary" command.
    This is the index of the start of the trajectory.
    """

    length = len(pusher_desired.t)
    first_non_zero_idx = 0
    for i in range(length):
        if (
            pusher_desired.x[i] != 0
            or pusher_desired.y[i] != 0
            or pusher_desired.theta[i] != 0
        ):
            first_non_zero_idx = i
            break

    initial_state = np.array(
        [
            pusher_desired.x[first_non_zero_idx],
            pusher_desired.y[first_non_zero_idx],
            pusher_desired.theta[first_non_zero_idx],
        ]
    )
    assert not np.allclose(initial_state, np.array([0.0, 0.0, 0.0]))

    for i in range(first_non_zero_idx + 1, length):
        state = np.array(
            [pusher_desired.x[i], pusher_desired.y[i], pusher_desired.theta[i]]
        )
        if not np.allclose(state, initial_state):
            return i

    return None


def _is_ik_fail(log_path, max_failures=5):
    with open(log_path, "r") as f:
        line = f.readline()
        if len(line) != 0:
            ik_fails = int(line.rsplit(" ", 1)[-1])
            if ik_fails > max_failures:
                return True
    return False


def _get_closest_index(arr, t, start_idx=None, end_idx=None):
    """Returns index of arr that is closest to t."""

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(arr)

    min_diff = float("inf")
    min_idx = -1
    eps = 1e-4
    for i in range(start_idx, end_idx):
        diff = abs(arr[i] - t)
        if diff > min_diff:
            return min_idx
        if diff < eps:
            return i
        if diff < min_diff:
            min_diff = diff
            min_idx = i


def _compute_angular_speed(time, orientation):
    dt = np.diff(time)
    dtheta = np.diff(orientation)
    angular_speed = abs(dtheta / dt)

    # Remove sharp angular velocity at beginning
    first_zero_idx = -1
    for i in range(len(angular_speed)):
        if np.allclose(angular_speed[i], 0.0):
            first_zero_idx = i
            break

    return angular_speed[first_zero_idx:]


# Function to identify high angular speed moments
def _has_high_angular_speed(slider_desired, threshold, window_size):
    if threshold is None:
        return False

    angular_speed = _compute_angular_speed(slider_desired.t, slider_desired.theta)
    angular_speed_cumsum = np.cumsum(angular_speed)
    max_window_avg = -1
    ret = False
    for i in range(len(angular_speed_cumsum) - window_size):
        window_avg = (
            angular_speed_cumsum[i + window_size] - angular_speed_cumsum[i]
        ) / window_size
        max_window_avg = max(max_window_avg, window_avg)
        if window_avg > threshold:
            return True
    return False


def _get_plan_start_and_goals_to_point(
    seed: int,
    num_plans: int,
    workspace: PlanarPushingWorkspace,
    config: PlanarPlanConfig,
    point: Tuple[float, float] = (0, 0),  # Default is origin
    init_pusher_pose: Optional[PlanarPose] = None,
    limit_rotations: bool = True,  # Use this to start with
    rotation_limit: float = None,
    noise_final_pose: bool = False,
) -> List[PlanarPushingStartAndGoal]:
    """Get start and goal pairs for planar pushing task"""

    # We want the plans to always be the same
    np.random.seed(seed)

    slider = config.slider_geometry

    # Hardcoded pusher start pose to be at top edge
    # of workspace
    ws = workspace.slider.new_workspace_with_buffer(new_buffer=0)
    if init_pusher_pose is not None:
        pusher_pose = init_pusher_pose
    else:
        pusher_pose = PlanarPose(ws.x_min, 0, 0)

    plans = []
    for _ in range(num_plans):
        slider_initial_pose = get_slider_pose_within_workspace(
            workspace, slider, pusher_pose, config, limit_rotations, rotation_limit
        )

        if noise_final_pose:
            tran_tol = 0.01  # 0.01cm
            rot_tol = 1 * np.pi / 180  # 1 degrees
            slider_target_pose = PlanarPose(
                point[0] + np.random.uniform(-tran_tol, tran_tol),
                point[1] + np.random.uniform(-tran_tol, tran_tol),
                0 + np.random.uniform(-rot_tol, rot_tol),
            )
        else:
            slider_target_pose = PlanarPose(point[0], point[1], 0)

        plans.append(
            PlanarPushingStartAndGoal(
                slider_initial_pose, slider_target_pose, pusher_pose, pusher_pose
            )
        )

    return plans


def _print_data_collection_config_info(data_collection_config: DataCollectionConfig):
    """Output diagnostic info about the data collection configuration."""

    print("This data collection script is configured to perform the following steps.\n")
    step_num = 1
    if data_collection_config.generate_plans:
        print(
            f"{step_num}. Generate new plans in '{data_collection_config.plans_dir}' "
            f"according to the following config:"
        )
        print(data_collection_config.plan_config, end="\n\n")
        step_num += 1
    if data_collection_config.render_plans:
        print(
            f"{step_num}. Render the plans in '{data_collection_config.plans_dir}' "
            f"to '{data_collection_config.rendered_plans_dir}'\n"
        )
        step_num += 1
    if data_collection_config.convert_to_zarr:
        print(
            f"{step_num}. Convert the rendered plans in '{data_collection_config.rendered_plans_dir}' "
            f"to zarr format in '{data_collection_config.zarr_path}'"
        )
        if data_collection_config.convert_to_zarr_reduce:
            print(
                "Converting to zarr in 'reduce' mode (i.e. performing the reduce step of map-reduce)"
            )
            print(
                "The 'convert_to_zarr_reduce = True' flag is usually only set for Supercloud runs."
            )
        print()
        step_num += 1


def _print_sim_config_info(sim_config: PlanarPushingSimConfig):
    """Output diagnostic info about the simulation configuration."""

    print(f"Initial finger pose: {sim_config.pusher_start_pose}")
    print(f"Target slider pose: {sim_config.slider_goal_pose}")
    print()


def _create_directory(dir_path):
    """Helper function for creating directories."""

    if os.path.exists(dir_path):
        user_input = input(
            f"{dir_path} already exists. Delete existing directory? (y/n)\n"
        )
        if user_input.lower() != "y":
            print("Exiting")
            exit()
        shutil.rmtree(dir_path)
    else:
        os.makedirs(dir_path)


if __name__ == "__main__":
    main()
