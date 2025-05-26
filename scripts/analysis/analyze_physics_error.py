import importlib
import logging
import os
import pathlib
import pickle
import random
import shutil
import time
from enum import Enum
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import zarr
from omegaconf import OmegaConf
from pydrake.all import HPolyhedron, StartMeshcat, VPolytope

from planning_through_contact.experiments.utils import get_default_plan_config
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    PlanarPushingWorkspace,
)
from planning_through_contact.simulation.controllers.cylinder_actuated_station import (
    CylinderActuatedStation,
)
from planning_through_contact.simulation.controllers.iiwa_hardware_station import (
    IiwaHardwareStation,
)
from planning_through_contact.simulation.controllers.physics_analysis_source import (
    PhysicsAnalysisSource,
)
from planning_through_contact.simulation.controllers.robot_system_base import (
    RobotSystemBase,
)
from planning_through_contact.simulation.environments.simulated_real_table_environment import (
    SimulatedRealTableEnvironment,
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


class AnalyzePhysicsError:
    def __init__(self, pusher_traj, slider_traj, delay, cfg, sim_config, meshcat):
        # start meshcat
        self.sim_config = sim_config
        self.pusher_traj = pusher_traj
        self.slider_traj = slider_traj
        self.delay = delay

        if self.sim_config.use_realtime:
            print_blue("Setting use_realtime to False for faster eval")
            self.sim_config.use_realtime = False
        assert self.sim_config.use_realtime == False

        if cfg.slider_type == "arbitrary":
            # create arbitrary shape sdf file
            create_arbitrary_shape_sdf_file(cfg, self.sim_config)

        # Diffusion Policy
        position_source = PhysicsAnalysisSource(pusher_traj, delay, 0.1)

        # Set up position controller
        module_name, class_name = cfg.robot_station._target_.rsplit(".", 1)
        robot_system_class = getattr(importlib.import_module(module_name), class_name)
        position_controller: RobotSystemBase = robot_system_class(
            sim_config=self.sim_config, meshcat=meshcat
        )

        # Set up environment
        self.environment = SimulatedRealTableEnvironment(
            desired_position_source=position_source,
            robot_system=position_controller,
            sim_config=self.sim_config,
            station_meshcat=meshcat,
            arbitrary_shape_pickle_path=cfg.arbitrary_shape_pickle_path,
        )
        # self.environment.export_diagram("analyze_physics_error_environment.pdf")

        # Useful variables for querying mbp
        self.plant = self.environment._plant
        self.mbp_context = self.environment.mbp_context
        self.pusher_body = self.plant.GetBodyByName("pusher")
        self.robot_model_instance = self.environment._robot_model_instance
        self.slider_model_instance = self.environment._slider_model_instance

        if isinstance(self.environment._robot_system, IiwaHardwareStation):
            self.run_flag_port = self.environment._robot_system.GetOutputPort(
                "run_flag"
            )
        self.robot_system_context = self.environment.robot_system_context

    def simulate_environment(
        self,
        end_time: float,
        recording_file: Optional[str] = None,
    ):
        # Loop variables
        time_step = 0.1
        t = time_step
        meshcat = self.environment._meshcat

        # Simulate
        i = 0
        reset_slider = False
        pos_error = []
        rot_error = []
        while t < end_time:
            self.environment._simulator.AdvanceTo(t)

            if not reset_slider and t > self.delay - 1:
                # Reset slider
                initial_slider_pose = PlanarPose(
                    self.slider_traj[i, 0],
                    self.slider_traj[i, 1],
                    self.slider_traj[i, 2],
                )
                self.environment.reset(
                    robot_position=None,
                    slider_pose=initial_slider_pose,
                    pusher_pose=None,
                )
                reset_slider = True

            # Loop updates
            t += time_step
            t = round(t / time_step) * time_step

            rel_t = t - self.delay
            if rel_t >= 0:
                slider_pose = self.get_slider_pose()
                slider_pos = slider_pose.pos().flatten()
                slider_rot = slider_pose.theta
                pos_error.append(np.linalg.norm(slider_pos - self.slider_traj[i, :2]))
                rot_error.append(np.abs(slider_rot - self.slider_traj[i, 2]))
                i += 1

        return np.array(pos_error), np.array(rot_error)

    def get_pusher_pose(self):
        pusher_position = self.plant.EvalBodyPoseInWorld(
            self.mbp_context, self.pusher_body
        ).translation()
        return PlanarPose(pusher_position[0], pusher_position[1], 0.0)

    def get_slider_pose(self):
        slider_pose = self.plant.GetPositions(
            self.mbp_context, self.slider_model_instance
        )
        return PlanarPose.from_generalized_coords(slider_pose)

    def get_robot_joint_angles(self):
        return self.plant.GetPositions(self.mbp_context, self.robot_model_instance)


def print_blue(text, end="\n"):
    print(f"\033[94m{text}\033[0m", end=end)


def generate_intervals(slider_episode, horizon, zero_threshold=1e-6):
    # detect intervals of constant movement
    deltas = np.linalg.norm(np.diff(slider_episode, axis=0), axis=1)
    deltas[deltas <= zero_threshold] = 0
    contact_intervals = []
    interval_starts = []
    interval_ends = []
    for j in range(1, deltas.shape[0]):
        if deltas[j] > zero_threshold and deltas[j - 1] == 0:
            interval_starts.append(j - 1)
        elif deltas[j] == 0 and deltas[j - 1] > zero_threshold:
            interval_ends.append(j)
    assert len(interval_starts) == len(interval_ends)
    contact_intervals = [
        (interval_starts[i], interval_ends[i]) for i in range(len(interval_starts))
    ]

    intervals = []
    for contact_interval in contact_intervals:
        interval_length = contact_interval[1] - contact_interval[0]
        num_intervals = interval_length // horizon
        for i in range(num_intervals):
            intervals.append(
                (
                    contact_interval[0] + i * horizon,
                    contact_interval[0] + (i + 1) * horizon,
                )
            )

    return intervals


def shift_slider_com(slider_traj, com_y_shift):
    # To accomodate for physics shift implementation
    theta = slider_traj[:, 2]
    slider_traj[:, 0] += com_y_shift * np.sin(theta)
    slider_traj[:, 1] -= com_y_shift * np.cos(theta)
    return slider_traj


def compute_physics_error(
    zarr_path,
    com_y_shifts,
    num_traj,
    plot_name,
    delay,
    horizon,
    cfg,
    sim_config,
    meshcat,
):
    root = zarr.open(zarr_path)
    pusher_state = np.array(root["data/state"])
    slider_state = np.array(root["data/slider_state"])
    episode_ends = np.array(root["meta/episode_ends"])
    threshold = 5e-3

    pos_errors = []
    rot_errors = []
    episode_start = 0
    num_intervals = 0
    for i in range(num_traj):
        print_blue(f"Starting trajectory {i}")
        episode_end = episode_ends[i]
        pusher_episode = pusher_state[episode_start:episode_end]
        slider_episode = slider_state[episode_start:episode_end]
        slider_episode = shift_slider_com(slider_episode, com_y_shifts)
        episode_length = episode_end - episode_start

        intervals = generate_intervals(slider_episode, horizon, threshold)
        for interval in intervals:
            pusher_traj = pusher_episode[interval[0] : interval[1]]
            slider_traj = slider_episode[interval[0] : interval[1]]

            # run simulation
            analyze_physics_error = AnalyzePhysicsError(
                pusher_traj, slider_traj, delay, cfg, sim_config, meshcat
            )
            pos_error, rot_error = analyze_physics_error.simulate_environment(
                delay + 0.1 * (len(pusher_traj) - 1)
            )
            pos_errors.append(pos_error)
            rot_errors.append(rot_error)

            num_intervals += 1
            print_blue(f"Completed {num_intervals} intervals.")
        episode_start = episode_end

    # Save pos_errors and rot_errors
    if not os.path.exists(f"eval/physics_error"):
        os.makedirs(f"eval/physics_error")
    pos_errors = np.array(pos_errors)
    rot_errors = np.array(rot_errors)
    with open(f"eval/physics_error/{plot_name}_pos_error.pkl", "wb") as f:
        pickle.dump(pos_errors, f)
    with open(f"eval/physics_error/{plot_name}_rot_error.pkl", "wb") as f:
        pickle.dump(rot_errors, f)
    print("Saved pos_errors and rot_errors")

    # Compute mean of pos_errors and rot_errors over time
    pos_errors_mean = np.mean(pos_errors, axis=0)
    rot_errors_mean = np.mean(rot_errors, axis=0)
    pos_errors_std = np.std(pos_errors, axis=0)
    rot_errors_std = np.std(rot_errors, axis=0)

    # plot pos error and rot error on 2 subplots
    fig, axs = plt.subplots(2)
    axs[0].errorbar(
        np.arange(horizon) * 0.1, 100 * pos_errors_mean, yerr=100 * pos_errors_std
    )
    axs[0].set_title("Position Error")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Error [cm]")

    axs[1].errorbar(np.arange(horizon) * 0.1, rot_errors_mean, yerr=rot_errors_std)
    axs[1].set_title("Rotation Error")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Error [rad]")
    plt.savefig(f"eval/physics_error/{plot_name}.png")

    return pos_errors_mean, rot_errors_mean


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[2].joinpath("config", "sim_config")),
)
def main(cfg: OmegaConf):
    zarr_paths = [
        f"/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/sim_tee_data_large.zarr",
        f"/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/physics_shift/physics_shift_level_1.zarr",
        f"/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/physics_shift/physics_shift_level_2.zarr",
        f"/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/physics_shift/physics_shift_level_3.zarr",
    ]
    plot_names = [
        "physics_error_level_0",
        "physics_error_level_1",
        "physics_error_level_2",
        "physics_error_level_3",
    ]
    com_y_shifts = [0, 0.03, -0.03, -0.06]

    delay = 4
    horizon = 21  # for (horizon-1)*0.1s worth of data
    num_traj = 50
    meshcat = StartMeshcat()

    for i in range(len(zarr_paths)):
        # Only run one at a time (for now)
        if i != 3:
            continue

        zarr_path = zarr_paths[i]
        com_y_shift = com_y_shifts[i]
        plot_name = plot_names[i]

        sim_config = PlanarPushingSimConfig.from_yaml(cfg)
        pos_error_mean, rot_error_mean = compute_physics_error(
            zarr_path,
            com_y_shift,
            num_traj,
            plot_name,
            delay,
            horizon,
            cfg,
            sim_config,
            meshcat,
        )

    # # Plot all errors
    pos_error_means = []
    rot_error_means = []
    for i in range(len(zarr_paths)):
        pos_error_path = f"eval/physics_error/{plot_names[i]}_pos_error.pkl"
        rot_error_path = f"eval/physics_error/{plot_names[i]}_rot_error.pkl"
        with open(pos_error_path, "rb") as f:
            pos_error = pickle.load(f)
        with open(rot_error_path, "rb") as f:
            rot_error = pickle.load(f)
        pos_error_means.append(pos_error.mean(axis=0))
        rot_error_means.append(rot_error.mean(axis=0))

    fig, axs = plt.subplots(2)
    for i in range(len(pos_error_means)):
        axs[0].plot(100 * pos_error_means[i], label=plot_names[i])
    axs[0].set_title("Position Error")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Error [cm]")
    axs[0].legend()

    for i in range(len(rot_error_means)):
        axs[1].plot(rot_error_means[i], label=plot_names[i])
    axs[1].set_title("Rotation Error")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Error [rad]")
    axs[1].legend()
    plt.savefig(f"eval/physics_error/physics_error_all_levels.png")


if __name__ == "__main__":
    """
    Configure sim config through hydra yaml file
    Ex: python analysis/analyze_physics_error.py --config-dir <dir> --config-name <file>
    """
    main()
