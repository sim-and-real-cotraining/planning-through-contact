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
import numpy as np
from omegaconf import OmegaConf
from pydrake.all import StartMeshcat

from planning_through_contact.experiments.utils import get_default_plan_config
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    PlanarPushingWorkspace,
)
from planning_through_contact.simulation.controllers.gamepad_controller_source import (
    GamepadControllerSource,
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


class FSMState(Enum):
    REGULAR = "regular"
    DATA_COLLECTION = "data collection"
    TERMINATE = "terminate"


class GamepadDataCollection:
    def __init__(self, cfg: OmegaConf):
        seed = int(1e6 * time.time() % 1e6)
        np.random.seed(seed)

        # start meshcat
        print(f"Station meshcat")
        station_meshcat = StartMeshcat()

        # load sim_config
        self.cfg = cfg
        self.sim_config = PlanarPushingSimConfig.from_yaml(cfg)
        self.pusher_start_pose = self.sim_config.pusher_start_pose
        self.slider_goal_pose = self.sim_config.slider_goal_pose
        print(f"Initial pusher pose: {self.pusher_start_pose}")
        print(f"Target slider pose: {self.slider_goal_pose}")

        self.workspace = PlanarPushingWorkspace(
            slider=BoxWorkspace(
                width=0.5,
                height=0.35,
                center=np.array([self.slider_goal_pose.x, self.slider_goal_pose.y]),
                buffer=0,
            ),
        )
        self.plan_config = get_default_plan_config(
            slider_type=self.sim_config.slider.name
            if self.sim_config.slider.name != "t_pusher"
            else "tee",
            arbitrary_shape_pickle_path=self.sim_config.arbitrary_shape_pickle_path,
            pusher_radius=0.015,
            hardware=False,
        )

        if cfg.slider_type == "arbitrary":
            # create arbitrary shape sdf file
            create_arbitrary_shape_sdf_file(cfg, self.sim_config)

        # Gamepad Controller Source
        position_source = GamepadControllerSource(
            station_meshcat,
            translation_scale=cfg.gamepad.translation_scale,
            deadzone=cfg.gamepad.deadzone,
            gamepad_orientation=np.array(cfg.gamepad.gamepad_orientation),
        )

        # Set up position controller
        module_name, class_name = cfg.robot_station._target_.rsplit(".", 1)
        robot_system_class = getattr(importlib.import_module(module_name), class_name)
        position_controller: RobotSystemBase = robot_system_class(
            sim_config=self.sim_config, meshcat=station_meshcat
        )

        # Remove existing temporary image writer directory
        image_writer_dir = "trajectories_rendered/temp"
        if os.path.exists(image_writer_dir):
            shutil.rmtree(image_writer_dir)

        # Set up environment
        self.environment = SimulatedRealTableEnvironment(
            desired_position_source=position_source,
            robot_system=position_controller,
            sim_config=self.sim_config,
            station_meshcat=station_meshcat,
            arbitrary_shape_pickle_path=cfg.arbitrary_shape_pickle_path,
        )
        self.environment.export_diagram("gamepad_teleop_environment.pdf")

        # fsm state
        self.fsm_state = FSMState.REGULAR
        self.traj_start_time = 0.0
        self.num_saved_trajectories = 0

    def simulate_environment(
        self,
        end_time: float,
        recording_file: Optional[str] = None,
    ):
        # Loop variables
        prev_button_values = self.environment.get_button_values()
        translation_scale = (
            self.environment._desired_position_source.get_translation_scale()
        )
        time_step = self.sim_config.time_step * 10
        t = time_step
        validated_image_writer = False

        # Simulate
        self.environment.visualize_desired_slider_pose()
        self.environment.visualize_desired_pusher_pose()
        while t < end_time and self.fsm_state != FSMState.TERMINATE:
            self.environment._simulator.AdvanceTo(t)

            # Validate image writer directory
            if t > 0.2 and not validated_image_writer:
                self.validate_image_writer_dir()
                validated_image_writer = True

            # Get pressed buttons
            button_values = self.environment.get_button_values()
            pressed_buttons = self.get_pressed_buttons(
                prev_button_values, self.environment.get_button_values()
            )

            # FSM logic
            self.fsm_state, self.traj_start_time = self.fsm_logic(
                self.fsm_state, pressed_buttons, t, self.traj_start_time
            )

            if button_values["RT"]:
                self.environment._desired_position_source.set_translation_scale(
                    0.5 * translation_scale
                )
            elif button_values["LT"]:
                self.environment._desired_position_source.set_translation_scale(
                    3 * translation_scale
                )
            else:
                self.environment._desired_position_source.set_translation_scale(
                    translation_scale
                )

            # Loop updates
            t += time_step
            t = round(t / time_step) * time_step
            prev_button_values = button_values

        # Delete temporary image writer directory
        if os.path.exists("trajectories_rendered/temp"):
            shutil.rmtree("trajectories_rendered/temp")

    def fsm_logic(self, fsm_state, pressed_buttons, curr_time, traj_start_time):
        pressed_A = pressed_buttons["A"]  # Start/Save trajectory
        pressed_B = pressed_buttons["B"]  # Reset environment
        pressed_X = pressed_buttons["X"]  # Terminate

        if pressed_X:
            print_blue("Terminating data collection.")
            print_blue(f"Collected {self.num_saved_trajectories} trajectories.")
            return FSMState.TERMINATE, traj_start_time

        if fsm_state == FSMState.REGULAR:
            if pressed_A:
                print_blue(f"Entering data collection mode at time: {curr_time:.2f}")
                return FSMState.DATA_COLLECTION, curr_time
            if pressed_B:
                self.reset_environment()
                print_blue("Reset environment. Entering regular mode.")
                return FSMState.REGULAR, traj_start_time
        elif fsm_state == FSMState.DATA_COLLECTION:
            if pressed_A:
                self.save_trajectory()
                print_blue("Saved trajectory. Entering regular mode.")
                return FSMState.REGULAR, traj_start_time
            elif pressed_B:
                self.delete_trajectory()
                print_blue("Deleted trajectory. Entering regular mode.")
                return FSMState.REGULAR, traj_start_time
        return fsm_state, traj_start_time

    def reset_environment(self):
        slider_geometry = self.sim_config.dynamics_config.slider.geometry
        slider_pose = get_slider_pose_within_workspace(
            self.workspace, slider_geometry, self.pusher_start_pose, self.plan_config
        )

        self.environment.reset(
            np.array([0.6202, 1.0135, -0.5873, -1.4182, 0.6449, 0.8986, 2.9879]),
            slider_pose,
            self.pusher_start_pose,
        )

    def save_trajectory(self):
        traj_dir = self.create_trajectory_dir()

        # Move images from temp to traj_dir
        initial_image_id = int(round(self.traj_start_time, 2) * 1000)
        for camera in os.listdir("trajectories_rendered/temp"):
            camera_dir = f"trajectories_rendered/temp/{camera}"
            for file in os.listdir(camera_dir):
                image_id = int(file.split(".")[0])
                if image_id >= initial_image_id:
                    new_image_name = f"{image_id - initial_image_id}.png"
                    shutil.move(
                        f"{camera_dir}/{file}", f"{traj_dir}/{camera}/{new_image_name}"
                    )

        # Create combined_logs.pkl file
        pusher_log = self.environment.get_pusher_pose_log()
        pusher_desired = self.get_planar_pushing_log(pusher_log, self.traj_start_time)
        slider_log = self.environment.get_slider_pose_log()
        slider_desired = self.get_planar_pushing_log(slider_log, self.traj_start_time)

        combined_logs = CombinedPlanarPushingLogs(
            pusher_desired=pusher_desired,
            slider_desired=slider_desired,
            pusher_actual=None,
            slider_actual=None,
        )

        # Save combined_logs.pkl
        self.save_planar_pushing_log(combined_logs, traj_dir)
        with open(f"{traj_dir}/combined_logs.pkl", "wb") as f:
            pickle.dump(combined_logs, f)

        self.clear_image_writer_dir()
        self.reset_environment()
        self.num_saved_trajectories += 1

    def delete_trajectory(self):
        self.clear_image_writer_dir()
        self.reset_environment()

    def get_pressed_buttons(self, prev_button_values, button_values):
        pressed_buttons = {}
        for button, value in button_values.items():
            if value and not prev_button_values[button]:
                pressed_buttons[button] = True
            else:
                pressed_buttons[button] = False
        return pressed_buttons

    def create_trajectory_dir(self):
        rendered_plans_dir = self.cfg.data_collection_config.rendered_plans_dir
        if not os.path.exists(rendered_plans_dir):
            os.makedirs(rendered_plans_dir)

        # Find the next available trajectory index
        traj_idx = 0
        for path in os.listdir(rendered_plans_dir):
            if os.path.isdir(os.path.join(rendered_plans_dir, path)):
                traj_idx += 1

        # Setup the current directory
        os.makedirs(f"{rendered_plans_dir}/{traj_idx}")
        for camera in os.listdir("trajectories_rendered/temp"):
            os.makedirs(f"{rendered_plans_dir}/{traj_idx}/{camera}")
        open(f"{rendered_plans_dir}/{traj_idx}/log.txt", "w").close()
        return f"{rendered_plans_dir}/{traj_idx}"

    def clear_image_writer_dir(self):
        # remove all files in trajectories_temp/{camera}
        for camera in os.listdir("trajectories_rendered/temp"):
            camera_dir = f"trajectories_rendered/temp/{camera}"
            for file in os.listdir(camera_dir):
                os.remove(f"{camera_dir}/{file}")

    def validate_image_writer_dir(self):
        # Asserts that image writers are aligned to context time 0.0
        valid = True
        for camera in os.listdir("trajectories_rendered/temp"):
            if not os.path.exists(f"trajectories_rendered/temp/{camera}/0.png"):
                valid = False
                break

        if not valid:
            print_blue(
                "Exiting: image writer directory not aligned to context time 0.0."
            )
            print_blue("Please restart the script.")
            exit(1)

    def get_planar_pushing_log(self, vector_log, traj_start_time):
        start_idx = 0
        sample_times = vector_log.sample_times()
        while sample_times[start_idx] < traj_start_time:
            start_idx += 1

        t = sample_times[start_idx:] - sample_times[start_idx]
        nan_array = np.array([float("nan") for _ in t])
        return PlanarPushingLog(
            t=t,
            x=vector_log.data()[0, start_idx:],
            y=vector_log.data()[1, start_idx:],
            theta=vector_log.data()[2, start_idx:],
            lam=nan_array,
            c_n=nan_array,
            c_f=nan_array,
            lam_dot=nan_array,
        )

    def save_planar_pushing_log(self, planar_pushing_log, traj_dir):
        import matplotlib.pyplot as plt

        # Create 2 subplots for pusher_desired and slider_desired
        pusher_desired = planar_pushing_log.pusher_desired
        slider_desired = planar_pushing_log.slider_desired

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(pusher_desired.t, pusher_desired.x, label="x")
        axs[0].plot(pusher_desired.t, pusher_desired.y, label="y")
        axs[0].plot(pusher_desired.t, pusher_desired.theta, label="theta")
        axs[0].set_title("Pusher Desired")
        axs[0].legend()
        axs[1].plot(slider_desired.t, slider_desired.x, label="x")
        axs[1].plot(slider_desired.t, slider_desired.y, label="y")
        axs[1].plot(slider_desired.t, slider_desired.theta, label="theta")
        axs[1].set_title("Slider Desired")
        axs[1].legend()
        # plt.show() # This throws error with QT installation...
        plt.savefig(f"{traj_dir}/plot.png")
        plt.close()


def print_blue(text, end="\n"):
    print(f"\033[94m{text}\033[0m", end=end)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[2].joinpath("config", "sim_config")),
)
def main(cfg: OmegaConf):
    gamepad_data_collection = GamepadDataCollection(cfg)
    gamepad_data_collection.simulate_environment(float("inf"))


if __name__ == "__main__":
    """
    Configure sim config through hydra yaml file
    Ex: python scripts/planar_pushing/run_gamepad_teleop.py --config-dir <dir> --config-name <file>
    """
    main()
