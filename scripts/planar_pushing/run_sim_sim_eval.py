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
from planning_through_contact.simulation.controllers.diffusion_policy_source import (
    DiffusionPolicySource,
)
from planning_through_contact.simulation.controllers.iiwa_hardware_station import (
    IiwaHardwareStation,
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


class Result(Enum):
    NONE = "none"
    SLIDER_FELL_OFF_TABLE = "slider fell"
    TIMEOUT = "timeout"
    MISSED_GOAL = "missed goal"
    ELBOW_DOWN = "elbow down"
    SUCCESS = "success"


class SimulationMode(Enum):
    EVAL = "eval"
    RETURN_TO_START = "return_to_start"


class SimSimEval:
    def __init__(self, cfg: OmegaConf, output_dir=None):
        # start meshcat
        print(f"Station meshcat")
        station_meshcat = StartMeshcat()

        if cfg.use_realtime:
            print_blue("Setting use_realtime to False for faster eval")
            cfg.use_realtime = False

        # load sim_config
        self.cfg = cfg
        self.output_dir = output_dir
        self.sim_config = PlanarPushingSimConfig.from_yaml(cfg)
        self.multi_run_config = self.sim_config.multi_run_config
        self.pusher_start_pose = self.sim_config.pusher_start_pose
        self.slider_goal_pose = self.sim_config.slider_goal_pose
        print(f"Initial pusher pose: {self.pusher_start_pose}")
        print(f"Target slider pose: {self.slider_goal_pose}")
        assert self.sim_config.use_realtime == False

        self.continue_eval = False
        if "continue_eval" in cfg and cfg.continue_eval:
            self.continue_eval = True
            assert os.path.exists(os.path.join(self.output_dir, "summary.pkl"))

        self.workspace = self.multi_run_config.workspace
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

        # Diffusion Policy
        position_source = DiffusionPolicySource(self.sim_config.diffusion_policy_config)

        # Set up position controller
        module_name, class_name = cfg.robot_station._target_.rsplit(".", 1)
        robot_system_class = getattr(importlib.import_module(module_name), class_name)
        position_controller: RobotSystemBase = robot_system_class(
            sim_config=self.sim_config, meshcat=station_meshcat
        )

        # Set up environment
        self.environment = SimulatedRealTableEnvironment(
            desired_position_source=position_source,
            robot_system=position_controller,
            sim_config=self.sim_config,
            station_meshcat=station_meshcat,
            arbitrary_shape_pickle_path=cfg.arbitrary_shape_pickle_path,
        )
        self.environment.export_diagram("sim_sim_environment.pdf")

        # Set up random seeds
        random.seed(self.multi_run_config.seed)
        np.random.seed(self.multi_run_config.seed)

        # Random initial condition
        self.reset_environment()

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

        # Success_criteria
        valid_success_criteria = ["tolerance", "convex_hull"]
        self.success_criteria = self.multi_run_config.success_criteria
        assert self.success_criteria in valid_success_criteria

        if self.success_criteria == "convex_hull":
            dataset_path = self.multi_run_config.dataset_path
            convex_hull = self.get_pusher_goal_polyhedron(dataset_path)
            self.pusher_goal_convex_hull = convex_hull.Scale(
                scale=self.multi_run_config.convex_hull_scale,
                center=self.pusher_start_pose.vector().flatten()[:2],
            )
            convex_hull = self.get_slider_goal_polyhedron(dataset_path)
            self.slider_goal_convex_hull = convex_hull.Scale(
                scale=self.multi_run_config.convex_hull_scale,
                center=self.slider_goal_pose.vector().flatten(),
            )

        # Delete log file if it already exists
        if os.path.exists(os.path.join(self.output_dir, "summary.txt")):
            os.remove(os.path.join(self.output_dir, "summary.txt"))

    def simulate_environment(
        self,
        end_time: float,
        recording_file: Optional[str] = None,
    ):
        # Loop variables
        time_step = self.sim_config.time_step * 10
        t = time_step
        last_reset_time = 0.0
        num_completed_trials = 0
        prev_completed_trials = 0
        meshcat = self.environment._meshcat
        sim_mode = SimulationMode.EVAL
        prev_run_flag = True  # default value is True

        summary = {
            "successful_trials": [],
            "trial_times": [],
            "initial_conditions": [self.get_slider_pose().vector()],
            "final_error": [],
            "trial_result": [],
            "total_eval_sim_time": 0.0,
            "total_eval_wall_time": 0.0,
        }

        if self.continue_eval:
            # Load in existing summary from pkl file
            with open(os.path.join(self.output_dir, "summary.pkl"), "rb") as f:
                summary = pickle.load(f)

            # update prev_completed_trials, num_completed_trials
            prev_completed_trials = len(summary["trial_times"])
            num_completed_trials = prev_completed_trials
            summary["initial_conditions"].append(self.get_slider_pose().vector())

            # Override existing summary.txt
            with open(os.path.join(self.output_dir, "summary.txt"), "w") as f:
                for i in range(prev_completed_trials):
                    self.update_summary(
                        i,
                        Result(summary["trial_result"][i]),
                        summary["trial_times"][i],
                        summary["initial_conditions"][i],
                        summary["final_error"][i],
                    )

        # Simulate
        start_time = time.time()
        if not self.continue_eval:
            meshcat.StartRecording(frames_per_second=10)
        self.environment.visualize_desired_slider_pose()
        self.environment.visualize_desired_pusher_pose()
        while t < end_time:
            self.environment._simulator.AdvanceTo(t)

            if sim_mode == SimulationMode.EVAL:
                # Waiting for policy delay
                if self.get_trial_duration(t, last_reset_time) < 0.0:
                    t += time_step
                    t = round(t / time_step) * time_step
                    continue

                reset_environment = False
                success = self.check_success()
                if success:
                    reset_environment = True
                    result = Result.SUCCESS
                    summary["successful_trials"].append(num_completed_trials)
                    summary["trial_result"].append(Result.SUCCESS.value)
                    summary["trial_times"].append(
                        self.get_trial_duration(t, last_reset_time)
                    )
                # Check for failure
                else:
                    failure, mode = self.check_failure(t, last_reset_time)
                    if failure:
                        reset_environment = True
                        summary["trial_result"].append(mode.value)
                        result = mode
                        if mode == Result.TIMEOUT or mode == Result.MISSED_GOAL:
                            summary["trial_times"].append(
                                self.multi_run_config.max_attempt_duration
                            )
                        else:
                            summary["trial_times"].append(
                                self.get_trial_duration(t, last_reset_time)
                            )

                # Reset environment
                if reset_environment:
                    # Logging
                    final_error = self.get_final_error()
                    summary["final_error"].append(final_error)
                    self.update_summary(
                        num_completed_trials,
                        result,
                        summary["trial_times"][-1],
                        summary["initial_conditions"][-1],
                        final_error,
                    )
                    combined_logs = self.save_log(
                        f"combined_logs_{num_completed_trials}.pkl",
                        min(t - summary["trial_times"][-1], t),
                        t,
                    )
                    self.save_plot(
                        combined_logs,
                        f"{self.output_dir}/analysis/{num_completed_trials:03}.png",
                        result,
                    )

                    # Reset environment
                    if isinstance(self.environment._robot_system, IiwaHardwareStation):
                        self.plan_to_start()
                    sim_mode = SimulationMode.RETURN_TO_START
                    num_completed_trials += 1

                    if (
                        num_completed_trials
                        >= self.multi_run_config.num_trials_to_record
                        and not self.continue_eval
                    ):
                        meshcat.StopRecording()

                # Finished Eval
                if (
                    num_completed_trials - prev_completed_trials
                    >= self.multi_run_config.num_runs
                ):
                    break
            elif sim_mode == SimulationMode.RETURN_TO_START:
                # Repeatedly reset diffusion policy until run_flag is True
                self.reset_controller()
                if isinstance(self.environment._robot_system, CylinderActuatedStation):
                    should_reset = True
                    run_flag = True  # dummy value
                else:
                    # evaluate run_flag/reset
                    run_flag = bool(self.run_flag_port.Eval(self.robot_system_context))
                    should_reset = run_flag and not prev_run_flag

                if should_reset:  # run flag switched from False to True
                    self.reset_environment()
                    last_reset_time = t
                    summary["initial_conditions"].append(
                        self.get_slider_pose().vector()
                    )
                    sim_mode = SimulationMode.EVAL
                prev_run_flag = run_flag
            else:
                raise ValueError(f"Invalid mode: {sim_mode}")

            # Loop updates
            t += time_step
            t = round(t / time_step) * time_step

        # Save logs
        summary["total_eval_sim_time"] += t
        summary["total_eval_wall_time"] += time.time() - start_time
        if not self.continue_eval and self.multi_run_config.num_trials_to_record > 0:
            self.environment.save_recording("eval.html", self.output_dir)
        self.save_summary(summary)
        self.print_summary(os.path.join(self.output_dir, "summary.txt"))

    def check_success(self):
        if self.success_criteria == "tolerance":
            return self._check_success_tolerance(
                self.multi_run_config.trans_tol, self.multi_run_config.rot_tol
            )
        elif self.success_criteria == "convex_hull":
            return self._check_success_convex_hull()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _check_success_tolerance(self, trans_tol, rot_tol):
        # slider
        slider_pose = self.get_slider_pose()
        slider_goal_pose = self.sim_config.slider_goal_pose
        slider_error = slider_goal_pose.vector() - slider_pose.vector()
        reached_goal_slider_position = np.linalg.norm(slider_error[:2]) <= trans_tol
        reached_goal_slider_orientation = np.abs(slider_error[2]) <= np.deg2rad(rot_tol)

        # pusher
        pusher_pose = self.get_pusher_pose()
        pusher_goal_pose = self.sim_config.pusher_start_pose
        pusher_error = pusher_goal_pose.vector() - pusher_pose.vector()
        # Note: pusher goal criterion is intentionally very lenient
        # since the teleoperator (me) did a poor job as well, oops :) oops
        reached_goal_pusher_position = np.linalg.norm(pusher_error[:2]) <= 0.04

        if not reached_goal_slider_position:
            return False
        if (
            self.multi_run_config.evaluate_final_slider_rotation
            and not reached_goal_slider_orientation
        ):
            return False
        if (
            self.multi_run_config.evaluate_final_pusher_position
            and not reached_goal_pusher_position
        ):
            return False
        return True

    def check_close_to_goal(self):
        return self._check_success_tolerance(
            2 * self.multi_run_config.trans_tol, 2 * self.multi_run_config.rot_tol
        )

    def _check_success_convex_hull(self):
        slider_pose = self.get_slider_pose().vector()
        pusher_position = self.get_pusher_pose().vector()[:2]

        slider_success = self.is_contained(slider_pose, self.slider_goal_convex_hull)
        pusher_success = self.is_contained(
            pusher_position, self.pusher_goal_convex_hull
        )
        return slider_success and pusher_success

    def check_failure(self, t, last_reset_time):
        # Check timeout
        duration = self.get_trial_duration(t, last_reset_time)
        if duration > self.multi_run_config.max_attempt_duration:
            if self.check_close_to_goal():
                return True, Result.MISSED_GOAL
            else:
                return True, Result.TIMEOUT

        # Check if slider is on table
        slider_pose = self.plant.GetPositions(
            self.mbp_context, self.slider_model_instance
        )
        if slider_pose[-1] < 0.0:  # z value
            return True, Result.SLIDER_FELL_OFF_TABLE

        q = self.get_robot_joint_angles()
        if len(q) == 7:
            ELBOW_INDEX = 3
            ELBOW_THRESHOLD = np.deg2rad(5)
            elbow_angle = q[ELBOW_INDEX]
            if elbow_angle > ELBOW_THRESHOLD:
                return True, Result.ELBOW_DOWN

        # No immediate failures
        return False, Result.NONE

    def get_trial_duration(self, t, last_reset_time):
        return t - last_reset_time - self.sim_config.diffusion_policy_config.delay

    def get_final_error(self):
        pusher_pose = self.get_pusher_pose()
        pusher_goal_pose = self.sim_config.pusher_start_pose
        pusher_error = pusher_goal_pose.vector() - pusher_pose.vector()

        slider_pose = self.get_slider_pose()
        slider_goal_pose = self.sim_config.slider_goal_pose
        slider_error = slider_goal_pose.vector() - slider_pose.vector()

        return {"pusher_error": pusher_error[:2], "slider_error": slider_error}

    def reset_environment(self):
        slider_geometry = self.sim_config.dynamics_config.slider.geometry
        slider_pose = get_slider_pose_within_workspace(
            self.workspace, slider_geometry, self.pusher_start_pose, self.plan_config
        )

        self.environment.reset(
            self.sim_config.default_joint_positions,
            slider_pose,
            self.pusher_start_pose,
        )

    def reset_controller(self):
        self.environment.reset(
            None,
            None,
            self.pusher_start_pose,
        )

    def plan_to_start(self):
        self.environment._robot_system._planner.reset()

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

    def get_pusher_goal_polyhedron(self, dataset_path):
        root = zarr.open(dataset_path, mode="r")
        indices = np.array(root["meta/episode_ends"]) - 1
        state = np.array(root["data/state"])
        final_positions = state[indices][:, :2]
        return HPolyhedron(VPolytope(final_positions.transpose()))

    def get_slider_goal_polyhedron(self, dataset_path):
        root = zarr.open(dataset_path, mode="r")
        indices = np.array(root["meta/episode_ends"]) - 1
        state = np.array(root["data/slider_state"])
        final_states = state[indices]
        return HPolyhedron(VPolytope(final_states.transpose()))

    def is_contained(self, point, polyhedron):
        A, b = polyhedron.A(), polyhedron.b()
        return np.all(A @ point <= b)

    # Logging infrastructure
    def save_log(self, log_name, start_time, end_time=None):
        pusher_log = self.environment.get_pusher_pose_log()
        pusher_actual = self.get_planar_pushing_log(pusher_log, start_time, end_time)
        slider_log = self.environment.get_slider_pose_log()
        slider_actual = self.get_planar_pushing_log(slider_log, start_time, end_time)

        combined_logs = CombinedPlanarPushingLogs(
            pusher_desired=None,
            slider_desired=None,
            pusher_actual=pusher_actual,
            slider_actual=slider_actual,
        )

        # Save combined_logs.pkl
        with open(f"{self.output_dir}/analysis/{log_name}", "wb") as f:
            pickle.dump(combined_logs, f)
        return combined_logs

    def get_planar_pushing_log(self, vector_log, start_time, end_time=None):
        if end_time is not None:
            assert end_time > start_time
        start_idx = 0
        sample_times = vector_log.sample_times()
        while sample_times[start_idx] < start_time:
            start_idx += 1
        if end_time is None or end_time >= sample_times[-1]:
            end_idx = len(sample_times)
        else:
            end_idx = start_idx
            while sample_times[end_idx] < end_time:
                end_idx += 1

        # sample every 4th element between start_idx and end_idx (25hz)
        t = sample_times[start_idx:end_idx:4] - sample_times[start_idx]
        nan_array = np.array([float("nan") for _ in t])
        return PlanarPushingLog(
            t=t,
            x=vector_log.data()[0, start_idx:end_idx:4],
            y=vector_log.data()[1, start_idx:end_idx:4],
            theta=vector_log.data()[2, start_idx:end_idx:4],
            lam=nan_array,
            c_n=nan_array,
            c_f=nan_array,
            lam_dot=nan_array,
        )

    def save_plot(self, planar_pushing_log, filepath, result):
        import matplotlib.pyplot as plt
        import numpy as np

        # Create 3 subplots for pusher_error, slider_error, and slider_orientation_error
        pusher_actual = planar_pushing_log.pusher_actual
        slider_actual = planar_pushing_log.slider_actual

        # Calculate errors
        pusher_error_x = np.array(pusher_actual.x) - self.pusher_start_pose.x
        pusher_error_y = np.array(pusher_actual.y) - self.pusher_start_pose.y

        slider_error_x = np.array(slider_actual.x) - self.slider_goal_pose.x
        slider_error_y = np.array(slider_actual.y) - self.slider_goal_pose.y
        slider_error_theta = np.array(slider_actual.theta) - self.slider_goal_pose.theta

        fig, axs = plt.subplots(3, 1, figsize=(6, 9))

        # Plot Pusher Error
        axs[0].plot(pusher_actual.t, pusher_error_x, label="Error x", color="C0")
        axs[0].plot(pusher_actual.t, pusher_error_y, label="Error y", color="C1")
        axs[0].set_title("Pusher Position Error")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Error (m)")
        axs[0].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        axs[0].legend()

        # Plot Slider Position Error
        axs[1].plot(slider_actual.t, slider_error_x, label="Error x", color="C0")
        axs[1].plot(slider_actual.t, slider_error_y, label="Error y", color="C1")
        axs[1].set_title("Slider Position Error")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Error (m)")
        axs[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        axs[1].legend()

        # Plot Slider Orientation Error
        axs[2].plot(
            slider_actual.t, slider_error_theta, label="Error theta", color="C2"
        )
        axs[2].set_title("Slider Orientation Error")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Error (rad)")
        axs[2].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        axs[2].legend()

        # Add the result string to the third subplot
        axs[2].text(
            0.95,
            -0.15,  # Positioned below the third plot
            f"Result: {result.value}",
            horizontalalignment="right",
            verticalalignment="center",
            transform=axs[2].transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="gray"),
        )

        # Save and close the plot
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    def update_summary(
        self, trial_idx, result, trial_time, initial_conditions, final_error
    ):
        with open(os.path.join(self.output_dir, "summary.txt"), "a") as f:
            f.write(f"Trial {trial_idx + 1}\n")
            f.write("--------------------\n")
            f.write(f"Result: {result.value}\n")
            f.write(f"Trial time: {trial_time:.2f}\n")
            f.write(f"Initial slider pose: {initial_conditions}\n")
            f.write(f"Final pusher error: {final_error['pusher_error']}\n")
            f.write(f"Final slider error: {final_error['slider_error']}\n")
            f.write("\n")

    def save_summary(self, summary):
        if len(summary["successful_trials"]) == 0:
            average_successful_trans_error = "N/A"
            average_successful_rot_error = "N/A"
        else:
            successful_translation_errors = []
            successful_rotation_errors = []
            for trial_idx in summary["successful_trials"]:
                successful_translation_errors.append(
                    np.linalg.norm(
                        summary["final_error"][trial_idx]["slider_error"][:2]
                    )
                )
                successful_rotation_errors.append(
                    np.abs(summary["final_error"][trial_idx]["slider_error"][2])
                )

            average_succesful_trans_error = np.mean(successful_translation_errors)
            average_succesful_rot_error = np.mean(successful_rotation_errors)
            average_successful_trans_error = (
                f"{100*average_succesful_trans_error:.2f}cm"
            )
            average_successful_rot_error = (
                f"{np.rad2deg(average_succesful_rot_error):.2f}°"
            )

        summary_path = os.path.join(self.output_dir, "summary.pkl")
        with open(summary_path, "wb") as f:
            pickle.dump(summary, f)

        # Read the current content
        with open(os.path.join(self.output_dir, "summary.txt"), "r") as f:
            existing_content = f.read()

        # Write the new content
        with open(os.path.join(self.output_dir, "summary.txt"), "w") as f:
            num_runs = len(summary["trial_times"])
            f.write("Evaluation Summary\n")
            f.write("====================================\n")
            f.write("Units: seconds, meters, radians\n\n")
            f.write(f"Total trials: {num_runs}\n")
            f.write(f"Total successful trials: {len(summary['successful_trials'])}\n")
            f.write(
                f"Success rate: {len(summary['successful_trials']) / num_runs:.6f}\n"
            )
            f.write(
                f"Average successful translation error: {average_successful_trans_error}\n"
            )
            f.write(
                f"Average successful rotation error: {average_successful_rot_error}\n"
            )
            f.write(f"Total time (sim): {summary['total_eval_sim_time']:.2f}\n")
            f.write(f"Total time (wall): {summary['total_eval_wall_time']:.2f}\n\n")

            f.write(f"Success criteria: {self.success_criteria}\n")
            if self.success_criteria == "tolerance":
                f.write(f"Translation tolerance: {self.multi_run_config.trans_tol}\n")
                f.write(
                    f"Rotation tolerance: {np.deg2rad(self.multi_run_config.rot_tol):.6f}\n"
                )
                f.write(
                    f"Evaluate final slider rotation: {self.multi_run_config.evaluate_final_slider_rotation}\n"
                )
                f.write(
                    f"Evaluate final pusher position: {self.multi_run_config.evaluate_final_pusher_position}\n"
                )
            f.write(
                f"Max attempt duration: {self.multi_run_config.max_attempt_duration}\n\n"
            )
            f.write(f"Workspace width: {self.cfg.multi_run_config.workspace_width}\n")
            f.write(f"Workspace height: {self.cfg.multi_run_config.workspace_height}\n")
            f.write("====================================\n\n")

            # Append the existing content
            f.write(existing_content)

    def print_summary(self, summary_path):
        with open(summary_path, "r") as file:
            for line in file:
                print_blue(line, end="")


def print_blue(text, end="\n"):
    print(f"\033[94m{text}\033[0m", end=end)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[2].joinpath("config", "sim_config")),
)
def main(cfg: OmegaConf):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if not os.path.exists(f"{output_dir}/analysis"):
        os.makedirs(f"{output_dir}/analysis")
    sim_sim_eval = SimSimEval(cfg, output_dir)
    sim_sim_eval.simulate_environment(float("inf"))


if __name__ == "__main__":
    """
    Configure sim config through hydra yaml file
    Ex: python scripts/planar_pushing/run_sim_sim_eval.py --config-dir <dir> --config-name <file>
    """
    main()
