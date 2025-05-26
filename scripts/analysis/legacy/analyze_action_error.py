import importlib
import logging
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from collections import deque
from contextlib import contextmanager
from enum import Enum
from typing import Optional

import cv2
import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
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


def print_blue(text, end="\n"):
    print(f"\033[94m{text}\033[0m", end=end)


class AnalyzeActionError:
    def __init__(self, cfg: OmegaConf, zarr_path, com_y_shift=0.0):
        if cfg.use_realtime:
            print_blue("Setting use_realtime to False for faster eval")
            cfg.use_realtime = False

        self.cfg = cfg
        self.sim_config = PlanarPushingSimConfig.from_yaml(cfg)
        self.data_collection_config = hydra.utils.instantiate(
            cfg.data_collection_config
        )
        self.zarr_path = zarr_path
        self.checkpoint = pathlib.Path(
            self.sim_config.diffusion_policy_config.checkpoint
        )
        self.device = "cuda:0"
        self.load_policy_from_checkpoint(self.checkpoint)
        self.meshcat = StartMeshcat()

        if cfg.slider_type == "arbitrary":
            # create arbitrary shape sdf file
            create_arbitrary_shape_sdf_file(cfg, self.sim_config)

        self.horizon = self.dp_cfg.n_action_steps
        self.obs_horizon = self.dp_cfg.n_obs_steps
        self.batch_size = 128
        self.com_y_shift = com_y_shift
        # hardcoded target since target changes to match com in physic shift
        self.target = np.array([0.587, -0.0355, 0.0])

    def load_policy_from_checkpoint(self, checkpoint: str):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        self.dp_cfg = payload["cfg"]

        cls = hydra.utils.get_class(self.dp_cfg._target_)
        workspace: BaseWorkspace
        workspace = cls(self.dp_cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        self._normalizer = self.load_normalizer()

        # get policy from workspace
        self.policy = workspace.model
        self.policy.set_normalizer(self._normalizer)
        if self.dp_cfg.training.use_ema:
            self.policy = workspace.ema_model
            self.policy.set_normalizer(self._normalizer)
        self.policy.to(self.device)
        self.policy.eval()

    def load_normalizer(self):
        normalizer_path = self.checkpoint.parent.parent.joinpath("normalizer.pt")
        return torch.load(normalizer_path)

    def shift_slider_com(self, slider_traj):
        # To accomodate for physics shift implementation
        theta = slider_traj[:, 2]
        slider_traj[:, 0] += self.com_y_shift * np.sin(theta)
        slider_traj[:, 1] -= self.com_y_shift * np.cos(theta)
        return slider_traj

    def run(self, num_traj, plot_name):
        # Load the zarr file
        root = zarr.open(self.zarr_path)
        pusher_state = np.array(root["data/state"])
        slider_state = np.array(root["data/slider_state"])
        episode_ends = np.array(root["meta/episode_ends"])

        mses = np.zeros((0, self.horizon))
        episode_start = 0
        offset = 0

        log_file = f"eval/action_mse/log_{plot_name.split('.')[0].split('/')[-1]}.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        for i in range(offset, min(num_traj + offset, len(episode_ends))):
            episode_end = episode_ends[i]
            pusher_traj = pusher_state[episode_start:episode_end]
            slider_traj = slider_state[episode_start:episode_end]
            slider_traj = self.shift_slider_com(slider_traj)
            mse = self.compute_action_mse(pusher_traj, slider_traj)
            max_first_mse = np.max(mse[:, 0])

            # Outlier detection
            outlier_threshold = 0.01**2  # 1cm^2

            # Check for outliers based on the first index of each point in `mse`
            initial_size = mse.shape[0]
            mse = mse[mse[:, 0] <= outlier_threshold]
            with open(log_file, "a") as f:
                f.write((f"Trajectory {i}\n--------------------\n"))
                f.write(f"Max MSE at first timestep: {max_first_mse:.6f}. ")
                removed_count = initial_size - mse.shape[0]
                if removed_count > 0:
                    f.write(
                        f"Removed {removed_count} outlier(s) out of {initial_size} points."
                    )
                f.write("\n\n")

            mses = np.vstack((mses, mse))
            episode_start = episode_end

        # Compute the mean and standard deviation of the MSEs for each column
        # mean_mse and std_mse are arrays of size horizon
        mean_mse = np.mean(mses, axis=0)
        std_mse = np.std(mses, axis=0)

        # Plot the mean MSE with error bars
        plt.errorbar(range(self.horizon), mean_mse, yerr=std_mse, fmt="o")
        for i in range(self.horizon):
            plt.text(i, mean_mse[i], f"{mean_mse[i]:.6f}", ha="center", va="bottom")
        plt.xlabel("Time step")
        plt.ylabel("Mean MSE")
        plt.title(f"Mean MSE with Error Bars ({num_traj} trajectories)")
        plt.savefig(plot_name)
        plt.close()

        return mean_mse, std_mse

    def compute_action_mse(self, pusher_traj, slider_traj):
        output_dir = self.simulate_plan(
            traj=self.create_combined_logs(pusher_traj, slider_traj),
            meshcat=self.meshcat,
        )

        mses = np.zeros((0, self.horizon))
        traj_obs_dict = self.get_traj_obs_dict(pusher_traj, slider_traj, output_dir)
        num_datapoints = len(traj_obs_dict["target"])
        num_batches = num_datapoints // self.batch_size
        if num_datapoints % self.batch_size != 0:
            num_batches += 1
        for i in range(num_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, num_datapoints)
            batch_obs_dict = {
                "target": traj_obs_dict["target"][batch_start:batch_end].to(
                    self.device
                ),
                "action": traj_obs_dict["action"][batch_start:batch_end].to(
                    self.device
                ),
                "obs": {
                    "agent_pos": traj_obs_dict["obs"]["agent_pos"][
                        batch_start:batch_end
                    ].to(self.device),
                    "overhead_camera": traj_obs_dict["obs"]["overhead_camera"][
                        batch_start:batch_end
                    ].to(self.device),
                    "wrist_camera": traj_obs_dict["obs"]["wrist_camera"][
                        batch_start:batch_end
                    ].to(self.device),
                },
            }
            with torch.no_grad():
                action = self.policy.predict_action(batch_obs_dict, use_DDIM=True)[
                    "action"
                ]
                gt_action = batch_obs_dict["action"]
                mse = (
                    torch.mean((action - gt_action) ** 2, axis=-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            mses = np.vstack((mses, mse))
            breakpoint()

            del batch_obs_dict
        del traj_obs_dict

        shutil.rmtree(output_dir)
        breakpoint()
        return mses

    def create_combined_logs(self, pusher_traj, slider_traj):
        t = np.arange(len(pusher_traj)) * 0.1
        pusher_actual = PlanarPushingLog(
            t=t,
            x=pusher_traj[:, 0],
            y=pusher_traj[:, 1],
            theta=pusher_traj[:, 2],
            lam=None,
            c_n=None,
            c_f=None,
            lam_dot=None,
        )
        slider_actual = PlanarPushingLog(
            t=t,
            x=slider_traj[:, 0],
            y=slider_traj[:, 1],
            theta=slider_traj[:, 2],
            lam=None,
            c_n=None,
            c_f=None,
            lam_dot=None,
        )
        traj = CombinedPlanarPushingLogs(
            pusher_actual=pusher_actual,
            slider_actual=slider_actual,
            pusher_desired=pusher_actual,
            slider_desired=slider_actual,
        )
        return traj

    def simulate_plan(self, traj, meshcat):
        """Simulate a single plan to render the images."""

        position_source = ReplayPositionSource(
            traj=traj, dt=0.025, delay=self.sim_config.delay_before_execution
        )

        ## Set up position controller
        # TODO: load with hydra instead (currently giving camera config errors)
        module_name, class_name = self.cfg.robot_station._target_.rsplit(".", 1)
        robot_system_class = getattr(importlib.import_module(module_name), class_name)
        position_controller: RobotSystemBase = robot_system_class(
            sim_config=self.sim_config, meshcat=meshcat
        )

        # generate temporary directory
        zp = self.zarr_path.split("/")[-1]
        temp_dir = f"./analyze_action_error_images_{zp}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        environment = DataCollectionTableEnvironment(
            desired_position_source=position_source,
            robot_system=position_controller,
            sim_config=self.sim_config,
            data_collection_config=self.data_collection_config,
            data_collection_dir=temp_dir,
            state_estimator_meshcat=meshcat,
        )
        environment.export_diagram("data_collection_table_environment.pdf")

        # Simulate the environment (silently)
        end_time = traj.pusher_desired.t[-1]
        environment.simulate(end_time)
        environment.resize_saved_images()
        return temp_dir

    def get_traj_obs_dict(self, pusher_traj, slider_traj, output_dir):
        # assert that number of files in f"{output_dir}/overhead_camera" == len(pusher_traj)
        assert len(os.listdir(f"{output_dir}/overhead_camera")) == len(pusher_traj)
        assert len(os.listdir(f"{output_dir}/wrist_camera")) == len(pusher_traj)
        assert len(pusher_traj) == len(slider_traj)

        n = len(pusher_traj)
        traj_obs_dict = {}
        traj_obs_dict["target"] = torch.zeros((0, 3))
        traj_obs_dict["obs"] = {}
        traj_obs_dict["obs"]["agent_pos"] = torch.zeros((0, self.obs_horizon, 3))
        traj_obs_dict["obs"]["overhead_camera"] = torch.zeros(
            (0, self.obs_horizon, 3, 128, 128)
        )
        traj_obs_dict["obs"]["wrist_camera"] = torch.zeros(
            (0, self.obs_horizon, 3, 128, 128)
        )
        traj_obs_dict["action"] = torch.zeros((0, self.horizon, 2))

        pusher_pose_deque = deque(maxlen=self.obs_horizon)
        overhead_deque = deque(maxlen=self.obs_horizon)
        wrist_deque = deque(maxlen=self.obs_horizon)
        for i in range(n - self.horizon - 1):
            time = i * 0.1
            pusher_pose_deque.append(pusher_traj[i])

            overhead_image = cv2.imread(
                f"{output_dir}/overhead_camera/{int(1000*time)}.png"
            )
            overhead_image = cv2.cvtColor(overhead_image, cv2.COLOR_BGR2RGB)
            overhead_deque.append(overhead_image)

            wrist_image = cv2.imread(f"{output_dir}/wrist_camera/{int(1000*time)}.png")
            wrist_image = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
            wrist_deque.append(wrist_image)

            if len(pusher_pose_deque) == self.obs_horizon:
                agent_pos = torch.cat(
                    [torch.from_numpy(obs) for obs in pusher_pose_deque], dim=0
                ).reshape(1, self.obs_horizon, 3)

                actions = pusher_traj[i + 1 : i + self.horizon + 1, :2].reshape(
                    1, self.horizon, 2
                )
                actions = torch.from_numpy(actions)

                overhead_tensor = torch.cat(
                    [
                        torch.from_numpy(np.moveaxis(img, -1, -3) / 255.0)  # C H W
                        for img in overhead_deque
                    ],
                    dim=0,
                ).reshape(
                    1,
                    self.obs_horizon,
                    overhead_image.shape[2],  # C
                    overhead_image.shape[0],  # H
                    overhead_image.shape[1],  # W
                )

                wrist_tensor = torch.cat(
                    [
                        torch.from_numpy(np.moveaxis(img, -1, -3) / 255.0)  # C H W
                        for img in wrist_deque
                    ],
                    dim=0,
                ).reshape(
                    1,
                    self.obs_horizon,
                    wrist_image.shape[2],  # C
                    wrist_image.shape[0],  # H
                    wrist_image.shape[1],  # W
                )

                # Stack the tensors into batches
                traj_obs_dict["target"] = torch.cat(
                    (
                        traj_obs_dict["target"],
                        torch.from_numpy(self.target.reshape(1, 3)),
                    ),
                    dim=0,
                )
                traj_obs_dict["obs"]["agent_pos"] = torch.cat(
                    (traj_obs_dict["obs"]["agent_pos"], agent_pos), dim=0
                )
                traj_obs_dict["obs"]["overhead_camera"] = torch.cat(
                    (traj_obs_dict["obs"]["overhead_camera"], overhead_tensor), dim=0
                )
                traj_obs_dict["obs"]["wrist_camera"] = torch.cat(
                    (traj_obs_dict["obs"]["wrist_camera"], wrist_tensor), dim=0
                )
                traj_obs_dict["action"] = torch.cat(
                    (traj_obs_dict["action"], actions), dim=0
                )

        return traj_obs_dict


def plot_from_pickle(dir_path):
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(".pkl") and "mse" in file:
            files.append(file)
    files.sort()
    for file in files:
        with open(os.path.join(dir_path, file), "rb") as f:
            data = pickle.load(f)
            # files are named action_mse_{level}.pkl
            label = file.split("_")[-1].split(".")[0]
            if "level" in file:
                label = "Level " + label
            plt.plot(data, label=label)
    plt.xlabel("Time step")
    plt.ylabel("MSE")
    plt.title("MSE vs Time step")
    plt.legend()
    plt.savefig(f"{dir_path}/mse_all_levels.png")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[2].joinpath("config", "sim_config")),
)
def main(cfg: OmegaConf):
    # mse_per_level = []
    # legend = [
    #     "Gamepad",
    #     "Level 0",
    #     "Level 1",
    #     "Level 2",
    #     "Level 3",
    # ]
    # num_traj = 50

    com_y_shifts = [0.03, -0.03, -0.06]

    # # gamepad
    # zarr_path = f"/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/sim_sim_tee_data_carbon.zarr"
    # sim_sim_eval = AnalyzeActionError(cfg, zarr_path)
    # mse, std = sim_sim_eval.run(num_traj, f"eval/action_mse/action_mse_gamepad.png")
    # mse_per_level.append(mse)
    # with open("eval/action_mse/action_mse_gamepad.pkl", "wb") as f:
    #     pickle.dump(mse, f)
    # with open("eval/action_mse/action_std_gamepad.pkl", "wb") as f:
    #     pickle.dump(std, f)

    # # level 0
    # zarr_path = f"/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/sim_tee_data_large.zarr"
    # sim_sim_eval = AnalyzeActionError(cfg, zarr_path)
    # mse, std = sim_sim_eval.run(num_traj, f"eval/action_mse/action_mse_level_0.png")
    # mse_per_level.append(mse)
    # with open("eval/action_mse/action_mse_level_0.pkl", "wb") as f:
    #     pickle.dump(mse, f)
    # with open("eval/action_mse/action_std_level_0.pkl", "wb") as f:
    #     pickle.dump(std, f)

    # for level in range(1, 4):
    #     zarr_path = f"/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/physics_shift/physics_shift_level_{level}.zarr"
    #     sim_sim_eval = AnalyzeActionError(cfg, zarr_path)
    #     mse, std = sim_sim_eval.run(num_traj, f"eval/action_mse/action_mse_level_{level}.png")
    #     mse_per_level.append(mse)
    #     with open(f"eval/action_mse/action_mse_level_{level}.pkl", "wb") as f:
    #         pickle.dump(mse, f)
    #     with open(f"eval/action_mse/action_std_level_{level}.pkl", "wb") as f:
    #         pickle.dump(std, f)

    # # Plot all 4 MSE as line plots with points
    # for level, mse in enumerate(mse_per_level):
    #     plt.plot(range(8), mse, label=legend[level])
    # plt.xlabel("Time step")
    # plt.ylabel("MSE")
    # plt.title("MSE vs Time step")
    # plt.legend()
    # plt.savefig("action_mse.png")

    """
    Running into some errors with above (can't run script for too long?)
    """

    num_traj = 50
    zarr_path = f"/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/sim_sim_tee_data_carbon.zarr"
    sim_sim_eval = AnalyzeActionError(cfg, zarr_path)
    mse, std = sim_sim_eval.run(num_traj, f"eval/action_mse/action_mse_gamepad.png")
    with open("eval/action_mse/action_mse_gamepad.pkl", "wb") as f:
        pickle.dump(mse, f)
    with open("eval/action_mse/action_std_gamepad.pkl", "wb") as f:
        pickle.dump(std, f)

    # num_traj = 50
    # zarr_path = f"/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/sim_tee_data_large.zarr"
    # sim_sim_eval = AnalyzeActionError(cfg, zarr_path)
    # mse, std = sim_sim_eval.run(num_traj, f"eval/action_mse/action_mse_level_0.png")
    # with open("eval/action_mse/action_mse_level_0.pkl", "wb") as f:
    #     pickle.dump(mse, f)
    # with open("eval/action_mse/action_std_level_0.pkl", "wb") as f:
    #     pickle.dump(std, f)

    # num_traj = 50
    # level=1
    # zarr_path = f"/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/physics_shift/physics_shift_level_{level}.zarr"
    # # need to shift com for physics shift rendering
    # # cfg.physical_properties.center_of_mass = com[level-1]
    # sim_sim_eval = AnalyzeActionError(cfg, zarr_path, com_y_shifts[level-1])
    # mse, std = sim_sim_eval.run(num_traj, f"eval/action_mse/action_mse_level_{level}.png")
    # with open(f"eval/action_mse/action_mse_level_{level}.pkl", "wb") as f:
    #     pickle.dump(mse, f)
    # with open(f"eval/action_mse/action_std_level_{level}.pkl", "wb") as f:
    #     pickle.dump(std, f)

    plot_from_pickle("eval/action_mse")


if __name__ == "__main__":
    """
    Configure sim config through hydra yaml file
    Ex: python scripts/planar_pushing/run_sim_sim_eval.py --config-dir <dir> --config-name <file>
    """
    main()
