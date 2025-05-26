import logging
import os
import pathlib
import pickle
import time as pytime
from collections import deque
from typing import List, Optional, Tuple

import cv2
import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf

# Diffusion Policy imports
import torch
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# Pydrake imports
from pydrake.common.value import AbstractValue, Value
from pydrake.math import RigidTransform
from pydrake.systems.framework import Context, LeafSystem
from pydrake.systems.sensors import Image, PixelType

from planning_through_contact.geometry.planar.planar_pose import PlanarPose

logger = logging.getLogger(__name__)

# Set the print precision to 4 decimal places
np.set_printoptions(precision=4)


class DiffusionPolicyController(LeafSystem):
    def __init__(
        self,
        checkpoint: str,
        initial_pusher_pose: PlanarPose,
        target_slider_pose: PlanarPose,
        diffusion_policy_path: str = "/home/adam/workspace/gcs-diffusion",
        freq: float = 10.0,
        delay: float = 1.0,
        device="cuda:0",
        debug: bool = False,
        save_logs: bool = False,
        cfg_overrides: dict = {},
    ):
        super().__init__()
        self._checkpoint = pathlib.Path(checkpoint)
        self._diffusion_policy_path = pathlib.Path(diffusion_policy_path)
        self._initial_pusher_pose = initial_pusher_pose
        self._target_slider_pose = target_slider_pose
        self._freq = freq
        self._dt = 1.0 / freq
        self._delay = delay
        self._debug = debug
        self._device = torch.device(device)
        self._cfg_overrides = cfg_overrides
        self._load_policy_from_checkpoint(self._checkpoint)

        # get parameters
        self._obs_horizon = self._cfg.n_obs_steps
        self._action_steps = self._cfg.n_action_steps
        self._state_dim = self._cfg.shape_meta.obs.agent_pos.shape[0]
        self._action_dim = self._cfg.shape_meta.action.shape[0]
        self._target_dim = self._cfg.policy.target_dim
        self._B = 1  # batch size is 1

        # indexing parameters for action predictions
        # Note: this used to be self._state = self._obs_horizon - 1
        self._start = self._obs_horizon
        self._end = self._start + self._action_steps

        # variables for DoCalcOutput
        self._actions = deque([], maxlen=self._action_steps)
        self._current_action = np.array(
            [
                self._initial_pusher_pose.x,
                self._initial_pusher_pose.y,
            ]
        )

        # Input port for pusher pose
        self.pusher_pose_measured = self.DeclareAbstractInputPort(
            "pusher_pose_measured",
            AbstractValue.Make(RigidTransform()),
        )

        # Camera input ports
        self.camera_port_dict = {}
        self._camera_shape_dict = {}
        for key, value in self._cfg.policy.shape_meta.obs.items():
            if value["type"] == "rgb":
                shape = value["shape"]
                self.camera_port_dict[key] = self.DeclareAbstractInputPort(
                    key,
                    Value[Image[PixelType.kRgba8U]].Make(
                        Image[PixelType.kRgba8U](shape[1], shape[2])  # H, W
                    ),
                )
                self._camera_shape_dict[key] = shape  # Shape is C, H, W

        self.output = self.DeclareVectorOutputPort(
            "planar_position_command", 2, self.DoCalcOutput
        )

        # observation histories
        self._pusher_pose_deque = deque(
            [self._initial_pusher_pose.vector() for _ in range(self._obs_horizon)],
            maxlen=self._obs_horizon,
        )
        self._image_deque_dict = {
            name: deque([], maxlen=self._obs_horizon)
            for name in self.camera_port_dict.keys()
        }

        # Reset
        self._last_reset_time = 0.0
        self._received_reset_signal = True

        # Logging data structures
        self._save_logs = save_logs
        if self._save_logs:
            self._logs = {
                "checkpoint": self._checkpoint,
                "actions": [],  # N x action horizon x action dim
                "poses": [],  # N x observation horizon x pose dim
                "images": {
                    name: [] for name in self.camera_port_dict.keys()
                },  # N x observation horizon x H x W x C
                "embeddings": [],  # N x embedding dim
            }

    def _load_policy_from_checkpoint(self, checkpoint: str):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        self._cfg = payload["cfg"]

        # Override diffusion policy config
        for key, value in self._cfg_overrides.items():
            if isinstance(value, omegaconf.dictconfig.DictConfig):
                for k, v in value.items():
                    if key in self._cfg:
                        self._cfg[key][k] = v
            elif key in self._cfg:
                self._cfg[key] = value

        # update pretrained path if it exists
        if "pretrained_checkpoint" in self._cfg and self._cfg["pretrained_checkpoint"]:
            if not os.path.isabs(self._cfg["pretrained_checkpoint"]):
                self._cfg[
                    "pretrained_checkpoint"
                ] = self._diffusion_policy_path.joinpath(
                    self._cfg["pretrained_checkpoint"]
                )

        # self._cfg.training.device = self._device
        cls = hydra.utils.get_class(self._cfg._target_)
        workspace: BaseWorkspace
        workspace = cls(self._cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        self._normalizer = self._load_normalizer()

        # get policy from workspace
        self._policy = workspace.model
        self._policy.set_normalizer(self._normalizer)
        if self._cfg.training.use_ema:
            self._policy = workspace.ema_model
            self._policy.set_normalizer(self._normalizer)
        self._policy.to(self._device)
        self._policy.eval()

    def DoCalcOutput(self, context: Context, output):
        time = context.get_time()
        if self._received_reset_signal:
            self._last_reset_time = time
            self._received_reset_signal = False

        # Continually update ports until delay is over
        if time < self._last_reset_time + self._delay:
            self._update_history(context)
            output.set_value(self._current_action)
            return
        # Accumulate new observations after reset
        if len(self._pusher_pose_deque) < self._obs_horizon:
            self._update_history(context)
            output.set_value(self._current_action)
            return

        # Update observation history
        self._update_history(context)

        obs_dict = self._deque_to_dict(
            self._pusher_pose_deque,
            self._image_deque_dict,
            self._target_slider_pose.vector(),
        )

        if len(self._actions) == 0:
            # Compute new actions
            start_time = pytime.time()
            with torch.no_grad():
                action_prediction = self._policy.predict_action(
                    obs_dict, use_DDIM=True
                )["action_pred"][0]

                # Save logs
                if self._save_logs:
                    self._logs["actions"].append(action_prediction.cpu().numpy())
                    self._logs["poses"].append(
                        np.array([pose for pose in self._pusher_pose_deque])
                    )
                    for camera, image_deque in self._image_deque_dict.items():
                        self._logs["images"][camera].append(
                            np.array([img for img in image_deque])
                        )
                    self._logs["embeddings"].append(
                        self._policy.compute_obs_embedding(obs_dict)
                        .cpu()
                        .numpy()
                        .flatten()
                    )

            actions = action_prediction[self._start : self._end]
            for action in actions:
                self._actions.append(action.cpu().numpy())

            if self._debug:
                print(
                    f"[TIME: {time:.3f}] Computed new actions in {pytime.time() - start_time:.3f}s\n"
                )
                print("Observations:")
                for state in self._pusher_pose_deque:
                    print(state)
                for image_deque in self._image_deque_dict.values():
                    for img in image_deque:
                        plt.imshow(img)
                        plt.show()
                print("\nAction Predictions:")
                print(action_prediction)
                print("\nActions")
                print(actions)

        # get next action
        assert len(self._actions) > 0
        prev_action = self._current_action
        self._current_action = self._actions.popleft()
        output.set_value(self._current_action)

        # debug print statements
        if self._debug:
            print(
                f"Time: {time:.3f}, action delta: {np.linalg.norm(self._current_action - prev_action)}"
            )
            print(f"Time: {time:.3f}, action: {self._current_action}")

    def reset(self, reset_position: np.ndarray = None):
        if reset_position is not None:
            self._current_action = reset_position
        self._actions.clear()
        self._pusher_pose_deque.clear()
        for image_deque in self._image_deque_dict.values():
            image_deque.clear()
        self._received_reset_signal = True

    def _deque_to_dict(
        self, obs_deque: deque, image_deque_dict: dict, target: np.ndarray
    ):
        state_tensor = torch.cat(
            [torch.from_numpy(obs) for obs in obs_deque], dim=0
        ).reshape(self._B, self._obs_horizon, self._state_dim)
        target_tensor = torch.from_numpy(target).reshape(1, self._target_dim)  # 1, D_t

        data = {
            "obs": {
                "agent_pos": state_tensor.to(self._device),  # 1, T_obs, D_x
            },
            "target": target_tensor.to(self._device),  # 1, D_t
        }

        # Note: Assuming sim is the first element in one hot encoding
        if (
            hasattr(self._policy, "one_hot_encoding_dim")
            and self._policy.one_hot_encoding_dim > 0
        ):
            data["one_hot_encoding"] = torch.zeros(
                1, self._policy.one_hot_encoding_dim
            ).to(self._device)
            data["one_hot_encoding"][0, 0] = 1

        # Load images into data dict
        for camera, image_deque in image_deque_dict.items():
            img_tensor = torch.cat(
                [
                    torch.from_numpy(np.moveaxis(img, -1, -3) / 255.0)  # C H W
                    for img in image_deque
                ],
                dim=0,
            ).reshape(
                self._B,
                self._obs_horizon,
                self._camera_shape_dict[camera][0],  # C
                self._camera_shape_dict[camera][1],  # H
                self._camera_shape_dict[camera][2],  # W
            )
            data["obs"][camera] = img_tensor.to(self._device)  # 1, T_obs, C, H, W

        return data

    def _update_history(self, context):
        """Update state and image observation history"""

        # Update end effector deque
        pusher_pose: RigidTransform = self.pusher_pose_measured.Eval(context)  # type: ignore
        pusher_planer_pose = PlanarPose.from_pose(pusher_pose).vector()
        self._pusher_pose_deque.append(pusher_planer_pose)

        # Update image deques
        for camera, port in self.camera_port_dict.items():
            image = port.Eval(context).data
            image_height = self._camera_shape_dict[camera][1]
            image_width = self._camera_shape_dict[camera][2]
            if image.shape[0] != image_height or image.shape[1] != image_width:
                image = cv2.resize(image, (image_width, image_height))
            self._image_deque_dict[camera].append(image[:, :, :-1])  # H W C

    def _load_normalizer(self):
        normalizer_path = self._checkpoint.parent.parent.joinpath("normalizer.pt")
        if os.path.exists(normalizer_path):
            return torch.load(normalizer_path)
        else:
            # get normalizer: this might be expensive for larger datasets
            LEGACY_COTRAIN_DATASET = "diffusion_policy.dataset.drake_cotrain_planar_pushing_hybrid_dataset.DrakeCotrainPlanarPushingHybridDataset"
            PLANAR_PUSHING_DATASET = (
                "diffusion_policy.dataset.planar_pushing_dataset.PlanarPushingDataset"
            )
            cotraining_datasets = [LEGACY_COTRAIN_DATASET, PLANAR_PUSHING_DATASET]
            # Fix config paths for datasets
            if self._cfg.task.dataset._target_ in cotraining_datasets:
                zarr_configs = self._cfg.task.dataset.zarr_configs
                for config in zarr_configs:
                    config["path"] = self._diffusion_policy_path.joinpath(
                        config["path"]
                    )
            else:
                self._cfg.task.dataset.zarr_path = self._diffusion_policy_path.joinpath(
                    self._cfg.task.dataset.zarr_path
                )

            # Extract and save normalizer
            dataset: BaseImageDataset = hydra.utils.instantiate(self._cfg.task.dataset)
            normalizer = dataset.get_normalizer()
            torch.save(normalizer, normalizer_path)
            return normalizer

    def save_logs_to_file(self, save_path: str):
        if self._save_logs:
            # Convert logs to numpy
            self._logs["actions"] = np.array(self._logs["actions"])
            self._logs["poses"] = np.array(self._logs["poses"])
            self._logs["embeddings"] = np.array(self._logs["embeddings"])
            for camera in self._logs["images"].keys():
                self._logs["images"][camera] = np.array(self._logs["images"][camera])

            # Save logs to file
            with open(save_path, "wb") as f:
                pickle.dump(self._logs, f)
            logger.info(f"Saved logs to {save_path}")
        else:
            logger.warning("No logs to save. Set save_logs=True to save logs.")
