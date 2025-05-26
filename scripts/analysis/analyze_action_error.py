import os
import pathlib
import random
import shutil

import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.planar_pushing_dataset import PlanarPushingDataset
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import DataLoader


class AnalyzeActionError:
    def __init__(
        self,
        checkpoint,
        zarr_path,
        experiment_name,
        action_horizon=8,
        num_traj=None,
        seed: int = 42,
    ):
        # set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.checkpoint = pathlib.Path(checkpoint)
        self.zarr_path = zarr_path
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_horizon = action_horizon
        self.num_traj = num_traj

        # Parse plot path
        self.experiment_name = experiment_name
        self.output_dir = "eval/action_error"

        self._load_policy_from_checkpoint(self.checkpoint)

        # Create dataset (hardcoded)
        shape_meta = {
            "action": {"shape": [2]},
            "obs": {
                "agent_pos": {"shape": [3], "type": "low_dim"},
                "overhead_camera": {"shape": [3, 128, 128], "type": "rgb"},
                "wrist_camera": {"shape": [3, 128, 128], "type": "rgb"},
            },
        }
        zarr_configs = [
            {
                "path": zarr_path,
                "sampling_rate": 1,
                "val_ratio": 0,
                "max_train_episodes": num_traj,
            }
        ]
        self.dataset = PlanarPushingDataset(
            shape_meta=shape_meta,
            horizon=16,
            n_obs_steps=2,
            pad_after=7,
            pad_before=1,
            seed=42,
            zarr_configs=zarr_configs,
        )

    def _load_policy_from_checkpoint(self, checkpoint: str):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        self._cfg = payload["cfg"]

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

    def _load_normalizer(self):
        normalizer_path = self.checkpoint.parent.parent.joinpath("normalizer.pt")
        if os.path.exists(normalizer_path):
            return torch.load(normalizer_path)

    def run(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=64,
            num_workers=4,
            persistent_workers=False,
            pin_memory=True,
            shuffle=False,
        )

        action_errors = torch.zeros(0, self.action_horizon)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
                batch = dict_apply(
                    batch, lambda x: x.to(self._device, non_blocking=True)
                )
                predicted_actions = (
                    self._policy.predict_action(batch, use_DDIM=True)["action"]
                    .detach()
                    .cpu()
                )
                gt_actions = batch["action"][:, :8].cpu()

                # Compute errors
                action_error = torch.norm(predicted_actions - gt_actions, dim=2)

                # Reject outliers
                outlier_threshold = 0.01  # 1cm
                initial_len = action_error.shape[0]
                action_error = action_error[action_error[:, 0] <= outlier_threshold]
                action_errors = torch.cat([action_errors, action_error], dim=0)

                del batch
                del predicted_actions
                del gt_actions
                del action_error
                torch.cuda.empty_cache()

        # Save errors
        action_errors = action_errors.detach().cpu().numpy()
        output_dir = pathlib.Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            output_dir.joinpath(f"action_errors_{self.experiment_name}.npy"),
            action_errors,
        )

        # Plot errors
        self.plot_errors(action_errors)

        return action_errors

    def plot_errors(self, action_errors):
        # Create plots for errors
        mean_pos_errors = action_errors.mean(axis=0)
        std_pos_errors = action_errors.std(axis=0)

        # Create and save plot
        plt.figure()
        plt.errorbar(
            range(self.action_horizon), mean_pos_errors, yerr=std_pos_errors, fmt="o"
        )
        # label each point with its value
        for i, txt in enumerate(mean_pos_errors):
            plt.annotate(f"{txt:.6f}", (i, mean_pos_errors[i]))
        plt.xlabel("Timestep")
        plt.ylabel("Mean Action Error [m]")
        plt.title(f"Mean Action Error Over Timesteps")
        plt.savefig(f"eval/action_error/action_error_{self.experiment_name}.png")


def plot_all_errors(experiment_names):
    # Load and plot errors
    plt.figure()
    for experiment_name in experiment_names:
        action_errors = np.load(
            f"eval/action_error/action_errors_{experiment_name}.npy"
        )
        mean_pos_errors = action_errors.mean(axis=0)
        plt.plot(range(len(mean_pos_errors)), mean_pos_errors, label=experiment_name)
    plt.xlabel("Timestep")
    plt.ylabel("Mean Action Error [m]")
    plt.title(f"Mean Action Error Over Timesteps")
    plt.legend()
    plt.savefig(f"eval/action_error/action_error_all.png")


if __name__ == "__main__":
    experiments = {
        "gamepad": "/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/sim_sim_tee_data_carbon.zarr",
        "level_0": "/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/sim_tee_data_carbon.zarr",
        "level_1": "/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/physics_shift/physics_shift_level_1_no_visual_gap.zarr",
        "level_2": "/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/physics_shift/physics_shift_level_2_no_visual_gap.zarr",
        "level_3": "/home/adam/workspace/gcs-diffusion/data/planar_pushing_cotrain/physics_shift/physics_shift_level_3_no_visual_gap.zarr",
    }
    checkpoint = "/home/adam/workspace/gcs-diffusion/data/outputs/sim_sim/baseline_carbon/150/checkpoints/latest.ckpt"
    for experiment, zarr_path in experiments.items():
        analyze = AnalyzeActionError(checkpoint, zarr_path, experiment, num_traj=None)
        action_errors = analyze.run()

    plot_all_errors(experiments.keys())
