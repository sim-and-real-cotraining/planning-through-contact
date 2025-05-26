from dataclasses import dataclass, field

import numpy as np
from pydrake.all import DiagramBuilder, LeafSystem, ZeroOrderHold

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.controllers.desired_planar_position_source_base import (
    DesiredPlanarPositionSourceBase,
)
from planning_through_contact.simulation.planar_pushing.diffusion_policy_controller import (
    DiffusionPolicyController,
)


@dataclass
class DiffusionPolicyConfig:
    checkpoint: str
    initial_pusher_pose: PlanarPose
    target_slider_pose: PlanarPose
    diffusion_policy_path: str = "/home/adam/workspace/gcs-diffusion"
    freq: float = 10.0
    delay: float = 1.0
    debug: bool = False
    device: str = "cuda:0"
    cfg_overrides: dict = field(default_factory={})
    save_logs: bool = False

    def __eq__(self, other: "DiffusionPolicyConfig"):
        return (
            self.checkpoint == other.checkpoint
            and self.initial_pusher_pose == other.initial_pusher_pose
            and self.target_slider_pose == other.target_slider_pose
            and self.diffusion_policy_path == other.diffusion_policy_path
            and self.freq == other.freq
            and self.delay == other.delay
            and self.debug == other.debug
            and self.device == other.device
            and self.cfg_overrides == other.cfg_overrides
            and self.save_logs == other.save_logs
        )


class DiffusionPolicySource(DesiredPlanarPositionSourceBase):
    """Uses the desired trajectory of the entire system and diffusion controller
    to generate desired positions for the robot."""

    def __init__(self, diffusion_policy_config: DiffusionPolicyConfig):
        super().__init__()

        builder = DiagramBuilder()

        ## Add Leaf systems

        # Diffusion Policy Controller
        freq = diffusion_policy_config.freq
        self._diffusion_policy_controller = builder.AddNamedSystem(
            "DiffusionPolicyController",
            DiffusionPolicyController(
                checkpoint=diffusion_policy_config.checkpoint,
                diffusion_policy_path=diffusion_policy_config.diffusion_policy_path,
                initial_pusher_pose=diffusion_policy_config.initial_pusher_pose,
                target_slider_pose=diffusion_policy_config.target_slider_pose,
                freq=diffusion_policy_config.freq,
                delay=diffusion_policy_config.delay,
                debug=diffusion_policy_config.debug,
                device=diffusion_policy_config.device,
                cfg_overrides=diffusion_policy_config.cfg_overrides,
                save_logs=diffusion_policy_config.save_logs,
            ),
        )

        # Zero Order Hold
        self._zero_order_hold = builder.AddNamedSystem(
            "ZeroOrderHold",
            ZeroOrderHold(
                period_sec=1 / freq,
                vector_size=2,
            ),
        )

        # AppendZeros (add theta to x y positions)
        self._append_zeros = builder.AddSystem(AppendZeros(input_size=2, num_zeros=1))

        ## Internal connections

        builder.Connect(
            self._diffusion_policy_controller.get_output_port(),
            self._zero_order_hold.get_input_port(),
        )

        builder.Connect(
            self._zero_order_hold.get_output_port(), self._append_zeros.get_input_port()
        )

        ## Export inputs and outputs (external)

        builder.ExportInput(
            self._diffusion_policy_controller.GetInputPort("pusher_pose_measured"),
            "pusher_pose_measured",
        )

        for camera in self._diffusion_policy_controller.camera_port_dict.keys():
            builder.ExportInput(
                self._diffusion_policy_controller.GetInputPort(camera), camera
            )

        builder.ExportOutput(
            self._zero_order_hold.get_output_port(), "planar_position_command"
        )

        builder.ExportOutput(
            self._append_zeros.get_output_port(), "planar_pose_command"
        )

        builder.BuildInto(self)

    def reset(self, pusher_position: np.ndarray = None):
        self._diffusion_policy_controller.reset(pusher_position)


class AppendZeros(LeafSystem):
    def __init__(self, input_size: int, num_zeros: int):
        super().__init__()
        self._input_size = input_size
        self._num_zeros = num_zeros
        self.DeclareVectorInputPort("input", input_size)
        self.DeclareVectorOutputPort("output", input_size + num_zeros, self.CalcOutput)

    def CalcOutput(self, context, output):
        input = self.EvalVectorInput(context, 0).get_value()
        output_vec = np.zeros(self._input_size + self._num_zeros)
        output_vec[: self._input_size] = input
        output.SetFromVector(output_vec)
