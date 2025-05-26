import numpy as np
from pydrake.all import StartMeshcat

# Pydrake imports
from pydrake.common.value import AbstractValue, Value
from pydrake.math import RigidTransform
from pydrake.systems.framework import Context, LeafSystem

from planning_through_contact.geometry.planar.planar_pose import PlanarPose

# Set the print precision to 4 decimal places
np.set_printoptions(precision=4)


def print_blue(text):
    print(f"\033[94m{text}\033[0m")


class GamepadController(LeafSystem):
    def __init__(
        self,
        meshcat,
        translation_scale: float,
        deadzone: float,
        gamepad_orientation: np.ndarray,
    ):
        super().__init__()

        self.translation_scale = translation_scale
        self.deadzone = deadzone
        self.gamepad_orientation = gamepad_orientation

        self.init_xy = None

        self.button_index = {
            0: "A",
            1: "B",
            2: "X",
            3: "Y",
            4: "LB",
            5: "RB",
            6: "LT",
            7: "RT",
            8: "BACK",
            9: "START",
            10: "LS",
            11: "RS",
            12: "UP",
            13: "DOWN",
            14: "LEFT",
            15: "RIGHT",
            16: "LOGO",
        }

        # Set up ports
        self.pusher_pose_measured = self.DeclareAbstractInputPort(
            "pusher_pose_measured",
            AbstractValue.Make(RigidTransform()),
        )
        self.run_flag_port = self.DeclareVectorInputPort("run_flag", 1)

        self.output = self.DeclareVectorOutputPort(
            "planar_position_command", 2, self.DoCalcOutput
        )

        # Wait for gamepad connection
        self.meshcat = meshcat
        print_blue("\nPlease connect gamepad.")
        print_blue("1. Open meshcat (default: http://localhost:7000)")
        print_blue("2. Press any button on the gamepad.")

        while self.meshcat.GetGamepad().index is None:
            continue
        print_blue("\nGamepad connected!\n")

    def DoCalcOutput(self, context: Context, output):
        # Read in pose
        pusher_pose: RigidTransform = self.pusher_pose_measured.Eval(context)  # type: ignore
        run_flag = round(self.run_flag_port.Eval(context)[0])
        curr_xy = PlanarPose.from_pose(pusher_pose).pos().reshape(2)
        xy_offset = self.get_xy_offset()
        if self.init_xy is None and run_flag == 1:
            self.init_xy = curr_xy
        elif self.init_xy is None:
            output.SetFromVector([0.0, 0.0])
            return

        # Compute and set target pose
        target_xy = self.init_xy + xy_offset
        self.init_xy = target_xy
        output.SetFromVector(target_xy)

    def get_xy_offset(self):
        gamepad = self.meshcat.GetGamepad()
        position = self.create_stick_dead_zone(gamepad.axes[0], gamepad.axes[1])
        return self.translation_scale * self.gamepad_orientation @ position

    def create_stick_dead_zone(self, x, y):
        stick = np.array([x, y])
        m = np.linalg.norm(stick)

        if m < self.deadzone:
            return np.array([0, 0])
        over = (m - self.deadzone) / (1 - self.deadzone)
        return stick * over / m

    def get_button_values(self):
        gamepad = self.meshcat.GetGamepad().button_values
        return {self.button_index[i]: gamepad[i] for i in range(len(gamepad))}

    def reset(self, reset_xy=None):
        self.init_xy = reset_xy

    def set_translation_scale(self, translation_scale):
        self.translation_scale = translation_scale

    def get_translation_scale(self):
        return self.translation_scale
