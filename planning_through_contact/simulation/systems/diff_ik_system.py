import numpy as np
from pydrake.all import (
    AbstractValue,
    DifferentialInverseKinematicsParameters,
    DifferentialInverseKinematicsStatus,
    DoDifferentialInverseKinematics,
    InverseKinematics,
    MultibodyPlant,
    RigidTransform,
    RotationMatrix,
    Solve,
)
from pydrake.systems.framework import LeafSystem


class DiffIKSystem(LeafSystem):
    """Solves inverse kinematics"""

    def __init__(
        self,
        plant: MultibodyPlant,
        time_step: float,
        default_joint_positions: np.ndarray = None,
        log_path: str = None,
    ):
        super().__init__()

        self._plant = plant
        self._plant_context = self._plant.CreateDefaultContext()
        self._time_step = time_step
        self._default_joint_positions = default_joint_positions
        self._log_path = log_path
        self._paramters = self._get_diff_ik_params()
        self._pusher_frame = self._plant.GetFrameByName("pusher_end")
        self._consequtive_ik_fails = 0
        self._max_consequtive_ik_fails = 0

        # Declare I/O ports
        self.DeclareAbstractInputPort(
            "rigid_transform_input", AbstractValue.Make(RigidTransform())
        )
        self.DeclareVectorInputPort(
            "state", plant.num_positions() + plant.num_velocities()
        )

        self.DeclareVectorOutputPort(
            "q", plant.num_positions(), self.DoCalcVectorOutput
        )

    def DoCalcVectorOutput(self, context, output):
        # Read input ports
        rigid_transform = self.EvalAbstractInput(context, 0).get_value()
        state = self.EvalVectorInput(context, 1).get_mutable_value()
        if np.allclose(state, np.zeros_like(state)):
            state[: self._plant.num_positions()] = self._default_joint_positions

        # Solve diff IK
        diff_ik_result = self._solve_diff_ik(state, rigid_transform)
        if diff_ik_result.status == DifferentialInverseKinematicsStatus.kSolutionFound:
            v = diff_ik_result.joint_velocities
            q = state[: self._plant.num_positions()] + v * self._time_step
            self._consequtive_ik_fails = 0
            output.SetFromVector(q)
            return

        # Diff IK failed: try optimization-based IK
        ik_result = self._attempt_to_solve_ik(
            rigid_transform, state[: self._plant.num_positions()]
        )
        if ik_result.is_success():
            output.SetFromVector(ik_result.GetSolution())
            self._consequtive_ik_fails = 0
            return

        # Diff IK and optimization-based IK failed
        self._consequtive_ik_fails += 1
        if self._consequtive_ik_fails > self._max_consequtive_ik_fails:
            self._max_consequtive_ik_fails = self._consequtive_ik_fails
            if self._log_path is not None:
                with open(self._log_path, "w") as f:
                    f.write(
                        f"Max consequtive IK fails: {self._max_consequtive_ik_fails}"
                    )
        output.SetFromVector(state[: self._plant.num_positions()])

    def _get_diff_ik_params(self):
        # Initialize parameters
        param = DifferentialInverseKinematicsParameters(
            num_positions=self._plant.num_positions(),
            num_velocities=self._plant.num_velocities(),
        )

        # Set parameters
        param.set_time_step(self._time_step)
        if self._default_joint_positions is not None:
            param.set_nominal_joint_position(self._default_joint_positions)
        param.set_joint_position_limits(
            (self._plant.GetPositionLowerLimits(), self._plant.GetPositionUpperLimits())
        )
        param.set_joint_velocity_limits(
            (self._plant.GetVelocityLowerLimits(), self._plant.GetVelocityUpperLimits())
        )
        # param.set_joint_acceleration_limits(
        #     (
        #         self._plant.GetAccelerationLowerLimits(),
        #         self._plant.GetAccelerationUpperLimits(),
        #     )
        # )
        param.set_maximum_scaling_to_report_stuck(1e-5)

        return param

    def _attempt_to_solve_ik(
        self,
        pose: RigidTransform,
        prev_q: np.ndarray,
        disregard_angle: bool = False,
    ):
        # ik solve with default parameters
        ik_result = self._solve_ik(pose, prev_q, disregard_angle, eps=1e-3)
        if ik_result.is_success():
            return ik_result

        # # increase eps
        # ik_result = self._solve_ik(pose, prev_q, disregard_angle, eps=1e-2)
        # if ik_result.is_success():
        #     return ik_result

        # all ik attempts failed
        return ik_result

    def _solve_diff_ik(self, state, rigid_transform):
        self._plant.SetPositionsAndVelocities(self._plant_context, state)
        return DoDifferentialInverseKinematics(
            self._plant,
            self._plant_context,
            rigid_transform,
            self._pusher_frame,
            self._paramters,
        )

    def _solve_ik(
        self,
        pose: RigidTransform,
        prev_q: np.ndarray,
        disregard_angle: bool = False,
        eps=1e-3,
    ):
        # Plant needs to be just the robot without other objects
        # Need to create a new context that the IK can use for solving the problem

        ik = InverseKinematics(self._plant, with_joint_limits=True)  # type: ignore

        ik.AddPositionConstraint(
            self._pusher_frame,
            np.zeros(3),
            self._plant.world_frame(),
            pose.translation() - np.ones(3) * eps,
            pose.translation() + np.ones(3) * eps,
        )

        if disregard_angle:
            z_unit_vec = np.array([0, 0, 1])
            ik.AddAngleBetweenVectorsConstraint(
                self._pusher_frame,
                z_unit_vec,
                self._plant.world_frame(),
                -z_unit_vec,  # The pusher object has z-axis pointing up
                0 - eps,
                0 + eps,
            )
        else:
            ik.AddOrientationConstraint(
                self._pusher_frame,
                RotationMatrix(),
                self._plant.world_frame(),
                pose.rotation(),
                eps,
            )

        # Cost on deviation from default joint positions
        prog = ik.get_mutable_prog()
        q = ik.q()

        q0 = self._default_joint_positions
        # prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
        prog.AddQuadraticErrorCost(1000000 * np.identity(len(q)), prev_q, q)
        prog.SetInitialGuess(q, prev_q)

        return Solve(ik.prog())
