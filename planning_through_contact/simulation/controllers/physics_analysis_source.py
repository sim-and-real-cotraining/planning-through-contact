import numpy as np
from pydrake.all import DiagramBuilder, PiecewisePolynomial
from pydrake.systems.framework import Context, LeafSystem

from planning_through_contact.simulation.controllers.desired_planar_position_source_base import (
    DesiredPlanarPositionSourceBase,
)


class PhysicsAnalysisController(LeafSystem):
    def __init__(
        self,
        traj: np.ndarray,
        delay: float,
        action_dt: float = 0.1,
    ):
        super().__init__()
        self.traj = traj
        self.delay = delay

        # Create trajectory piecewise polynomial
        self.traj_polynomial = PiecewisePolynomial.FirstOrderHold(
            breaks=np.arange(len(traj)) * action_dt,
            samples=traj[:, :2].T,
        )

        # Declare output ports
        self.DeclareVectorOutputPort(
            "planar_position_command",
            2,
            self.DoCalcPlanarPositionCommand,
        )

    def _get_rel_t(self, t: float) -> float:
        return t - self.delay

    def DoCalcPlanarPositionCommand(self, context: Context, output):
        t = context.get_time()
        rel_t = self._get_rel_t(t)
        command = self.traj_polynomial.value(rel_t)
        output.SetFromVector(command)


class PhysicsAnalysisSource(DesiredPlanarPositionSourceBase):
    def __init__(
        self,
        traj: np.ndarray,
        delay: float,
        action_dt: float = 0.1,
    ):
        super().__init__()
        self._traj = traj
        self._delay = delay

        # Add systems
        self._builder = builder = DiagramBuilder()
        self._controller = builder.AddSystem(
            PhysicsAnalysisController(traj, delay, action_dt)
        )

        # Export ports
        builder.ExportOutput(
            self._controller.get_output_port(0), "planar_position_command"
        )
        builder.BuildInto(self)
