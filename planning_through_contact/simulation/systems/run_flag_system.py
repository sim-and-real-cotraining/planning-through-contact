from pydrake.all import AbstractValue, InputPortIndex
from pydrake.systems.framework import LeafSystem

from planning_through_contact.geometry.planar.planar_pose import PlanarPose


class RunFlagSystem(LeafSystem):
    """
    Converts InputPort to boolean output
    """

    def __init__(self, true_port_index: int):
        super().__init__()
        self.true_port_index = true_port_index
        self.DeclareAbstractInputPort(
            "port_select", AbstractValue.Make(InputPortIndex(0))
        )
        self.DeclareVectorOutputPort("run_flag", 1, self.DoCalcOutput)

    def DoCalcOutput(self, context, output):
        port_select = self.EvalAbstractInput(context, 0).get_value()
        if port_select == InputPortIndex(self.true_port_index):
            output.set_value([1.0])
        else:
            output.set_value([0.0])
