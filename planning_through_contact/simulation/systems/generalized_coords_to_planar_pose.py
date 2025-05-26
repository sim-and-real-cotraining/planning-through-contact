from pydrake.systems.framework import LeafSystem

from planning_through_contact.geometry.planar.planar_pose import PlanarPose


class GeneralizedCoordsToPlanarPose(LeafSystem):
    """
    Converts Planar Pose ([x, y, theta]) to generalized coords
    """

    def __init__(self):
        super().__init__()
        self.DeclareVectorInputPort("generalized_coords_input", 7)
        self.DeclareVectorOutputPort("planar_pose_output", 3, self.DoCalcOutput)

    def DoCalcOutput(self, context, output):
        generalized_coords = self.EvalVectorInput(context, 0).get_value()
        planar_pose = PlanarPose.from_generalized_coords(generalized_coords)
        output.set_value(planar_pose.vector())
