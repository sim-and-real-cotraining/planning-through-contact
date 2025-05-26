import logging
import os
import pathlib
from typing import Optional

import numpy as np
from pydrake.all import (
    Cylinder,
    Demultiplexer,
    DiagramBuilder,
    GeometryInstance,
    ImageWriter,
    LogVectorOutput,
    MakePhongIllustrationProperties,
    Meshcat,
    PixelType,
    Rgba,
    Simulator,
)

from planning_through_contact.experiments.utils import get_default_plan_config
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    PlanarPushingWorkspace,
)
from planning_through_contact.simulation.controllers.desired_planar_position_source_base import (
    DesiredPlanarPositionSourceBase,
)
from planning_through_contact.simulation.controllers.gamepad_controller_source import (
    GamepadControllerSource,
)
from planning_through_contact.simulation.controllers.robot_system_base import (
    RobotSystemBase,
)
from planning_through_contact.simulation.controllers.teleop_position_source import (
    TeleopPositionSource,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_utils import (
    check_collision,
    create_goal_geometries,
    get_slider_pose_within_workspace,
    get_slider_shapes,
    slider_within_workspace,
    visualize_desired_slider_pose,
)
from planning_through_contact.simulation.systems.generalized_coords_to_planar_pose import (
    GeneralizedCoordsToPlanarPose,
)
from planning_through_contact.simulation.systems.rigid_transform_to_planar_pose_vector_system import (
    RigidTransformToPlanarPoseVectorSystem,
)
from planning_through_contact.simulation.systems.robot_state_to_rigid_transform import (
    RobotStateToRigidTransform,
)
from planning_through_contact.visualize.colors import COLORS

logger = logging.getLogger(__name__)


class SimulatedRealTableEnvironment:
    def __init__(
        self,
        desired_position_source: DesiredPlanarPositionSourceBase,
        robot_system: RobotSystemBase,
        sim_config: PlanarPushingSimConfig,
        station_meshcat: Optional[Meshcat] = None,
        arbitrary_shape_pickle_path: Optional[str] = None,
    ):
        self._desired_position_source = desired_position_source
        self._robot_system = robot_system
        self._sim_config = sim_config
        self._meshcat = station_meshcat
        self._simulator = None
        self._goal_geometries = []
        self._pusher_goal_geometry = None

        self._plant = self._robot_system.get_station_plant()
        self._scene_graph = self._robot_system.get_scene_graph()
        self._slider = self._robot_system.get_slider()

        self._robot_model_instance = self._plant.GetModelInstanceByName(
            self._robot_system.robot_model_name
        )
        self._slider_model_instance = self._plant.GetModelInstanceByName(
            self._robot_system.slider_model_name
        )

        builder = DiagramBuilder()

        ## Add systems

        builder.AddNamedSystem(
            "DesiredPlanarPositionSource",
            self._desired_position_source,
        )

        builder.AddNamedSystem(
            "PositionController",
            self._robot_system,
        )

        self._robot_state_to_rigid_transform = builder.AddNamedSystem(
            "RobotStateToRigidTransform",
            RobotStateToRigidTransform(
                self._plant,
                self._robot_system.robot_model_name,
            ),
        )

        self._meshcat = self._robot_system.get_meshcat()

        ## Connect systems

        # Connect PositionController to RobotStateToOutputs
        builder.Connect(
            self._robot_system.GetOutputPort("robot_state_measured"),
            self._robot_state_to_rigid_transform.GetInputPort("state"),
        )

        # Inputs to desired position source
        if self._desired_position_source.HasInputPort("pusher_pose_measured"):
            builder.Connect(
                self._robot_state_to_rigid_transform.GetOutputPort("pose"),
                self._desired_position_source.GetInputPort("pusher_pose_measured"),
            )
        for camera_config in self._sim_config.camera_configs:
            if self._desired_position_source.HasInputPort(
                f"rgbd_sensor_{camera_config.name}"
            ):
                builder.Connect(
                    self._robot_system.GetOutputPort(
                        f"rgbd_sensor_{camera_config.name}"
                    ),
                    self._desired_position_source.GetInputPort(camera_config.name),
                )
            if self._desired_position_source.HasInputPort(camera_config.name):
                builder.Connect(
                    self._robot_system.GetOutputPort(
                        f"rgbd_sensor_{camera_config.name}"
                    ),
                    self._desired_position_source.GetInputPort(camera_config.name),
                )
        if self._desired_position_source.HasInputPort("run_flag"):
            builder.Connect(
                self._robot_system.GetOutputPort("run_flag"),
                self._desired_position_source.GetInputPort("run_flag"),
            )

        # Inputs to robot system
        assert self._desired_position_source.HasOutputPort("planar_position_command")
        builder.Connect(
            self._desired_position_source.GetOutputPort("planar_position_command"),
            self._robot_system.GetInputPort("planar_position_command"),
        )

        ## Add loggers

        # pusher logger
        pusher_pose_to_vector = builder.AddSystem(
            RigidTransformToPlanarPoseVectorSystem()
        )
        builder.Connect(
            self._robot_state_to_rigid_transform.GetOutputPort("pose"),
            pusher_pose_to_vector.get_input_port(),
        )
        self._pusher_pose_logger = LogVectorOutput(
            pusher_pose_to_vector.get_output_port(), builder, 0.01
        )

        # slider logger
        pose_selector = builder.AddSystem(Demultiplexer([7, 6]))
        slider_pose_to_planar_pose = builder.AddNamedSystem(
            "SliderPoseToPlanarPose",
            GeneralizedCoordsToPlanarPose(),
        )
        builder.Connect(
            self._robot_system.GetOutputPort("object_state_measured"),
            pose_selector.get_input_port(),
        )
        builder.Connect(
            pose_selector.get_output_port(0),
            slider_pose_to_planar_pose.get_input_port(),
        )
        self._slider_pose_logger = LogVectorOutput(
            slider_pose_to_planar_pose.get_output_port(), builder, 0.01
        )

        ## Image writers
        if isinstance(self._desired_position_source, GamepadControllerSource):
            # hardcoded path
            image_writer_dir = "trajectories_rendered/temp"
            image_writers = []
            for camera_config in sim_config.camera_configs:
                os.makedirs(f"{image_writer_dir}/{camera_config.name}", exist_ok=True)
                image_writers.append(ImageWriter())
                image_writers[-1].DeclareImageInputPort(
                    pixel_type=PixelType.kRgba8U,
                    port_name=f"{camera_config.name}_image",
                    file_name_format=f"{image_writer_dir}/{camera_config.name}"
                    + "/{time_msec}.png",
                    publish_period=0.1,
                    start_time=0.0,
                )
                builder.AddNamedSystem(
                    f"{camera_config}_image_writer", image_writers[-1]
                )
                builder.Connect(
                    self._robot_system.GetOutputPort(
                        f"rgbd_sensor_{camera_config.name}"
                    ),
                    image_writers[-1].get_input_port(),
                )

        diagram = builder.Build()
        self._diagram = diagram

        self._simulator = Simulator(diagram)
        if sim_config.use_realtime:
            self._simulator.set_target_realtime_rate(1.0)

        self.context = self._simulator.get_mutable_context()
        self._robot_system.pre_sim_callback(self.context)
        self.robot_system_context = self._robot_system.GetMyContextFromRoot(
            self.context
        )
        self.mbp_context = self._plant.GetMyContextFromRoot(self.context)
        # initialize slider above the table
        self.set_slider_planar_pose(PlanarPose(0.587, -0.0355, 0.0))

    def export_diagram(self, filename: str):
        import pydot

        pydot.graph_from_dot_data(self._diagram.GetGraphvizString())[0].write_pdf(  # type: ignore
            filename
        )
        print(f"Saved diagram to: {filename}")

    def set_slider_planar_pose(self, pose: PlanarPose):
        min_height = 0.02

        # add a small height to avoid the box penetrating the table
        q = pose.to_generalized_coords(min_height + 1e-2, z_axis_is_positive=True)
        self._plant.SetPositions(self.mbp_context, self._slider, q)
        self._plant.SetVelocities(self.mbp_context, self._slider, np.zeros(6))

    def set_robot_position(self, q: np.ndarray):
        # Set positions and velocities
        v = np.zeros(self._plant.num_velocities(self._robot_model_instance))
        self._plant.SetPositions(self.mbp_context, self._robot_model_instance, q)
        self._plant.SetVelocities(self.mbp_context, self._robot_model_instance, v)

        # Update diff ik integrator to match new robot position
        try:
            diff_ik_context = self._robot_system._diff_ik.GetMyContextFromRoot(
                self.context
            )
            self._robot_system._diff_ik.SetPositions(diff_ik_context, q)
        except AttributeError:
            pass

    def set_diff_ik_position(self, q: np.ndarray):
        diff_ik_context = self._robot_system._diff_ik

    def get_run_flag(self):
        robot_system_context = self._robot_system.GetMyContextFromRoot(self.context)

    def reset(
        self,
        robot_position: np.ndarray,
        slider_pose: PlanarPose,
        pusher_pose: PlanarPose,
    ):
        if robot_position is not None:
            self.set_robot_position(robot_position)  # set robot position
        if slider_pose is not None:
            self.set_slider_planar_pose(slider_pose)  # set slider position
        if pusher_pose is not None:
            self._desired_position_source.reset(
                np.array([pusher_pose.x, pusher_pose.y])
            )  # reset controller

    def visualize_desired_slider_pose(self, time_in_recording: float = 0.0):
        if len(self._goal_geometries) == 0:
            self._goal_geometries = create_goal_geometries(
                self._robot_system,
                self._sim_config.slider_goal_pose,
            )

        visualize_desired_slider_pose(
            self._robot_system,
            self._sim_config.slider_goal_pose,
            self._goal_geometries,
            time_in_recording=time_in_recording,
        )

    def visualize_desired_pusher_pose(self, time_in_recording: float = 0.0):
        slider_shapes = get_slider_shapes(self._robot_system)
        height = min([shape.height() for shape in slider_shapes])

        if self._pusher_goal_geometry is None:
            color = COLORS["emeraldgreen"].diffuse(0.3)
            pusher_shape = Cylinder(
                self._sim_config.dynamics_config.pusher_radius, height
            )
            desired_pose = self._sim_config.pusher_start_pose.to_pose(
                height / 2, z_axis_is_positive=True
            )
            source_id = self._robot_system._scene_graph.RegisterSource()

            geom_instance = GeometryInstance(
                desired_pose,
                pusher_shape,
                f"pusher_shape",
            )
            shape_geometry_id = (
                self._robot_system._scene_graph.RegisterAnchoredGeometry(
                    source_id,
                    geom_instance,
                )
            )
            self._robot_system._scene_graph.AssignRole(
                source_id,
                shape_geometry_id,
                MakePhongIllustrationProperties(color),
            )
            self._pusher_goal_geometry = "pusher_goal_shape"
            self._robot_system._meshcat.SetObject(
                self._pusher_goal_geometry, pusher_shape, rgba=Rgba(*color)
            )

        self._robot_system._meshcat.SetTransform(
            self._pusher_goal_geometry,
            self._sim_config.pusher_start_pose.to_pose(
                height / 2, z_axis_is_positive=True
            ),
            time_in_recording,
        )

    def print_distance_to_target_pose(
        self, target_slider_pose: PlanarPose = PlanarPose(0.5, 0.0, 0.0)
    ):
        # Extract slider poses
        slider_position = self._plant.GetPositions(
            self.mbp_context, self._slider_model_instance
        )
        slider_pose = PlanarPose.from_generalized_coords(slider_position)

        # print distance to target pose
        x_error = target_slider_pose.x - slider_pose.x
        y_error = target_slider_pose.y - slider_pose.y
        theta_error = target_slider_pose.theta - slider_pose.theta
        print(f"\nx error: {100*x_error:.2f}cm")
        print(f"y error: {100*y_error:.2f}cm")
        print(
            f"orientation error: {theta_error*180.0/np.pi:.2f} degrees ({theta_error:.2f}rads)"
        )

    def save_recording(self, recording_file: Optional[str], save_dir: str):
        if recording_file:
            self._meshcat.StopRecording()
            self._meshcat.SetProperty("/drake/contact_forces", "visible", False)
            self._meshcat.PublishRecording()
            res = self._meshcat.StaticHtml()
            if save_dir:
                recording_file = os.path.join(save_dir, recording_file)
            with open(recording_file, "w") as f:
                f.write(res)

    def get_button_values(self):
        if isinstance(self._desired_position_source, GamepadControllerSource):
            return self._desired_position_source.get_button_values()
        else:
            raise NotImplementedError

    def get_pusher_pose_log(self):
        return self._pusher_pose_logger.FindLog(self.context)

    def get_slider_pose_log(self):
        return self._slider_pose_logger.FindLog(self.context)
