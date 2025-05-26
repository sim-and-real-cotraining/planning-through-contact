from pydrake.all import (
    AddDefaultVisualization,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseDynamicsController,
    Meshcat,
    MultibodyPlant,
    StateInterpolatorWithDiscreteDerivative,
)

from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_utils import (
    AddSliderAndConfigureContact,
    GetParser,
)

from .robot_system_base import RobotSystemBase


class CylinderActuatedStation(RobotSystemBase):
    """Base controller class for an actuated floating cylinder robot."""

    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        meshcat: Meshcat,
    ):
        super().__init__()
        self._sim_config = sim_config
        self._meshcat = meshcat
        self._pid_gains = dict(kp=3200, ki=100, kd=50)
        self._num_positions = 2  # Number of dimensions for robot position

        builder = DiagramBuilder()

        # "Internal" plant for the robot controller
        robot_controller_plant = MultibodyPlant(time_step=self._sim_config.time_step)
        parser = GetParser(robot_controller_plant)
        parser.AddModelsFromUrl(
            "package://planning_through_contact/pusher_floating_hydroelastic_actuated.sdf"
        )[0]
        robot_controller_plant.set_name("robot_controller_plant")
        robot_controller_plant.Finalize()

        # "External" station plant
        self.station_plant, self._scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self._sim_config.time_step
        )
        self.slider = AddSliderAndConfigureContact(
            sim_config, self.station_plant, self._scene_graph
        )

        # Set the initial camera pose
        zoom = 1.8
        camera_in_world = [
            sim_config.slider_goal_pose.x,
            (sim_config.slider_goal_pose.y - 1) / zoom,
            1.5 / zoom,
        ]
        target_in_world = [
            sim_config.slider_goal_pose.x,
            sim_config.slider_goal_pose.y,
            0,
        ]
        self._meshcat.SetCameraPose(camera_in_world, target_in_world)
        AddDefaultVisualization(builder, self._meshcat)

        ## Add Leaf systems

        robot_controller = builder.AddNamedSystem(
            "RobotController",
            InverseDynamicsController(
                robot_controller_plant,
                kp=[self._pid_gains["kp"]] * self._num_positions,
                ki=[self._pid_gains["ki"]] * self._num_positions,
                kd=[self._pid_gains["kd"]] * self._num_positions,
                has_reference_acceleration=False,
            ),
        )

        # Add system to convert desired position to desired position and velocity.
        desired_state_source = builder.AddNamedSystem(
            "DesiredStateSource",
            StateInterpolatorWithDiscreteDerivative(
                self._num_positions,
                self._sim_config.time_step,
                suppress_initial_transient=True,
            ),
        )

        # Add cameras
        if sim_config.camera_configs:
            from pydrake.systems.sensors import ApplyCameraConfig

            for camera_config in sim_config.camera_configs:
                ApplyCameraConfig(config=camera_config, builder=builder)
                builder.ExportOutput(
                    builder.GetSubsystemByName(
                        f"rgbd_sensor_{camera_config.name}"
                    ).color_image_output_port(),
                    f"rgbd_sensor_{camera_config.name}",
                )

        ## Connect systems

        self._robot_model_instance = self.station_plant.GetModelInstanceByName(
            self.robot_model_name
        )
        builder.Connect(
            robot_controller.get_output_port_control(),
            self.station_plant.get_actuation_input_port(self._robot_model_instance),
        )

        builder.Connect(
            self.station_plant.get_state_output_port(self._robot_model_instance),
            robot_controller.get_input_port_estimated_state(),
        )

        builder.Connect(
            desired_state_source.get_output_port(),
            robot_controller.get_input_port_desired_state(),
        )

        ## Export inputs and outputs

        builder.ExportInput(
            desired_state_source.get_input_port(),
            "planar_position_command",
        )

        builder.ExportOutput(
            self.station_plant.get_state_output_port(self._robot_model_instance),
            "robot_state_measured",
        )

        # Only relevant when use_hardware=False
        # If use_hardware=True, this info will be updated by the optitrack system in the state estimator directly
        builder.ExportOutput(
            self.station_plant.get_state_output_port(self.slider),
            "object_state_measured",
        )

        builder.BuildInto(self)

        ## Set default position for the robot
        self.station_plant.SetDefaultPositions(
            self._robot_model_instance, self._sim_config.pusher_start_pose.pos()
        )

    @property
    def robot_model_name(self) -> str:
        """The name of the robot model."""
        return "pusher"

    @property
    def slider_model_name(self) -> str:
        """The name of the robot model."""
        """The name of the robot model."""
        if self._sim_config.slider.name == "box":
            return "box"
        elif self._sim_config.slider.name in ["tee", "t_pusher"]:
            return "t_pusher"
        elif self._sim_config.slider.name == "arbitrary":
            return "arbitrary"
        else:
            raise ValueError(f"Invalid slider name: {self._sim_config.slider.name}")

    def num_positions(self) -> int:
        return self._num_positions

    def get_station_plant(self):
        return self.station_plant

    def get_scene_graph(self):
        return self._scene_graph

    def get_slider(self):
        return self.slider

    def get_meshcat(self):
        return self._meshcat
