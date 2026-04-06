import time

import rclpy
from control_msgs.action import GripperCommand
from franka_msgs.action import Homing
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from rclpy.action import ActionClient
from rclpy.node import Node

DEFAULT_GRIPPER_COMMAND_TOPIC = "gripper/gripper_client/target_gripper_width_percent"
DEFAULT_GRIPPER_ACTION_TOPIC = "franka_gripper/gripper_action"
DEFAULT_HOMING_ACTION_TOPIC = "franka_gripper/homing"
DEFAULT_JOINT_STATES_TOPIC = "franka_gripper/joint_states"


class GripperTeleopController(Node):
    def __init__(self):
        super().__init__("gripper_teleop_controller")

        self.declare_parameter("gripper_command_topic", DEFAULT_GRIPPER_COMMAND_TOPIC)
        self.declare_parameter("gripper_action_topic", DEFAULT_GRIPPER_ACTION_TOPIC)
        self.declare_parameter("homing_action_topic", DEFAULT_HOMING_ACTION_TOPIC)
        self.declare_parameter("joint_states_topic", DEFAULT_JOINT_STATES_TOPIC)
        self.declare_parameter("skip_homing_if_unavailable", True)
        self.declare_parameter("default_max_width", 0.08)
        self.declare_parameter("action_server_timeout", 3.0)
        self.declare_parameter("joint_state_timeout", 2.0)
        self.declare_parameter("command_epsilon", 0.001)
        self.declare_parameter("control_rate_hz", 15.0)
        self.declare_parameter("max_width_step", 0.01)
        self.declare_parameter("max_effort", 40.0)

        self._gripper_command_topic = (
            self.get_parameter("gripper_command_topic").get_parameter_value().string_value
        )
        self._gripper_action_topic = (
            self.get_parameter("gripper_action_topic").get_parameter_value().string_value
        )
        self._homing_action_topic = (
            self.get_parameter("homing_action_topic").get_parameter_value().string_value
        )
        self._joint_states_topic = (
            self.get_parameter("joint_states_topic").get_parameter_value().string_value
        )
        self._skip_homing_if_unavailable = (
            self.get_parameter("skip_homing_if_unavailable").get_parameter_value().bool_value
        )
        self._default_max_width = (
            self.get_parameter("default_max_width").get_parameter_value().double_value
        )
        self._action_server_timeout = (
            self.get_parameter("action_server_timeout").get_parameter_value().double_value
        )
        self._joint_state_timeout = (
            self.get_parameter("joint_state_timeout").get_parameter_value().double_value
        )
        self._command_epsilon = max(
            0.0, self.get_parameter("command_epsilon").get_parameter_value().double_value
        )
        self._control_rate_hz = max(
            1.0, self.get_parameter("control_rate_hz").get_parameter_value().double_value
        )
        self._max_width_step = max(
            1e-4, self.get_parameter("max_width_step").get_parameter_value().double_value
        )
        self._max_effort = self.get_parameter("max_effort").get_parameter_value().double_value

        self._current_width = None
        self._desired_width = None
        self._max_width = self._default_max_width
        self._goal_in_flight = False
        self._last_commanded_width = None

        self.get_logger().info("Initializing gripper teleop controller...")
        self._home_gripper()
        self._initialize_joint_state_reader()

        self._action_client = ActionClient(self, GripperCommand, self._gripper_action_topic)
        self.get_logger().info(
            f"Waiting for gripper action server {self._gripper_action_topic}..."
        )
        if not self._action_client.wait_for_server(timeout_sec=self._action_server_timeout):
            raise RuntimeError(
                f"Gripper action server {self._gripper_action_topic} is not available after "
                f"{self._action_server_timeout} seconds."
            )

        self.create_subscription(
            Float32,
            self._gripper_command_topic,
            self._gripper_command_callback,
            10,
        )
        self.create_subscription(
            JointState,
            self._joint_states_topic,
            self._joint_state_callback,
            10,
        )
        self.timer = self.create_timer(1.0 / self._control_rate_hz, self._control_step)
        self.get_logger().info("Gripper teleop controller initialized")

    def _home_gripper(self) -> None:
        self.get_logger().info("Starting gripper homing...")
        homing_client = ActionClient(self, Homing, self._homing_action_topic)
        self.get_logger().info(
            f"Waiting for homing action server {self._homing_action_topic}..."
        )
        if not homing_client.wait_for_server(timeout_sec=self._action_server_timeout):
            if self._skip_homing_if_unavailable:
                self.get_logger().warning(
                    f"Homing action server {self._homing_action_topic} not available after "
                    f"{self._action_server_timeout} seconds; continuing without homing."
                )
                return
            raise RuntimeError(
                f"Homing action server {self._homing_action_topic} is not available after "
                f"{self._action_server_timeout} seconds."
            )

        goal_msg = Homing.Goal()
        future = homing_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            raise RuntimeError("Homing action rejected")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        time.sleep(2)
        if not result.result.success:
            raise RuntimeError("Gripper homing failed")

    def _initialize_joint_state_reader(self) -> None:
        future = rclpy.task.Future()

        def initial_joint_state_callback(msg: JointState) -> None:
            if not msg.position:
                return
            self._update_width_from_joint_state(msg)
            future.set_result(True)

        subscription = self.create_subscription(
            JointState, self._joint_states_topic, initial_joint_state_callback, 10
        )
        self.get_logger().info(f"Waiting for {self._joint_states_topic}...")
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._joint_state_timeout)
        self.destroy_subscription(subscription)

        if not future.done():
            self.get_logger().warning(
                f"No gripper joint state received on {self._joint_states_topic}; using default "
                f"max width {self._default_max_width} m."
            )
            self._current_width = self._default_max_width
            self._desired_width = self._default_max_width
            self._max_width = self._default_max_width
            return

        self._desired_width = self._current_width

    def _joint_state_callback(self, msg: JointState) -> None:
        self._update_width_from_joint_state(msg)

    def _update_width_from_joint_state(self, msg: JointState) -> None:
        if not msg.position:
            return
        finger_position = msg.position[0]
        self._current_width = max(0.0, 2.0 * finger_position)
        self._max_width = max(self._max_width, self._current_width, self._default_max_width)

    def _gripper_command_callback(self, msg: Float32) -> None:
        command_percent = max(0.0, min(1.0, msg.data))
        self._desired_width = self._max_width * command_percent

    def _control_step(self) -> None:
        if self._goal_in_flight:
            return
        if self._desired_width is None or self._current_width is None:
            return

        error = self._desired_width - self._current_width
        if abs(error) < self._command_epsilon:
            return

        step = max(-self._max_width_step, min(self._max_width_step, error))
        next_width = max(0.0, min(self._max_width, self._current_width + step))
        if (
            self._last_commanded_width is not None
            and abs(next_width - self._last_commanded_width) < self._command_epsilon
        ):
            return

        self._send_gripper_command(next_width)

    def _send_gripper_command(self, gripper_width: float) -> None:
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = 0.5 * gripper_width
        goal_msg.command.max_effort = self._max_effort
        self._goal_in_flight = True
        self._last_commanded_width = gripper_width
        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future: rclpy.task.Future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._goal_in_flight = False
            self.get_logger().warning("Gripper teleop goal rejected")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future: rclpy.task.Future) -> None:
        self._goal_in_flight = False
        result = future.result().result
        self.get_logger().debug(
            "Gripper teleop result: position=%s reached_goal=%s stalled=%s"
            % (result.position, result.reached_goal, result.stalled)
        )


def main(args=None):
    rclpy.init(args=args)
    node = GripperTeleopController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
