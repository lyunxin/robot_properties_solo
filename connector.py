#!/usr/bin/env python
import sys
import mujoco
import mujoco.viewer
import pybullet
import rclpy
from ftn_solo.utils.bullet_env import BulletEnvWithGround
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
from rosgraph_msgs.msg import Clock
import time
import math
import yaml
from robot_properties_solo import Resources
from ftn_solo.tasks import *
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from ftn_solo.utils.conversions import ToVector
from ftn_solo_control import SensorData
import xacro
import os
from ament_index_python.packages import get_package_share_directory

import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class Connector:
    def __init__(self, robot_version, logger, *args, **kwargs) -> None:
        self.resources = Resources(robot_version)
        with open(self.resources.config_path, "r") as stream:
            try:
                self.config = yaml.safe_load(stream)
            except Exception as exc:
                raise exc
        self.logger = logger

    def is_paused(self):
        return False


class RobotConnector(Connector):
    def __init__(self, robot_version, logger, *args, **kwargs) -> None:
        super().__init__(robot_version, logger, *args, **kwargs)
        import libodri_control_interface_pywrap as oci

        self.robot = oci.robot_from_yaml_file(self.resources.config_path)
        self.joint_names = self.config["robot"]["joint_modules"]["joint_names"]
        self.robot.initialize(
            np.array([0] * self.robot.joints.number_motors, dtype=np.float64)
        )
        self.running = True
        self.dt = 0.0010001
        self.nanoseconds = self.dt * 1e9

    def get_data(self):
        self.robot.parse_sensor_data()
        return self.robot.joints.positions, self.robot.joints.velocities

    def set_torques(self, torques):
        self.robot.joints.set_torques(torques)

    def is_running(self):
        if self.robot.has_error:
            self.logger.error("Error appeared")
        if self.robot.is_timeout:
            self.logger.error("Timeout happened with real robot")
        return not (self.robot.has_error)

    def step(self):
        self.robot.send_command_and_wait_end_of_cycle(self.dt)
        return True

    def num_joints(self):
        return self.robot.joints.number_motors

    def get_sensor_readings(self):
        q = self.robot.imu.attitude_quaternion
        data = SensorData()
        data.imu_data.attitude = np.array([q[3], q[0], q[1], q[2]])
        data.imu_data.angular_velocity = self.robot.imu.gyroscope
        return data


class SimulationConnector(Connector):
    def __init__(self, robot_version, logger, *args, **kwargs) -> None:
        super().__init__(robot_version, logger, *args, **kwargs)

        self.simulate_encoders = False
        self.initial_pose = None
        simulation_config = self.config.get("simulation")
        if self.config.get("simulation"):
            self.simulate_encoders = simulation_config.get(
                "simulate_encoders", False)
            self.initial_pose = simulation_config.get("initial_pose", None)

        if self.simulate_encoders:
            self.resolution = (2 * math.pi / self.config["robot"]["joint_modules"]
                               ["counts_per_revolution"] / self.config["robot"]["joint_modules"]["gear_ratios"])
            self.old_q = None

    def process_coordinates(self, q, qdot):
        if not self.simulate_encoders:
            return q, qdot
        else:
            q = np.round(q / self.resolution) * self.resolution
            if self.old_q is None:
                self.old_q = q
                return q, 0 * qdot
            qdot = (q - self.old_q) / self.dt
            self.old_q = q
            return q, qdot


class PybulletConnector(SimulationConnector):
    def __init__(self, robot_version, logger, fixed=False, pos=[0, 0, 0.4], rpy=[0.0, 0.0, 0.0]) -> None:
        super().__init__(robot_version, logger)

        self.env = BulletEnvWithGround(robot_version)
        orn = pybullet.getQuaternionFromEuler(rpy)
        self.dt = self.env.dt
        self.logger = logger
        self.robot_id = pybullet.loadURDF(
            self.resources.urdf_path, pos, orn, flags=pybullet.URDF_USE_INERTIA_FROM_FILE, useFixedBase=fixed)

        self.joint_names = []
        self.joint_ids = []
        self.end_effector_ids = []
        self.touch_sensors = ["fl", "fr", "hl", "hr"]
        self.end_effector_names = ["FR_ANKLE",
                                   "FL_ANKLE", "HR_ANKLE", "HL_ANKLE"]
        self.reading = {}
        self.running = True
        self.rot_base_to_imu = np.identity(3)
        self.r_base_to_imu = np.array([0.10407, -0.00635, 0.01540])

        for ji in range(pybullet.getNumJoints(self.robot_id)):
            if (pybullet.getJointInfo(self.robot_id, ji)[1].decode("UTF-8") in self.end_effector_names):
                self.end_effector_ids.append(
                    pybullet.getJointInfo(self.robot_id, ji)[0] - 1)
            elif pybullet.JOINT_FIXED != pybullet.getJointInfo(self.robot_id, ji)[2]:
                self.joint_names.append(pybullet.getJointInfo(
                    self.robot_id, ji)[1].decode("UTF-8"))
                self.joint_ids.append(
                    pybullet.getJointInfo(self.robot_id, ji)[0])

        pybullet.setJointMotorControlArray(
            self.robot_id, self.joint_ids, pybullet.VELOCITY_CONTROL, forces=np.zeros(len(self.joint_ids)))

    def get_data(self):
        q = np.empty(len(self.joint_ids))
        dq = np.empty(len(self.joint_ids))

        joint_states = pybullet.getJointStates(self.robot_id, self.joint_ids)

        for i in range(len(self.joint_ids)):
            q[i] = joint_states[i][0]
            dq[i] = joint_states[i][1]

        return self.process_coordinates(q, dq)

    def contact_sensors(self):
        contact_points = pybullet.getContactPoints(self.robot_id)
        bodies_in_contact = list()

        for contact_info in contact_points:
            bodies_in_contact.add(contact_info[3])

        self.reading = [self.end_effector_ids[j]
                        in bodies_in_contact for j, name in enumerate(self.touch_sensors)]

        return self.reading

    def imu_sensor(self):
        base_inertia_pos, base_inertia_quat = pybullet.getBasePositionAndOrientation(
            self.robot_id
        )
        rot_base_to_world = np.array(
            pybullet.getMatrixFromQuaternion(base_inertia_quat)
        ).reshape((3, 3))
        base_linvel, base_angvel = pybullet.getBaseVelocity(self.robot_id)

        imu_linacc = np.cross(base_angvel, np.cross(
            base_angvel, rot_base_to_world @ self.r_base_to_imu))

        return (
            base_inertia_quat,
            self.rot_base_to_imu.dot(
                rot_base_to_world.T.dot(np.array(base_angvel))),
            self.rot_base_to_imu.dot(
                rot_base_to_world.T.dot(
                    imu_linacc + np.array([0.0, 0.0, 9.81]))
            ),
        )

    def get_sensor_readings(self):
        readings = SensorData()
        q, gyro, accel = self.imu_sensor()

        readings.touch[i] = np.array(self.contact_sensors(), dtype=np.bool)
        readings.imu_data.attitude = np.array([q[3], q[0], q[1], q[2]])
        readings.imu_data.angular_velocity = gyro
        readings.imu_data.linear_acceleration = accel

        return readings

    def set_torques(self, torques):
        pybullet.setJointMotorControlArray(
            self.robot_id, self.joint_ids, pybullet.TORQUE_CONTROL, forces=torques
        )

    def step(self):
        self.env.step(True)
        return True

    def is_running(self):
        return self.running

    def num_joints(self):
        return len(self.joint_names)


class MujocoConnector(SimulationConnector):
    def __init__(self, robot_version, logger, use_gui=True, start_paused=False, fixed=False, pos=[0, 0, 0.4], rpy=[0.0, 0.0, 0.0], environment="", environments_package="") -> None:
        super().__init__(robot_version, logger)
        environment_path = ""
        if not environment == "":
            if not environments_package == "":
                environment_path = os.path.join(
                    get_package_share_directory(
                        environments_package), environment
                )
            else:
                environment_path = environment
        xml_string = xacro.process(
            self.resources.mjcf_path + ".xacro",
            mappings={
                "environment": environment_path,
                "resources_dir": self.resources.resources_dir,
            },
        )
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.model.opt.timestep = 0.0010001

        if fixed:
            self.model.equality("fixed").active0 = True
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0:3] = pos
        mujoco.mju_euler2Quat(self.data.qpos[3:7], rpy, "XYZ")
        self.data.qpos[7:] = 0
        if self.initial_pose is not None:
            self.data.qpos[7:] = self.initial_pose
        self.data.qvel[:] = 0
        self.joint_names = [self.model.joint(
            i + 1).name for i in range(self.model.nu)]
        self.paused = start_paused
        self.use_gui = use_gui
        self.viewer = None
        self.running = True
        self.dt = self.model.opt.timestep
        self.touch_sensors = ["fl", "fr", "hl", "hr"]
        if self.use_gui:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False, key_callback=self.key_callback)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    def key_callback(self, keycode):
        if chr(keycode) == " ":
            self.paused = not self.paused
        elif keycode == 256:  # ESC
            self.running = False

    def get_data(self):
        return self.process_coordinates(self.data.qpos[7:], self.data.qvel[6:])

    def get_sensor_readings(self):
        readings = SensorData()
        for i, sensor in enumerate(self.touch_sensors):
            name = sensor + "_touch"
            readings.touch[i] = self.data.sensor(name).data[0] > 0
        readings.imu_data.attitude = self.data.sensor("attitude").data
        readings.imu_data.angular_velocity = self.data.sensor(
            "angular-velocity").data
        readings.imu_data.linear_acceleration = self.data.sensor(
            "linear-acceleration"
        ).data
        readings.imu_data.magnetometer = self.data.sensor("magnetometer").data
        # qw, qx, qy, qz
        return readings

    def set_torques(self, torques):
        self.data.ctrl = torques

    def is_running(self):
        return self.running

    def step(self):
        if self.paused:
            time.sleep(self.model.opt.timestep)
            return False
        step_start = time.time()
        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()
        time_until_next_step = self.model.opt.timestep - \
            (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        return True

    def num_joints(self):
        return self.model.nu

    def is_paused(self):
        return self.paused


class FiveBarMujocoConnector(MujocoConnector):
    def __init__(self, robot_version, logger, **kwargs):
        # 临时保存虚拟initial_pose,避免维度不匹配
        # MuJoCo模型有16个关节(每条腿4个),但虚拟模型只有8个
        from robot_properties_solo import Resources
        import yaml

        resources = Resources(robot_version)
        with open(resources.config_path, "r") as stream:
            temp_config = yaml.safe_load(stream)

        virtual_initial_pose = None
        if temp_config.get("simulation") and temp_config["simulation"].get("initial_pose"):
            virtual_initial_pose = np.array(temp_config["simulation"]["initial_pose"])

        # 调用父类初始化(会因维度不匹配而失败,我们需要手动设置qpos)
        # 为了避免错误,我们先临时调用SimulationConnector来获取config
        SimulationConnector.__init__(self, robot_version, logger, **kwargs)

        # 保存并清除initial_pose,避免MujocoConnector尝试设置错误维度
        saved_initial_pose = self.initial_pose
        self.initial_pose = None

        # 现在手动执行MujocoConnector的初始化代码
        environment_path = ""
        environment = kwargs.get("environment", "")
        environments_package = kwargs.get("environments_package", "")

        if environment != "":
            if environments_package != "":
                environment_path = os.path.join(
                    get_package_share_directory(environments_package), environment
                )
            else:
                environment_path = environment

        xml_string = xacro.process(
            self.resources.mjcf_path + ".xacro",
            mappings={
                "environment": environment_path,
                "resources_dir": self.resources.resources_dir,
            },
        )
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.model.opt.timestep = 0.0010001

        fixed = kwargs.get("fixed", False)
        if fixed:
            self.model.equality("fixed").active0 = True

        self.data = mujoco.MjData(self.model)

        pos = kwargs.get("pos", [0, 0, 0.4])
        rpy = kwargs.get("rpy", [0.0, 0.0, 0.0])

        self.data.qpos[0:3] = pos
        mujoco.mju_euler2Quat(self.data.qpos[3:7], rpy, "XYZ")
        self.data.qpos[7:] = 0  # 默认所有关节为0
        self.data.qvel[:] = 0

        self.joint_names = [self.model.joint(i + 1).name for i in range(self.model.nu)]

        self.paused = kwargs.get("start_paused", False)
        self.use_gui = kwargs.get("use_gui", True)
        self.viewer = None
        self.running = True
        self.dt = self.model.opt.timestep
        self.touch_sensors = ["fl", "fr", "hl", "hr"]

        if self.use_gui:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False,
                key_callback=self.key_callback
            )
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        # 现在创建translator和设置初始位置
        self.translator = FiveBarTranslator()
        self.cached_q_virtual = np.zeros(8)
        self.cached_v_virtual = np.zeros(8)

        # 覆盖关节名称为虚拟串联关节名称
        # 真实的 MuJoCo 关节: [11_joint, 14_joint, 21_joint, 24_joint, ...]
        # 虚拟关节: [leg1_hip, leg1_knee, leg2_hip, leg2_knee, ...]
        self.joint_names = [
            "leg1_hip_joint", "leg1_knee_joint",
            "leg2_hip_joint", "leg2_knee_joint",
            "leg3_hip_joint", "leg3_knee_joint",
            "leg4_hip_joint", "leg4_knee_joint"
        ]
        self.leg_configs = [
            {"m1": "11_joint", "m2": "14_joint", "act1": "servo_11", "act2": "servo_14"},  # Leg 1
            {"m1": "21_joint", "m2": "24_joint", "act1": "servo_21", "act2": "servo_24"},  # Leg 2
            {"m1": "31_joint", "m2": "34_joint", "act1": "servo_31", "act2": "servo_34"},  # Leg 3
            {"m1": "41_joint", "m2": "44_joint", "act1": "servo_41", "act2": "servo_44"},  # Leg 4
        ]
        self.motor_indices = []

        for leg in self.leg_configs:
            # 获取关节 ID
            id1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, leg["m1"])
            id2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, leg["m2"])

            # 获取执行器 ID
            act1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, leg["act1"])
            act2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, leg["act2"])

            if id1 == -1 or id2 == -1:
                logger.error(f"Joints {leg['m1']} or {leg['m2']} not found in MJCF!")
                raise ValueError("Joint not found")

            self.motor_indices.append({
                # qpos 索引 (位置)
                "qpos": [self.model.jnt_qposadr[id1], self.model.jnt_qposadr[id2]],
                # qvel 索引 (速度 - 注意使用 dofadr)
                "qvel": [self.model.jnt_dofadr[id1], self.model.jnt_dofadr[id2]],
                # ctrl 索引 (力矩输入)
                "ctrl": [act1_id, act2_id]
            })

        # 如果有虚拟initial_pose,尝试设置真实关节位置
        if virtual_initial_pose is not None and len(virtual_initial_pose) == 8:
            logger.info("Setting initial pose from virtual joint angles...")
            for i in range(4):
                idx = i * 2
                q_hip = virtual_initial_pose[idx]
                q_knee = virtual_initial_pose[idx + 1]

                # 使用串联正运动学计算足端位置
                foot_pos = self.translator.forward_kinematics_serial(np.array([q_hip, q_knee]))

                # 使用数值方法求解五连杆逆运动学
                th1, th2 = self._solve_5bar_ik(foot_pos)

                # 设置真实关节位置
                indices = self.motor_indices[i]
                self.data.qpos[indices["qpos"][0]] = th1
                self.data.qpos[indices["qpos"][1]] = th2

        logger.info(">>> FIVE BAR Q8BOT CONNECTOR INITIALIZED <<<")
        logger.info(f"Virtual joint names: {self.joint_names}")

    def _solve_5bar_ik(self, foot_pos):
        """
        使用数值优化求解五连杆逆运动学
        给定足端位置,求解电机角度(th1, th2)
        """
        from scipy.optimize import minimize

        def objective(theta):
            # 计算给定电机角度下的足端位置
            pos = self.translator.forward_kinematics_5bar(theta[0], theta[1])
            if pos is None:
                return 1e6  # 惩罚无效配置
            # 最小化与目标位置的距离
            return np.linalg.norm(pos - foot_pos)

        # 使用初始猜测(接近零位)
        x0 = np.array([0.0, 0.0])
        result = minimize(objective, x0, method='BFGS', options={'maxiter': 100})

        if result.success and result.fun < 1e-3:
            return result.x[0], result.x[1]
        else:
            # 如果优化失败,返回默认值
            return 0.0, 0.0

    def get_data(self):
        # 1. 从 MuJoCo 获取真实电机数据 (8个关节)
        # MuJoCo qpos: [base_pos(3), base_quat(4), joints(8+)]
        # MuJoCo qvel: [base_vel(6), joints(8+)]
        q_virtual_all = np.zeros(8)
        v_virtual_all = np.zeros(8)

        # 2. 循环处理 4 条腿
        for i in range(4):
            idx = i * 2
            indices = self.motor_indices[i]  # 获取该腿在 MuJoCo 中的真实索引
            # 1. 使用映射获取真实电机数据
            # 获取位置 (qpos)
            th1 = self.data.qpos[indices["qpos"][0]]
            th2 = self.data.qpos[indices["qpos"][1]]

            # 获取速度 (qvel)
            th1_dot = self.data.qvel[indices["qvel"][0]]
            th2_dot = self.data.qvel[indices["qvel"][1]]

            # --- 正运动学 (Real -> Virtual) ---
            foot_pos = self.translator.forward_kinematics_5bar(th1, th2)

            if foot_pos is not None:
                q_virt = self.translator.inverse_kinematics_serial(foot_pos)
                q_virtual_all[idx] = q_virt[0]  # Hip
                q_virtual_all[idx + 1] = q_virt[1]  # Knee

                # 缓存虚拟位置
                self.cached_q_virtual[idx:idx + 2] = q_virt

                # --- 速度映射 (Real Vel -> Virtual Vel) ---
                J_p = self.translator.jacobian_5bar(th1, th2, foot_pos)
                J_s = self.translator.jacobian_serial(q_virt)

                v_motors = np.array([th1_dot, th2_dot])  # 使用映射后的速度
                v_foot = J_p @ v_motors

                try:
                    det_J_s = np.linalg.det(J_s)
                    if abs(det_J_s) > 1e-6:
                        v_virt = np.linalg.solve(J_s, v_foot)
                        v_virtual_all[idx] = v_virt[0]
                        v_virtual_all[idx + 1] = v_virt[1]
                        self.cached_v_virtual[idx:idx + 2] = v_virt
                    else:
                        v_virtual_all[idx:idx + 2] = self.cached_v_virtual[idx:idx + 2]
                except np.linalg.LinAlgError:
                    v_virtual_all[idx:idx + 2] = self.cached_v_virtual[idx:idx + 2]
            else:
                q_virtual_all[idx:idx + 2] = self.cached_q_virtual[idx:idx + 2]
                v_virtual_all[idx:idx + 2] = self.cached_v_virtual[idx:idx + 2]

            return self.process_coordinates(q_virtual_all, v_virtual_all)

    def set_torques(self, torques_virtual):
        # torques_virtual 是上层控制器算出来的 (8维虚拟关节力矩)

        # 这里的 torques_real_all 只是为了方便调试打印，实际赋值是直接操作 self.data.ctrl

        for i in range(4):
            idx = i * 2
            indices = self.motor_indices[i]

            # 获取真实角度用于计算 Jacobian (必须使用映射)
            th1 = self.data.qpos[indices["qpos"][0]]
            th2 = self.data.qpos[indices["qpos"][1]]

            tau_v = np.array([torques_virtual[idx], torques_virtual[idx + 1]])

            # --- 力矩映射 (Virtual -> Real) ---
            foot_pos = self.translator.forward_kinematics_5bar(th1, th2)

            tau_r = np.zeros(2)

            if foot_pos is not None:
                # 使用缓存的虚拟关节角度
                q_virt = self.cached_q_virtual[idx:idx + 2]

                # 防止初始时刻缓存为0导致的计算错误
                if np.allclose(q_virt, 0):
                    try:
                        q_virt = self.translator.inverse_kinematics_serial(foot_pos)
                    except:
                        pass

                J_s = self.translator.jacobian_serial(q_virt)
                J_p = self.translator.jacobian_5bar(th1, th2, foot_pos)

                try:
                    det_J_s = np.linalg.det(J_s)
                    det_J_p = np.linalg.det(J_p)

                    if abs(det_J_s) > 1e-6 and abs(det_J_p) > 1e-6:
                        # 雅可比力矩变换
                        F_foot = np.linalg.solve(J_s.T, tau_v)
                        tau_r = J_p.T @ F_foot
                    else:
                        tau_r = np.zeros(2)
                except np.linalg.LinAlgError:
                    tau_r = np.zeros(2)
            else:
                tau_r = np.zeros(2)

            # 2. 将计算出的力矩赋给对应的真实执行器
            self.data.ctrl[indices["ctrl"][0]] = tau_r[0]
            self.data.ctrl[indices["ctrl"][1]] = tau_r[1]

class ConnectorNode(Node):
    def __init__(self):
        super().__init__("first_node")
        self.declare_parameter("hardware", rclpy.Parameter.Type.STRING)
        hardware = self.get_parameter(
            "hardware").get_parameter_value().string_value
        self.time_publisher = None
        if hardware.lower() != "robot":
            self.time_publisher = self.create_publisher(Clock, "/clock", 10)
        self.clock = Clock()
        self.declare_parameter("use_gui", True)
        self.declare_parameter("start_paused", False)
        self.declare_parameter("fixed", False)
        self.declare_parameter("pos", [0.0, 0.0, 0.4])
        self.declare_parameter("rpy", [0.0, 0.0, 0.0])
        self.declare_parameter("robot_version", rclpy.Parameter.Type.STRING)
        self.declare_parameter("task", rclpy.Parameter.Type.STRING)
        self.declare_parameter("config", rclpy.Parameter.Type.STRING)
        self.join_state_pub = self.create_publisher(
            JointState, "/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        robot_version = (
            self.get_parameter(
                "robot_version").get_parameter_value().string_value
        )
        task = self.get_parameter("task").get_parameter_value().string_value

        self.fixed = False
        yaml_config = self.get_parameter(
            "config").get_parameter_value().string_value
        with open(yaml_config) as stream:
            try:
                self.config = yaml.safe_load(stream)
            except Exception as exc:
                raise exc

        if hardware.lower() != "robot":
            use_gui = self.get_parameter(
                "use_gui").get_parameter_value().bool_value
            self.fixed = self.get_parameter(
                "fixed").get_parameter_value().bool_value
            pos = self.get_parameter(
                "pos").get_parameter_value().double_array_value
            rpy = self.get_parameter(
                "rpy").get_parameter_value().double_array_value
            # Can go lower if we set niceness
            self.allowed_time = 1.0
            if hardware == "mujoco_5bar":
                mujoco_config = self.config.get("mujoco", {})
                self.connector = FiveBarMujocoConnector(
                    robot_version, self.get_logger(),
                    pos=pos, rpy=rpy, **mujoco_config
                )
            elif hardware.lower() == "mujoco":
                self.connector = MujocoConnector(robot_version, self.get_logger(), pos=pos, rpy=rpy, **self.config["mujoco"])
            elif hardware.lower() == "pybullet":
                self.connector = PybulletConnector(
                    robot_version, self.get_logger(), fixed=self.fixed, pos=pos, rpy=rpy)
        else:
            niceness = os.nice(0)
            niceness = os.nice(-20 - niceness)
            self.get_logger().info("Setting niceness to {}".format(niceness))
            self.allowed_time = 1.0
            self.connector = RobotConnector(robot_version, self.get_logger())

        self.get_logger().info("Allowed time to run is {}".format(self.allowed_time))
        if task == "joint_spline":
            self.task = TaskJointSpline(
                self.connector.num_joints(), robot_version, self.config
            )
        elif task == "move_base":
            self.task = TaskMoveBase(
                self.connector.num_joints(), robot_version, self.config
            )
        elif task == "draw_shapes":
            self.task = TaskDrawShapes(
                self.connector.num_joints(), robot_version, self.config
            )
        elif task == "friction_identification":
            self.task = TaskFrictionIdentification(
                self.connector.num_joints(), robot_version, self.config
            )
        else:
            self.get_logger().error("Unknown task selected!!! Switching to joint_spline task!")
            self.task = TaskJointSpline(
                robot_version, "/home/ajsmilutin/solo/solo_ws/src/ftn_solo/config/controllers/eurobot_demo.yaml",)
        self.task.dt = self.connector.dt

    def run(self):
        c = 0
        start = self.get_clock().now()
        joint_state = JointState()
        transform = TransformStamped()
        position, velocity = self.connector.get_data()
        self.task.init_pose(position, velocity)
        while self.connector.is_running():
            if self.connector.is_paused():
                continue
            position, velocity = self.connector.get_data()
            sensors = self.connector.get_sensor_readings()
            if self.time_publisher:
                elapsed = self.clock.clock.sec + self.clock.clock.nanosec / 1e9
            else:
                elapsed = (self.get_clock().now() - start).nanoseconds / 1e9

            try:
                with time_limit(self.allowed_time):
                    torques = self.task.compute_control(
                        elapsed, position, velocity, sensors
                    )
                    if self.time_publisher:
                        self.clock.clock.nanosec += int(
                            self.connector.dt * 1000000000)
                        self.clock.clock.sec += self.clock.clock.nanosec // 1000000000
                        self.clock.clock.nanosec = self.clock.clock.nanosec % 1000000000
                        self.time_publisher.publish(self.clock)
                    c += 1
                    if c % 2 == 0:
                        stamp = self.get_clock().now().to_msg()
                        if self.time_publisher:
                            joint_state.header.stamp = self.clock.clock
                        if hasattr(self.task, "estimator"):
                            if (self.task.estimator and self.task.estimator.initialized()):
                                self.task.estimator.publish_state(stamp.sec, stamp.nanosec)
                        else:
                            joint_state.position = position.tolist()
                            joint_state.velocity = velocity.tolist()
                            joint_state.name = self.connector.joint_names
                            self.join_state_pub.publish(joint_state)
                            transform.header.stamp = joint_state.header.stamp
                            transform.header.frame_id = "world"
                            transform.child_frame_id = "base_link"
                            if self.fixed:
                                transform.transform.translation.z = 0.4
                            transform.transform.rotation.w = sensors.imu_data.attitude[0]
                            transform.transform.rotation.x = sensors.imu_data.attitude[1]
                            transform.transform.rotation.y = sensors.imu_data.attitude[2]
                            transform.transform.rotation.z = sensors.imu_data.attitude[3]
                            self.tf_broadcaster.sendTransform(transform)
            except TimeoutException as e:
                self.get_logger().error("====== TIMED OUT! ======")
                exit()
            self.connector.set_torques(torques)
            self.connector.step()

class FiveBarTranslator:
    def __init__(self):
        self.l1 = 0.025
        self.l2 = 0.040
        self.motor_dist = 0.020
        self.d = self.motor_dist / 2.0

    def forward_kinematics_serial(self, q_virtual):
        """
        串联机构正运动学: 给定虚拟关节角度(hip, knee),计算足端位置
        虚拟串联模型: Hip(旋转) -> Thigh(l1) -> Knee(旋转) -> Shin(l2) -> Foot
        """
        q_hip, q_knee = q_virtual
        # 注意:虚拟串联机构中,hip角度是相对于垂直方向的
        # 第一段连杆末端位置
        x1 = self.l1 * np.sin(q_hip)
        y1 = -self.l1 * np.cos(q_hip)
        # 第二段连杆(相对于第一段)
        x2 = x1 + self.l2 * np.sin(q_hip + q_knee)
        y2 = y1 - self.l2 * np.cos(q_hip + q_knee)
        return np.array([x2, y2])

    def forward_kinematics_5bar(self, theta1, theta2):
        p1 = np.array([-self.d + self.l1 * np.sin(theta1), -self.l1 * np.cos(theta1)])
        p2 = np.array([ self.d + self.l1 * np.sin(theta2), -self.l1 * np.cos(theta2)])
        dist = np.linalg.norm(p2 - p1)
        if dist > 2 * self.l2 or dist == 0: return None
        mid = (p1 + p2) / 2
        h = np.sqrt(self.l2**2 - (dist / 2)**2)
        direction = p2 - p1
        normal = np.array([-direction[1], direction[0]]) / dist
        foot = mid - normal * h
        return foot

    def inverse_kinematics_serial(self, foot_pos):
        x, y = foot_pos
        r = np.linalg.norm(foot_pos)
        if r > (self.l1 + self.l2) * 0.999: r = (self.l1 + self.l2) * 0.999
        cos_angle = (self.l1**2 + self.l2**2 - r**2) / (2 * self.l1 * self.l2)
        q_knee = np.pi - np.arccos(cos_angle)
        alpha = np.arctan2(y, x)
        cos_beta = (self.l1**2 + r**2 - self.l2**2) / (2 * self.l1 * r)
        beta = np.arccos(cos_beta)
        q_hip = alpha + beta - (-np.pi/2)
        return np.array([q_hip, q_knee])

    def jacobian_serial(self, q_virtual):
        q1, q2 = q_virtual
        s1, c1 = np.sin(q1), np.cos(q1)
        s12, c12 = np.sin(q1+q2), np.cos(q1+q2)
        return np.array([[self.l1*c1 + self.l2*c12, self.l2*c12],
                         [self.l1*s1 + self.l2*s12, self.l2*s12]])

    def jacobian_5bar(self, theta1, theta2, foot_pos):
        x, y = foot_pos
        p1 = np.array([-self.d + self.l1 * np.sin(theta1), -self.l1 * np.cos(theta1)])
        p2 = np.array([ self.d + self.l1 * np.sin(theta2), -self.l1 * np.cos(theta2)])
        v1 = np.array([ self.l1 * np.cos(theta1), self.l1 * np.sin(theta1)])
        v2 = np.array([ self.l1 * np.cos(theta2), self.l1 * np.sin(theta2)])
        vec1, vec2 = foot_pos - p1, foot_pos - p2
        A = np.array([vec1, vec2])
        B = np.zeros((2, 2))
        B[0, 0] = np.dot(vec1, v1)
        B[1, 1] = np.dot(vec2, v2)
        if abs(np.linalg.det(A)) < 1e-6: return np.zeros((2,2))
        return np.linalg.inv(A) @ B


def main(args=None):
    rclpy.init(args=args)
    node = ConnectorNode()
    node.run()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
