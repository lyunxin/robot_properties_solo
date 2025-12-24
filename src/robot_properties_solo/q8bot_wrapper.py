"""q8bot_wrapper

Q8bot interface using pinocchio's convention.
This wrapper interfaces with the VIRTUAL SERIAL URDF for high-level dynamics.

License: BSD 3-Clause License
"""
import numpy as np
from robot_properties_solo.config import Q8botConfig

dt = 1e-3


class Q8botRobot():
    """
    Q8bot robot wrapper for ROS + Gazebo/Mujoco projects.

    IMPORTANT:
    This class wraps the URDF model (Serial chain abstraction: Hip -> Knee).
    The MJCF/Physical robot uses a parallel five-bar linkage structure.

    - Pinocchio (URDF): Controls 'legX_hip_joint' and 'legX_knee_joint' (Virtual Serial Joints)
    - Mujoco (MJCF): Controls 'XX_joint' and 'XY_joint' (Actual Parallel Motors)

    A mapping is required to convert torques/positions between this wrapper
    and the actual simulation hardware interface.
    """

    def __init__(self):

        self.urdf_path = Q8botConfig.urdf_path
        self.mjcf_path = Q8botConfig.mjcf_path

        # Create the robot wrapper in pinocchio (Loads the Serial URDF).
        self.pin_robot = Q8botConfig.buildRobotWrapper()

        self.base_link_name = "base_link"
        self.end_eff_ids = []
        self.end_effector_names = []
        controlled_joints = []

        # Q8bot has 4 legs: leg1, leg2, leg3, leg4
        # Each leg in the URDF has 2 active joints: hip and knee
        for leg in ["leg1", "leg2", "leg3", "leg4"]:
            controlled_joints += [leg + "_hip_joint", leg + "_knee_joint"]
            self.end_eff_ids.append(
                self.pin_robot.model.getFrameId(leg + "_foot")
            )
            self.end_effector_names.append(leg + "_foot")

        self.joint_names = controlled_joints
        self.nb_ee = len(self.end_effector_names)

        # Store Frame IDs for feet (useful for contact detection/kinematics)
        self.leg1_foot_id = self.pin_robot.model.getFrameId("leg1_foot")
        self.leg2_foot_id = self.pin_robot.model.getFrameId("leg2_foot")
        self.leg3_foot_id = self.pin_robot.model.getFrameId("leg3_foot")
        self.leg4_foot_id = self.pin_robot.model.getFrameId("leg4_foot")

    def forward_robot(self, q=None, dq=None):
        """
        Computes forward kinematics and dynamics based on the Serial URDF model.
        """
        if q is None:
            q, dq = self.get_state()
        elif dq is None:
            raise ValueError("Need to provide q and dq or non of them.")

        self.pin_robot.forwardKinematics(q, dq, 0 * dq)
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)
        self.pin_robot.centroidalMomentum(q, dq)

    def reset_to_initial_state(self) -> None:
        """Reset robot state to the initial configuration (based on Q8botConfig)."""
        q0 = np.array(Q8botConfig.initial_configuration)
        dq0 = np.array(Q8botConfig.initial_velocity)
        self.reset_state(q0, dq0)

    def update_pinocchio(self, q, dq):
        """Updates the pinocchio robot.
        This includes updating:
        - kinematics
        - joint and frame jacobian
        - centroidal momentum
        Args:
          q: Pinocchio generalized position vector (Serial Joint Angles).
          dq: Pinocchio generalize velocity vector (Serial Joint Velocities).
        """
        self.pin_robot.forwardKinematics(q, dq)
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)
        self.pin_robot.centroidalMomentum(q, dq)

    def get_state(self):
        # This method assumes usage within a simulator that implements get_state
        # For standard wrappers, this often reads from the simulator interface
        # Placeholder if strictly following Solo/Go2 wrapper pattern which usually
        # inherits or expects a simulator binding.
        pass
