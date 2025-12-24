import numpy as np
import matplotlib.pyplot as plt


class FiveLinkRobot:
    def __init__(self):
        # === 1. 定义机构参数 ===
        self.d = 0.02  # 电机间距
        self.L1 = 0.025  # 左主动臂
        self.L2 = 0.025  # 右主动臂
        self.L3 = 0.04  # 左从动臂
        self.L4 = 0.04  # 右从动臂

        # 虚拟腿参数
        self.Serial_Base = np.array([self.d / 2.0, 0])
        self.LS1 = 0.025  # 虚拟大臂
        self.LS2 = 0.04  # 虚拟小臂

    def get_circle_intersection(self, p1, r1, p2, r2):
        """ 计算两圆交点 """
        x1, y1 = p1
        x2, y2 = p2
        dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist = np.sqrt(dist_sq)

        if dist > (r1 + r2) or dist < abs(r1 - r2) or dist == 0:
            return None

        a = (r1 ** 2 - r2 ** 2 + dist_sq) / (2 * dist)
        h = np.sqrt(max(0, r1 ** 2 - a ** 2))

        dx, dy = x2 - x1, y2 - y1

        # 选择特定的解分支 (根据实际机构构型选择)
        ix = x1 + a * (dx / dist) - h * (dy / dist)
        iy = y1 + a * (dy / dist) + h * (dx / dist)
        return np.array([ix, iy])

    def forward_kinematics(self, theta1, theta2):
        """ 求解正运动学,返回关键点坐标 """
        motor1_pos = np.array([0, 0])
        motor2_pos = np.array([self.d, 0])

        elbow1 = np.array([self.L1 * np.cos(theta1), self.L1 * np.sin(theta1)])
        elbow2 = np.array([self.d + self.L2 * np.cos(theta2), self.L2 * np.sin(theta2)])

        p_end = self.get_circle_intersection(elbow1, self.L3, elbow2, self.L4)

        if p_end is None:
            return None, None, None, None

        virtual_elbow = self.get_circle_intersection(self.Serial_Base, self.LS1, p_end, self.LS2)

        return elbow1, elbow2, p_end, virtual_elbow

    def get_jacobian_parallel(self, theta1, theta2, p_end, elbow1, elbow2):
        """
        计算五连杆并联雅可比矩阵 Jp
        关系: v_end = Jp * [d_theta1, d_theta2]^T
        """
        r3 = p_end - elbow1
        r4 = p_end - elbow2

        d_E1_d_th1 = np.array([-self.L1 * np.sin(theta1), self.L1 * np.cos(theta1)])
        d_E2_d_th2 = np.array([-self.L2 * np.sin(theta2), self.L2 * np.cos(theta2)])

        A = np.vstack([r3, r4])
        b1 = np.dot(r3, d_E1_d_th1)
        b2 = np.dot(r4, d_E2_d_th2)
        B = np.diag([b1, b2])

        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-6:
            return None, det_A

        J_p = np.dot(np.linalg.inv(A), B)
        return J_p, det_A

    def get_jacobian_serial(self, p_end, virtual_elbow):
        """
        计算虚拟串联雅可比矩阵 Js
        关系: v_end = Js * [d_phi1, d_phi2]^T
        """
        v1 = virtual_elbow - self.Serial_Base
        phi1 = np.arctan2(v1[1], v1[0])

        v2 = p_end - virtual_elbow
        phi1_2 = np.arctan2(v2[1], v2[0])
        phi2 = phi1_2 - phi1

        s1 = np.sin(phi1)
        c1 = np.cos(phi1)
        s12 = np.sin(phi1 + phi2)
        c12 = np.cos(phi1 + phi2)

        Js = np.array([
            [-self.LS1 * s1 - self.LS2 * s12, -self.LS2 * s12],
            [self.LS1 * c1 + self.LS2 * c12, self.LS2 * c12]
        ])

        det_Js = np.linalg.det(Js)
        return Js, np.array([phi1, phi2]), det_Js

    def test_torque_mapping(self, theta1=None, theta2=None):
        """完整的力矩映射测试"""
        print("=" * 60)
        print("五连杆力矩映射验证")
        print("=" * 60)

        # 使用更合理的默认角度
        if theta1 is None or theta2 is None:
            theta1 = np.deg2rad(120)  # 改用度数表示更直观
            theta2 = np.deg2rad(60)

        print(f"\n【步骤1】输入电机角度")
        print(f"  theta1 = {np.rad2deg(theta1):.2f}° ({theta1:.4f} rad)")
        print(f"  theta2 = {np.rad2deg(theta2):.2f}° ({theta2:.4f} rad)")

        # 正运动学
        el1, el2, p_end, v_elbow = self.forward_kinematics(theta1, theta2)
        if p_end is None:
            print("\n❌ 错误: 目标点在工作空间外!")
            return False

        print(f"\n【步骤2】正运动学求解")
        print(f"  末端位置: x={p_end[0] * 1000:.2f}mm, y={p_end[1] * 1000:.2f}mm")

        # 计算雅可比
        J_p, det_p = self.get_jacobian_parallel(theta1, theta2, p_end, el1, el2)
        J_s, q_virtual, det_s = self.get_jacobian_serial(p_end, v_elbow)

        if J_p is None:
            print("\n❌ 错误: 并联机构处于奇异位置!")
            return False

        print(f"\n【步骤3】计算雅可比矩阵")
        print(f"  虚拟角度: phi_hip={np.rad2deg(q_virtual[0]):.2f}°, phi_knee={np.rad2deg(q_virtual[1]):.2f}°")
        print(f"  det(J_p) = {det_p:.6f}")
        print(f"  det(J_s) = {det_s:.6f}")

        # 检查奇异性
        if abs(det_s) < 1e-6:
            print("  ⚠️  警告: 虚拟串联接近奇异!")

        # 映射矩阵
        J_map = np.dot(np.linalg.inv(J_s), J_p)
        print(f"\n【步骤4】速度映射矩阵 J_map = inv(J_s) * J_p")
        print(f"  J_map =\n{J_map}")

        # 力矩映射测试
        F_ext = np.array([0.0, -10.0])
        print(f"\n【步骤5】力矩映射测试")
        print(f"  假设末端外力: F = [{F_ext[0]:.1f}, {F_ext[1]:.1f}] N")

        tau_virtual = np.dot(J_s.T, F_ext)
        print(f"  虚拟力矩: tau_hip={tau_virtual[0]:.4f}, tau_knee={tau_virtual[1]:.4f} Nm")

        tau_parallel = np.dot(J_map.T, tau_virtual)
        print(f"  映射力矩: tau_1={tau_parallel[0]:.4f}, tau_2={tau_parallel[1]:.4f} Nm")

        tau_direct = np.dot(J_p.T, F_ext)
        print(f"  直接计算: tau_1={tau_direct[0]:.4f}, tau_2={tau_direct[1]:.4f} Nm")

        # 验证一致性
        error = np.linalg.norm(tau_parallel - tau_direct)
        print(f"\n  力矩误差: {error:.10f}")
        if error < 1e-8:
            print("  ✅ 力矩映射正确!")
        else:
            print("  ❌ 力矩映射有误!")

        # 能量守恒验证
        q_dot_real = np.array([1.0, -0.5])
        q_dot_virtual = np.dot(J_map, q_dot_real)

        power_parallel = np.dot(tau_parallel, q_dot_real)
        power_virtual = np.dot(tau_virtual, q_dot_virtual)

        print(f"\n【步骤6】能量守恒验证")
        print(f"  并联功率: P_parallel = {power_parallel:.8f} W")
        print(f"  虚拟功率: P_virtual  = {power_virtual:.8f} W")
        print(f"  功率误差: {abs(power_parallel - power_virtual):.10f} W")

        if abs(power_parallel - power_virtual) < 1e-6:
            print("  ✅ 能量守恒验证通过!")
            return True
        else:
            print("  ❌ 能量不守恒!")
            return False

    def test_multiple_poses(self):
        """测试多个位姿"""
        print("\n" + "=" * 60)
        print("多位姿测试")
        print("=" * 60)

        test_cases = [
            (np.deg2rad(90), np.deg2rad(90), "对称位姿"),
            (np.deg2rad(120), np.deg2rad(60), "非对称位姿"),
            (np.deg2rad(100), np.deg2rad(80), "中间位姿"),
        ]

        results = []
        for theta1, theta2, desc in test_cases:
            print(f"\n测试: {desc}")
            success = self.test_torque_mapping(theta1, theta2)
            results.append((desc, success))

        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        for desc, success in results:
            status = "✅ 通过" if success else "❌ 失败"
            print(f"  {desc}: {status}")

    def visualize_mechanism(self, theta1, theta2):
        """可视化机构"""
        el1, el2, p_end, v_elbow = self.forward_kinematics(theta1, theta2)
        if p_end is None:
            print("无法绘制: 位置无解")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 左图: 五连杆
        ax1.set_title("Five-bar Parallel Mechanism", fontsize=14, fontweight='bold')
        ax1.plot([0, el1[0]], [0, el1[1]], 'b-o', linewidth=3, markersize=8, label='Link 1')
        ax1.plot([self.d, el2[0]], [0, el2[1]], 'r-o', linewidth=3, markersize=8, label='Link 2')
        ax1.plot([el1[0], p_end[0]], [el1[1], p_end[1]], 'g-o', linewidth=3, markersize=8, label='Link 3')
        ax1.plot([el2[0], p_end[0]], [el2[1], p_end[1]], 'm-o', linewidth=3, markersize=8, label='Link 4')
        ax1.plot(0, 0, 'ko', markersize=10)
        ax1.plot(self.d, 0, 'ko', markersize=10)
        ax1.plot(p_end[0], p_end[1], 'ro', markersize=12, label='End Effector')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        ax1.legend()
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')

        # 右图: 虚拟串联
        ax2.set_title("Virtual Serial Mechanism", fontsize=14, fontweight='bold')
        ax2.plot([self.Serial_Base[0], v_elbow[0]], [self.Serial_Base[1], v_elbow[1]],
                 'c-o', linewidth=3, markersize=8, label='Virtual Link 1')
        ax2.plot([v_elbow[0], p_end[0]], [v_elbow[1], p_end[1]],
                 'y-o', linewidth=3, markersize=8, label='Virtual Link 2')
        ax2.plot(self.Serial_Base[0], self.Serial_Base[1], 'ko', markersize=10, label='Virtual Base')
        ax2.plot(p_end[0], p_end[1], 'ro', markersize=12, label='End Effector')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        ax2.legend()
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    robot = FiveLinkRobot()

    # 单次测试
    print("\n单次详细测试:")
    robot.test_torque_mapping()

    # 多位姿测试
    robot.test_multiple_poses()

    # 可视化
    print("\n生成可视化...")
    robot.visualize_mechanism(np.deg2rad(90), np.deg2rad(90))