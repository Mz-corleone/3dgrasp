import json
import socket
import sys
import time
import threading
import select
import termios
import tty
import math

from libs.auxiliary import get_ip
import numpy as np
from scipy.spatial.transform import Rotation as R

from Robotic_Arm.rm_robot_interface import * #RoboticArm, rm_thread_mode_e, rm_frame_t,rm_peripheral_read_write_params_t
from detec_yolo import OrbbecYOLOGloveDetector


# ─── 常量 ───────────────────────────────────────────────
MODEL_NAME = 'glove'
CONF_THRESHOLD = 0.5
ARM_SPEED = 50
DETECTION_SLEEP = 0.01          # 检测线程 sleep 秒数
MAIN_LOOP_SLEEP = 0.1           # 主循环 timeout 秒数
DISPLAY_EVERY_N_FRAMES = 10     # 每 N 帧打印一次状态
ARM_CONNECT_TIMEOUT = 5.0       # socket 连接超时秒数
ARM_RECV_TIMEOUT = 5.0          # socket recv 超时秒数

# ─── 手眼标定矩阵 ───────────────────────────────────────
ROTATION_MATRIX = np.array([
    [-0.99531643,  0.09664755, -0.00210937],
    [-0.09513715, -0.98316625, -0.15599056],
    [-0.01714997, -0.15505928,  0.98775629]
])
TRANSLATION_VECTOR = np.array([0.00926601, 0.06804014, 0.07758337])

# ─── 辅助函数 ───────────────────────────────────────────
def _send_with_timeout(client, data: bytes, timeout: float) -> bytes:
    client.settimeout(timeout)
    client.send(data)
    return client.recv(8192)


def send_cmd(client, cmd, get_pose=True):
    response = _send_with_timeout(client, cmd.encode('utf-8'), ARM_RECV_TIMEOUT)
    if not get_pose:
        return True

    decoded = response.decode('utf-8')
    try:
        # 整个响应可能包含多个 JSON 对象，用 next() 找目标
        data_list = json.loads(decoded) if decoded.startswith('[') else [json.loads(decoded)]
    except json.JSONDecodeError as e:
        raise RuntimeError(f'JSON 解析失败: {e}\n原始响应: {decoded!r}')

    target_data = None
    for data in data_list:
        if isinstance(data, dict) and data.get('state') == 'current_arm_state':
            target_data = data
            break

    if target_data is None:
        raise RuntimeError(f'未找到有效的机械臂状态响应\n原始响应: {decoded!r}')

    if target_data.get('arm_state', {}).get('err', [0]) != [0]:
        raise RuntimeError(f"机械臂报错: {target_data['arm_state']['err']}")

    pose_raw = target_data['arm_state']['pose']
    return [
        pose_raw[0] / 1_000_000,
        pose_raw[1] / 1_000_000,
        pose_raw[2] / 1_000_000,
        pose_raw[3] / 1_000,
        pose_raw[4] / 1_000,
        pose_raw[5] / 1_000,
    ]


def set_base_frame(client):
    send_cmd(client, '{"command":"set_change_work_frame","frame_name":"Base"}', get_pose=False)


def get_end_effector_pose(client):
    return send_cmd(client, '{"command": "get_current_arm_state"}')


def connect_robot():
    robot_ip = get_ip()
    if not robot_ip:
        raise RuntimeError('无法找到机械臂IP，请检查网络连接')

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(ARM_CONNECT_TIMEOUT)
    client.connect((robot_ip, 8080))
    set_base_frame(client)
    return client


def convert_camera_to_base(x, y, z, ee_pose):
    obj_camera = np.array([x, y, z, 1.0])

    T_cam_to_ee = np.eye(4)
    T_cam_to_ee[:3, :3] = ROTATION_MATRIX
    T_cam_to_ee[:3, 3] = TRANSLATION_VECTOR

    position = np.array(ee_pose[:3])
    orientation = R.from_euler('xyz', ee_pose[3:], degrees=False).as_matrix()

    T_base_to_ee = np.eye(4)
    T_base_to_ee[:3, :3] = orientation
    T_base_to_ee[:3, 3] = position

    obj_base = T_base_to_ee @ T_cam_to_ee @ obj_camera
    return obj_base[:3].tolist()


def connect_arm(robot_ip: str):
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(robot_ip, 8080)
    return arm

# ─── 夹爪 ───────────────────────────────────────────
class Gripper:
    def __init__(self, arm):
        self.arm = arm
        self.arm.rm_set_modbus_mode(1,115200,1) #设置modbus模式
        write_params = rm_peripheral_read_write_params_t(1, 1, 2) #设置寄存器地址为1，寄存器数量为2
        self.arm.rm_write_single_coil(write_params, 1) #写入1使能夹爪控制
        write_params = rm_peripheral_read_write_params_t(1, 11, 1) #设置寄存器地址为11，寄存器数量为1
        self.arm.rm_write_single_register(write_params, 1000) #写入1000使夹爪松开
        write_params = rm_peripheral_read_write_params_t(1, 12, 1)
        self.arm.rm_write_single_register(write_params, 1000) #写入1000使夹爪松开

    def gripper_release(self):
        write_params = rm_peripheral_read_write_params_t(1, 10, 1) #设置寄存器地址为11，寄存器数量为1
        self.arm.rm_write_single_register(write_params, 1000) 
        

    def gripper_pick(self):
        write_params = rm_peripheral_read_write_params_t(1, 10, 1) #设置寄存器地址为10，寄存器数量为1
        self.arm.rm_write_single_register(write_params, 10)


#arm.rm_set_modbus_mode(1,115200,1)
#write_params = rm_peripheral_read_write_params_t(1, 10, 1)

#def gripper_release(arm):
#    write_params = rm_peripheral_read_write_params_t(1, 11, 1)
#    arm.rm_write_single_register(write_params, 1000)

#def gripper_pick(arm):
#    write_params = rm_peripheral_read_write_params_t(1, 10, 1)
#    arm.rm_write_single_register(write_params, 10)

# ─── GraspDetector ───────────────────────────────────────
class GraspDetector:
    def __init__(self, model_name=MODEL_NAME, conf_threshold=CONF_THRESHOLD):
        self.detector = OrbbecYOLOGloveDetector(model_name=model_name, conf_threshold=conf_threshold)
        self._latest_detection = None
        self._latest_position = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start_detection(self):
        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()

    def _detection_loop(self):
        while self._running:
            color_image, depth_image = self.detector.get_frames()
            if color_image is None or depth_image is None:
                time.sleep(DETECTION_SLEEP)
                continue

            detections = self.detector.detect_Gloves(color_image)
            with self._lock:
                if detections:
                    self._latest_detection = detections[0]
                    self._latest_position = self.detector.get_3d_position(
                        self._latest_detection, depth_image
                    )
                else:
                    self._latest_detection = None
                    self._latest_position = None

            time.sleep(DETECTION_SLEEP)

    def get_latest(self):
        with self._lock:
            return self._latest_detection, self._latest_position

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.detector.stop()


def adjust_target_position(arm, target_pose, scale=0.8):
    """如果目标不可达，将目标向机械臂方向拉近"""
    state, pose = arm.rm_get_current_arm_state()
    if state != 0: # 获取机械臂状态失败，无法调整
        return target_pose

    current_pos = pose['pose'][:3]
    adjusted = list(target_pose)

    # 将位置向当前位置方向拉近
    for i in range(2):
        adjusted[i] = current_pos[i] + (target_pose[i] - current_pos[i]) * scale
    adjusted[2] = target_pose[2]  # z保持不变
    return adjusted

# ─── 抓取序列 ───────────────────────────────────────────
def execute_grasp_sequence(arm, gripper, x, y, z):
    arm.rm_change_tool_frame("Arm_Tip")
    gripper.gripper_release()
    time.sleep(0.5)

    state, pose = arm.rm_get_current_arm_state()
    if state != 0:
        raise RuntimeError(f"获取机械臂状态失败, 错误码: {state}")
    print(f'当前工具位姿:\n {pose["pose"]}')

    ee_pose = pose['pose']
    base_pos = convert_camera_to_base(x, y, z, ee_pose)
    print(f'物体在基座下的坐标:\n {base_pos}')


    # 设置/切换工具坐标系
    tool_name = "tcp_offset"
    tcp_pose = [0, 0, 0.33, 0, 0, math.pi/2]  # z轴+0.33m，绕z轴逆时针90°
    frame = rm_frame_t(tool_name, tcp_pose, 1, 0, 0, 0)
    try:
        ret = arm.rm_set_manual_tool_frame(frame)
    except Exception as e:
        print(f"设置工具坐标系异常: {e}")
    arm.rm_change_tool_frame(tool_name)

    #往中心靠近一些，增加抓取成功率
    base_pos[1] = base_pos[1] * 0.75
    base_pos[2] = -0.530

    # 设置两种姿态进行抓取
    if base_pos[0] > -0.23:     # 物体靠近机械臂时
        base_pos[0] = base_pos[0] * 1.4 
        target_pose = list(base_pos) + [-3.141, 0.122, 2.828]
        move_ret = arm.rm_movej_p(target_pose, ARM_SPEED, 0, 0, True)
        print(f'运动到目标返回码: {move_ret}')
        if move_ret != 0:
            print(f"运动到目标位置失败(返回码:{move_ret})，尝试进一步调整...")
            target_pose = adjust_target_position(arm, target_pose)
            move_ret = arm.rm_movej_p(target_pose, ARM_SPEED, 0, 0, True)

            if move_ret != 0:
                print("无法运动到目标位置附近，跳过")
                return False
    else:
        target_pose = list(base_pos) + [-3.141, -0.222, -3.092]     #3.141, -0.222, -3.092
        # print(f'目标TCP位姿: {target_pose}')
        move_ret = arm.rm_movej_p(target_pose, ARM_SPEED, 0, 0, True)
        print(f'运动到目标返回码: {move_ret}')
        if move_ret != 0:
            print(f"运动到目标位置失败(返回码:{move_ret})，尝试进一步调整...")
            target_pose = adjust_target_position(arm, target_pose)
            move_ret = arm.rm_movej_p(target_pose, ARM_SPEED, 0, 0, True)

            if move_ret != 0:
                print("无法运动到目标位置附近，跳过")
                return False

    target_pose[2] = -0.534   # 固定z为-0.533
    target_pose[1] -= 0.02
    target_pose[0] += 0.01   # x减去0.03
    last_pose = list(target_pose)

    arm.rm_movej_p(last_pose, ARM_SPEED, 0, 0, 1)
    #gripper =Gripper(arm)
    gripper.gripper_pick()
    time.sleep(3)  # 等待夹爪闭合
    
    # arm.rm_movej_p([-0.263, -0.0001, -0.238, 3.141, -0.028, 3.141], ARM_SPEED, 0, 0, 1)
    
    arm.rm_movej_p([-0.443, 0.038, 0.339, 3.13, -0.791, -3.038], ARM_SPEED, 0, 0, 1)
    # arm.rm_movej_p([0.193, -0.247, 0.734, -0.559, -1.331, -3.067], ARM_SPEED, 0, 0, 1)
    arm.rm_movej_p([0.199, -0.246, 0.441, 2.987, -1.177, -0.367], ARM_SPEED, 0, 0, 1)

    gripper.gripper_release()
    time.sleep(0.5)  # 等待夹爪打开
    arm.rm_movej_p([0.199, -0.246, 0.441, 2.618, -1.117, 0.037], ARM_SPEED, 0, 0, 1)
    arm.rm_movej_p([0.199, -0.246, 0.441, -2.878, -1.167, -0.819], ARM_SPEED, 0, 0, 1)

    # 回到初始位置
    arm.rm_movej_p([-0.263, -0.0001, -0.238, 3.141, -0.028, 3.141], ARM_SPEED, 0, 0, 1)
    # arm.rm_movec

    print("=========Grasping sequence completed!==========")

import pdb

def main():
    print("Initializing vision detector...")
    try:
        detector = GraspDetector()
        detector.start_detection()
    except Exception as e:
        print(f"Failed to initialize vision detector: {e}")
        return

    robot_ip = get_ip()
    if not robot_ip:
        raise RuntimeError('无法找到机械臂IP，请检查网络连接')
    arm = connect_arm(robot_ip)

    try:
        ret = arm.rm_change_tool_frame("Arm_Tip")
        print(f"切换到Arm_Tip工具坐标系返回码: {ret}")

        arm.rm_movej_p([-0.254, -0.00001, 0.091, 3.113, 0.0, -1.572], ARM_SPEED, 0, 0, 1)
        gripper = Gripper(arm)
        gripper.gripper_release()
        time.sleep(0.5)
        print("\n=========== System Ready =============")

        old_settings = termios.tcgetattr(sys.stdin)
        cached_ee_pose = None   # 缓存末端位姿，避免每帧都查询
        ee_pose_counter = 0     # 控制末端位姿查询频率
        frame_counter = 0       # 总帧计数器

        try:
            tty.setraw(sys.stdin.fileno())

            while True:
                if select.select([sys.stdin], [], [], MAIN_LOOP_SLEEP) == ([sys.stdin], [], []):
                    key = sys.stdin.read(1)
                    #db.set_trace()
                    if key == 's':
                        _, latest_pos = detector.get_latest()
                        if latest_pos is not None:
                            x, y, z = (v / 1000 for v in latest_pos)
                            print(f"\n[相机] 手套位置: [{x:.3f}, {y:.3f}, {z:.3f}] m")
                            print("Starting grasping sequence...")
                            execute_grasp_sequence(arm,gripper, x, y, z)
                        else:
                            print("\nNo glove detected! Please keep glove in view and try again.")
                    elif key == 'q':
                        break

                _, latest_pos = detector.get_latest()
                frame_counter += 1
                if frame_counter % DISPLAY_EVERY_N_FRAMES == 0:
                    if latest_pos is not None:
                        x, y, z = (v / 1000 for v in latest_pos)
                        # 每 DISPLAY_EVERY_N_FRAMES 帧刷新一次末端位姿缓存
                        ee_pose_counter += 1
                        if ee_pose_counter >= DISPLAY_EVERY_N_FRAMES:
                            state, pose = arm.rm_get_current_arm_state()
                            if state == 0:
                                cached_ee_pose = pose["pose"]
                            ee_pose_counter = 0

                        if cached_ee_pose is not None:
                            base_pos = convert_camera_to_base(x, y, z, cached_ee_pose)
                            print(
                                f"\r[相机] [{x:.3f}, {y:.3f}, {z:.3f}]  "
                                f"[基座] [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]",
                                end="", flush=True
                            )
                        else:
                            print(f"\r[相机] [{x:.3f}, {y:.3f}, {z:.3f}]  [基座] 初始化中...", end="", flush=True)
                    else:
                        print(f"\rNo glove detected yet", end="", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted...")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    except Exception as e:
        print(f"Error during operation: {e}")
    finally:
        print("\nShutting down...")
        detector.stop()
        arm.rm_delete_robot_arm()


if __name__ == '__main__':
    main()