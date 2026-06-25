import json
import socket
import sys
import time
import threading
import select
import math
#临时测试用，键盘输入
try:
    import termios
    import tty
    POSIX_TERMIOS = True
except Exception:
    POSIX_TERMIOS = False

try:
    import msvcrt
    HAS_MSVCRT = True
except Exception:
    HAS_MSVCRT = False

from libs.auxiliary import get_ip
import numpy as np
from scipy.spatial.transform import Rotation as R

from Robotic_Arm.rm_robot_interface import *
from detec import YOLO2DDetector


class GraspController:
    MODEL_NAME = 'glove_v2'
    CONF_THRESHOLD = 0.5
    ARM_SPEED = 60
    DETECTION_SLEEP = 0.01
    MAIN_LOOP_SLEEP = 0.1
    DISPLAY_EVERY_N_FRAMES = 10
    ARM_CONNECT_TIMEOUT = 5.0
    ARM_RECV_TIMEOUT = 5.0
    OBJ_HIGHT = -0.539   #534, 切割过的这台用539,针对不同的机器需要微调

    ROTATION_MATRIX = np.array([
        [-0.99531643,  0.09664755, -0.00210937],
        [-0.09513715, -0.98316625, -0.15599056],
        [-0.01714997, -0.15505928,  0.98775629]
    ])
    TRANSLATION_VECTOR = np.array([0.00926601, 0.06804014, 0.07758337])

    def __init__(self, model_name=MODEL_NAME, conf_threshold=CONF_THRESHOLD):
        self._detector = YOLO2DDetector(model_name=model_name, conf_threshold=conf_threshold)
        self._latest_detection = None
        self._latest_position = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.client = None
        self.arm = None

    # ─── 3D Position ────────────────────────────────────────

    @staticmethod
    def _align_point_2d_to_3d(x, y, depth_mm, intrinsic):
        if depth_mm <= 0 or intrinsic.fx == 0 or intrinsic.fy == 0:
            return None
        px = (x - intrinsic.cx) * depth_mm / intrinsic.fx
        py = (y - intrinsic.cy) * depth_mm / intrinsic.fy
        pz = depth_mm
        return (px, py, pz)

    def get_3d_position(self, bbox, depth_image):
        x1, y1, x2, y2, conf, cls = bbox
        h, w = depth_image.shape[:2]

        bbox_w = x2 - x1
        bbox_h = y2 - y1
        half = int(min(max(min(bbox_w, bbox_h) / 4, 5), 20))

        cx_c = (x1 + x2) / 2.0
        cy_c = (y1 + y2) / 2.0
        points = [
            (cx_c, cy_c),
            (cx_c, (y1 + 3 * y2) / 4.0),
            (cx_c, (3 * y1 + y2) / 4.0),
            ((x1 + 3 * x2) / 4.0, cy_c),
            ((3 * x1 + x2) / 4.0, cy_c),
        ]

        depths = []
        for px, py in points:
            cx = int(np.clip(px, 0, w - 1))
            cy = int(np.clip(py, 0, h - 1))
            x_start = max(0, cx - half)
            y_start = max(0, cy - half)
            x_end = min(w, cx + half + 1)
            y_end = min(h, cy + half + 1)

            roi = depth_image[y_start:y_end, x_start:x_end]
            if roi.size == 0:
                continue

            valid = roi[(roi > 0) & (roi < 5000.0)]
            if valid.size == 0:
                valid = roi[roi > 0]
            if valid.size == 0:
                continue

            med = np.median(valid)
            std = np.std(valid)
            if std > 0:
                filtered = valid[np.abs(valid - med) < 1.5 * std]
                if filtered.size > 0:
                    depths.append(float(np.median(filtered)))
                else:
                    depths.append(float(med))
            else:
                depths.append(float(med))

        if not depths:
            return None

        z = float(np.median(depths))
        cx = int(np.clip(cx_c, 0, w - 1))
        cy = int(np.clip(cy_c, 0, h - 1))
        return self._align_point_2d_to_3d(cx, cy, z,
                                          self._detector.rgb_intrinsic)

    # ─── Arm Socket Communication ───────────────────────────

    def _send_with_timeout(self, client, data: bytes, timeout: float) -> bytes:
        client.settimeout(timeout)
        client.send(data)
        return client.recv(8192)

    def send_cmd(self, client, cmd, get_pose=True):
        response = self._send_with_timeout(client, cmd.encode('utf-8'),
                                           self.ARM_RECV_TIMEOUT)
        if not get_pose:
            return True

        decoded = response.decode('utf-8')
        try:
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

    def set_base_frame(self, client):
        self.send_cmd(client, '{"command":"set_change_work_frame","frame_name":"Base"}',
                      get_pose=False)

    def get_end_effector_pose(self, client):
        return self.send_cmd(client, '{"command": "get_current_arm_state"}')

    def connect_robot(self):
        robot_ip = get_ip()
        if not robot_ip:
            raise RuntimeError('无法找到机械臂IP，请检查网络连接')

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(self.ARM_CONNECT_TIMEOUT)
        client.connect((robot_ip, 8080))
        self.set_base_frame(client)
        self.client = client
        return client

    def connect_arm(self, robot_ip: str):
        arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        arm.rm_create_robot_arm(robot_ip, 8080)
        self.arm = arm
        return arm

    # ─── Coordinate Transform ───────────────────────────────

    def convert_camera_to_base(self, x, y, z, ee_pose):
        obj_camera = np.array([x, y, z, 1.0])

        T_cam_to_ee = np.eye(4)
        T_cam_to_ee[:3, :3] = self.ROTATION_MATRIX
        T_cam_to_ee[:3, 3] = self.TRANSLATION_VECTOR

        position = np.array(ee_pose[:3])
        orientation = R.from_euler('xyz', ee_pose[3:], degrees=False).as_matrix()

        T_base_to_ee = np.eye(4)
        T_base_to_ee[:3, :3] = orientation
        T_base_to_ee[:3, 3] = position

        obj_base = T_base_to_ee @ T_cam_to_ee @ obj_camera
        return obj_base[:3].tolist()

    # ─── Gripper ────────────────────────────────────────────

    def gripper_release(self):
        self.arm.rm_set_modbus_mode(1, 115200, 2)
        write_params = rm_peripheral_read_write_params_t(1, 10, 1)
        for _ in range(5):
        #port:1代表机械臂末端485接口 0代表控制器485接口 3代表控制器ModbusTCP设备；Modbus寄存器地址5；Modbus设备ID    
            ret = self.arm.rm_write_single_register(write_params, 1000) # 1000对应打开    
            if ret == 0:
                ret = self.arm.rm_write_single_register(write_params, 1000)  
                break
        time.sleep(0.1)
        self.arm.rm_close_modbus_mode(1)


    def gripper_pick(self):
        self.arm.rm_set_modbus_mode(1, 115200, 2)
        write_params = rm_peripheral_read_write_params_t(1, 10, 1)
        for _ in range(5):
            ret = self.arm.rm_write_single_register(write_params, 10) # 10对应闭合    
            if ret == 0:
                ret = self.arm.rm_write_single_register(write_params, 10) 
                break
        time.sleep(0.1)
        self.arm.rm_close_modbus_mode(1)

    # ─── Detection Thread ───────────────────────────────────

    def start_detection(self):
        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()

    def _detection_loop(self):
        while self._running:
            color_image, depth_image = self._detector.get_frames()
            if color_image is None or depth_image is None:
                time.sleep(self.DETECTION_SLEEP)
                continue

            detections = self._detector.detect_Gloves(color_image)
            with self._lock:
                if detections:
                    self._latest_detection = detections[0]
                    self._latest_position = self.get_3d_position(
                        self._latest_detection, depth_image
                    )
                else:
                    self._latest_detection = None
                    self._latest_position = None

            time.sleep(self.DETECTION_SLEEP)

    def get_latest(self):
        with self._lock:
            return self._latest_detection, self._latest_position

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._detector.stop()

    # ─── Grasp Logic ────────────────────────────────────────

    def adjust_target_position(self, target_pose, scale=0.8):
        state, pose = self.arm.rm_get_current_arm_state()
        if state != 0:
            return target_pose

        current_pos = pose['pose'][:3]
        adjusted = list(target_pose)

        for i in range(2):
            adjusted[i] = current_pos[i] + (target_pose[i] - current_pos[i]) * scale
        adjusted[2] = target_pose[2]
        return adjusted

    def _retry_or_skip(self, target_pose):
        print(f"运动到目标位置失败，尝试进一步调整...")
        target_pose = self.adjust_target_position(target_pose)
        move_ret = self.arm.rm_movej_p(target_pose, self.ARM_SPEED, 0, 0, True)
        if move_ret != 0:
            print("无法运动到目标位置附近，跳过")
            self.arm.rm_change_tool_frame("Arm_Tip")
            return None
        return target_pose

    def execute_grasp_sequence(self, x, y, z):
        self.arm.rm_change_tool_frame("Arm_Tip")
        self.gripper_release()  #整机部署时可删可不删

        state, pose = self.arm.rm_get_current_arm_state()
        if state != 0:
            raise RuntimeError(f"获取机械臂状态失败, 错误码: {state}")
        print(f'当前工具位姿:\n {pose["pose"]}')

        ee_pose = pose['pose']
        base_pos = self.convert_camera_to_base(x, y, z, ee_pose)
        print(f'物体在基座下的坐标:\n {base_pos}')

        tool_name = "tcp_offset"
        tcp_pose = [0, 0, 0.33, 0, 0, math.pi/2]
        frame = rm_frame_t(tool_name, tcp_pose, 1, 0, 0, 0)
        try:
            self.arm.rm_set_manual_tool_frame(frame)
        except Exception as e:
            print(f"设置工具坐标系异常: {e}")
        self.arm.rm_change_tool_frame(tool_name)

        # base_pos[1] = base_pos[1] * 0.75
        # base_pos[2] = -0.530

        distance = math.sqrt(base_pos[0]**2 + base_pos[1]**2)
        if distance < 0.23:
            target_pose = list(base_pos) + [-3.109, 0.117, 3.069]
            move_ret = self.arm.rm_movej_p(target_pose, self.ARM_SPEED, 0, 0, True)
            print(f'运动到目标返回码: {move_ret}')
            if move_ret != 0:
                target_pose = self._retry_or_skip(target_pose)
                if target_pose is None:
                    return False

        elif distance > 0.38:
            target_pose = list(base_pos) + [-3.141, -0.222, -3.092]
            move_ret = self.arm.rm_movej_p(target_pose, self.ARM_SPEED, 0, 0, True)
            print(f'运动到目标返回码: {move_ret}')
            if move_ret != 0:
                target_pose = self._retry_or_skip(target_pose)
                if target_pose is None:
                    return False

        else:
            target_pose = list(base_pos) + [-3.141, -0.222, -3.092]
            move_ret = self.arm.rm_movej_p(target_pose, self.ARM_SPEED, 0, 0, True)
            print(f'运动到目标返回码: {move_ret}')
            if move_ret != 0:
                target_pose = self._retry_or_skip(target_pose)
                if target_pose is None:
                    return False

        if distance > 0.38:
            # off = (-int(self.OBJ_HIGHT * 1000)) % 10
            # target_pose[2] = self.OBJ_HIGHT + off/3 * 0.001
            target_pose[2] = self.OBJ_HIGHT - 0.002
        else:
            target_pose[2] = self.OBJ_HIGHT
        target_pose[1] -= 0.015
        target_pose[0] -= 0.01
        last_pose = list(target_pose)

        self.arm.rm_movej_p(last_pose, self.ARM_SPEED, 0, 0, 1)
        self.gripper_pick()
        time.sleep(3)

        self.arm.rm_movej_p([-0.443, 0.038, 0.339, 3.13, -0.791, -3.038],
                            self.ARM_SPEED, 0, 0, 1)
        self.arm.rm_movej_p([0.199, -0.246, 0.441, 2.987, -1.177, -0.367],
                            self.ARM_SPEED, 0, 0, 1)

        self.gripper_release() # 释放夹爪
        
        #旋转平移夹爪，增加物体掉落概率（新增点位）
        self.arm.rm_movej([-0.108,24.719,-40.406,147.014,118.97,-76.026], 
                              self.ARM_SPEED + 20, 0, 0, 1)
        self.arm.rm_movej_p([0.199, -0.246, 0.441, 2.987, -1.177, -0.367],
                            self.ARM_SPEED + 20, 0, 0, 1) 
        
        #回到初始状态
        self.arm.rm_movej_p([-0.263, -0.0001, -0.238, 3.141, -0.028, 3.141],
                            self.ARM_SPEED + 10, 0, 0, 1)

        self.arm.rm_change_tool_frame("Arm_Tip")
        print("=========Grasping sequence completed!==========")
        return True

    # ─── Main Loop ──────────────────────────────────────────

    def run(self):
        print("Initializing vision detector...")
        try:
            self.start_detection()
        except Exception as e:
            print(f"Failed to initialize vision detector: {e}")
            return

        robot_ip = get_ip()
        if not robot_ip:
            raise RuntimeError('无法找到机械臂IP，请检查网络连接')
        arm = self.connect_arm(robot_ip)

        try:
            ret = arm.rm_change_tool_frame("Arm_Tip")
            print(f"切换到Arm_Tip工具坐标系返回码: {ret}")

            arm.rm_movej_p([-0.254, -0.00001, 0.091, 3.113, 0.0, -1.572],
                           self.ARM_SPEED, 0, 0, 1)
           
            print("\n=========== System Ready =============")

            frame_counter = 0
            old_settings = None
            if POSIX_TERMIOS:
                try:
                    old_settings = termios.tcgetattr(sys.stdin)
                    tty.setraw(sys.stdin.fileno())
                except Exception:
                    old_settings = None

            try:
                while True:
                    key = None
                    # Windows: use msvcrt for non-blocking keyboard
                    if HAS_MSVCRT:
                        if msvcrt.kbhit():
                            try:
                                k = msvcrt.getwch()
                            except Exception:
                                k = msvcrt.getch().decode('utf-8', errors='ignore')
                            key = k
                    else:
                        # POSIX fallback using select
                        if select.select([sys.stdin], [], [], self.MAIN_LOOP_SLEEP) == ([sys.stdin], [], []):
                            key = sys.stdin.read(1)

                    if key is not None:
                        if key == 's':
                            _, latest_pos = self.get_latest()
                            if latest_pos is not None:
                                x, y, z = (v / 1000 for v in latest_pos)
                                print(f"\n[相机] 手套位置: [{x:.3f}, {y:.3f}, {z:.3f}] m")
                                print("Starting grasping sequence...")
                                first_success = self.execute_grasp_sequence(x, y, z)
                                if first_success:
                                    time.sleep(0.5)
                                    _, latest_pos2 = self.get_latest()
                                    if latest_pos2 is not None:
                                        x2, y2, z2 = (v / 1000 for v in latest_pos2)
                                        print(f"\n[相机] 再次检测到手套，开始第二次抓取: [{x2:.3f}, {y2:.3f}, {z2:.3f}] m")
                                        self.execute_grasp_sequence(x2, y2, z2)
                                    else:
                                        print("\n没有检测到第二次手套，停止再次抓取。")
                            else:
                                print("\nNo glove detected! Please keep glove in view and try again.")
                        elif key == 'q':
                            break

                    # 如果使用 msvcrt，则在没有按键时也需要短暂睡眠
                    if HAS_MSVCRT:
                        time.sleep(self.MAIN_LOOP_SLEEP)

                    _, latest_pos = self.get_latest()
                    frame_counter += 1
                    if frame_counter % self.DISPLAY_EVERY_N_FRAMES == 0:
                        if latest_pos is not None:
                            x, y, z = (v / 1000 for v in latest_pos)
                            state, pose = arm.rm_get_current_arm_state()
                            if state == 0:
                                ee_pose = pose["pose"]
                                base_pos = self.convert_camera_to_base(x, y, z, ee_pose)
                                print(
                                    f"\r[相机] [{x:.3f}, {y:.3f}, {z:.3f}]  "
                                    f"[基座] [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]",
                                    end="", flush=True
                                )
                            else:
                                print(f"\r[相机] [{x:.3f}, {y:.3f}, {z:.3f}]  [基座] 查询失败...",
                                      end="", flush=True)
                        else:
                            print(f"\rNo glove detected yet", end="", flush=True)
            except KeyboardInterrupt:
                print("\nInterrupted...")
            finally:
                if POSIX_TERMIOS and old_settings is not None:
                    try:
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    except Exception:
                        pass

        except Exception as e:
            print(f"Error during operation: {e}")
        finally:
            print("\nShutting down...")
            self.stop()
            arm.rm_delete_robot_arm()


if __name__ == '__main__':
    controller = GraspController()
    controller.run()