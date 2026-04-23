import json
import socket
import sys
import threading
import time

from libs.auxiliary import get_ip

import numpy as np
from scipy.spatial.transform import Rotation as R

import math
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e, rm_frame_t

# Import our vision detector
from vision_detector import OrbbecGroundingDINOGloveDetector

# 相机坐标系到机械臂末端坐标系的旋转矩阵和平移向量（手眼标定得到）
rotation_matrix = np.array([
    [-0.99531643,  0.09664755, -0.00210937],
    [-0.09513715, -0.98316625, -0.15599056],
    [-0.01714997, -0.15505928,  0.98775629]
])
translation_vector = np.array([0.00926601, 0.06804014, 0.07758337])


def send_cmd(client, cmd, get_pose=True):
    client.send(cmd.encode('utf-8'))

    if not get_pose:
        client.recv(1024)
        return True

    response = client.recv(4096).decode('utf-8')
    decoder = json.JSONDecoder()
    data_list = []
    index = 0

    while index < len(response):
        try:
            while index < len(response) and response[index].isspace():
                index += 1
            if index >= len(response):
                break
            obj, idx = decoder.raw_decode(response[index:])
            data_list.append(obj)
            index += idx
        except json.JSONDecodeError:
            break

    for data in reversed(data_list):
        if data.get('state') == 'current_arm_state':
            target_data = data
            break
    else:
        raise RuntimeError('未找到有效的机械臂状态响应')

    if target_data['arm_state']['err'] != [0]:
        raise RuntimeError(f"机械臂报错: {target_data['arm_state']['err']}")

    pose_raw = target_data['arm_state']['pose']
    pose_converted = [
        pose_raw[0] / 1000000,
        pose_raw[1] / 1000000,
        pose_raw[2] / 1000000,
        pose_raw[3] / 1000,
        pose_raw[4] / 1000,
        pose_raw[5] / 1000,
    ]
    return pose_converted


def set_base_frame(client):
    socket_command = '{"command":"set_change_work_frame","frame_name":"Base"}'
    send_cmd(client, socket_command, get_pose=False)


def get_end_effector_pose(client):
    socket_command = '{"command": "get_current_arm_state"}'
    return send_cmd(client, socket_command)


def connect_robot():
    robot_ip = get_ip()
    if not robot_ip:
        raise RuntimeError('无法找到机械臂IP，请检查网络连接')

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((robot_ip, 8080))
    set_base_frame(client)
    return client


def convert_camera_to_base(x, y, z, ee_pose):
    obj_camera_coordinates = np.array([x, y, z, 1.0])

    T_camera_to_end_effector = np.eye(4)
    T_camera_to_end_effector[:3, :3] = rotation_matrix
    T_camera_to_end_effector[:3, 3] = translation_vector

    position = ee_pose[:3]
    orientation = R.from_euler('xyz', ee_pose[3:], degrees=False).as_matrix()

    T_base_to_end_effector = np.eye(4)
    T_base_to_end_effector[:3, :3] = orientation
    T_base_to_end_effector[:3, 3] = position

    obj_end_effector_coordinates_homo = T_camera_to_end_effector.dot(obj_camera_coordinates)
    obj_base_coordinates_homo = T_base_to_end_effector.dot(obj_end_effector_coordinates_homo)
    return obj_base_coordinates_homo[:3].tolist()


def main():
    # Initialize vision detector
    print("Initializing vision detector...")
    try:
        detector = OrbbecGroundingDINOGloveDetector(
            text_prompt="a yellow gloves or a white gloves with thin black stripes.",
            conf_threshold=0.4,
            box_threshold=0.45,
            text_threshold=0.35
        )
        # Start background detection
        detector.start_detection()
    except Exception as e:
        print(f"Failed to initialize vision detector: {e}")
        return

    # 连接机械臂
    robot_ip = get_ip()
    if not robot_ip:
        raise RuntimeError('无法找到机械臂IP，请检查网络连接')
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(robot_ip, 8080)
    
    try:

        # 切换到Base工作坐标系，确保从原始坐标系开始计算
        ret = arm.rm_change_tool_frame("Arm_Tip")
        print(f"切换到Base工具坐标系返回码: {ret}")

        # 松开夹爪
        arm.rm_set_gripper_release(500, True, 2)
        
        # 到初始位姿
        arm.rm_movej_p([-0.254,-0.00001,0.091,3.113,0.0,-1.572], 20, 0, 0, 1)
        arm.rm_set_gripper_release(500, False, 2)

        print("\n=========== System Ready =============")

        # Main loop - wait for key press
        import select
        import termios
        import tty
        
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # Set terminal to raw mode for keypress detection
            tty.setraw(sys.stdin.fileno())
            
            while True:
                # Check if key is pressed (non-blocking)
                if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                    key = sys.stdin.read(1)
                    if key == 's':
                        # Get latest detection
                        latest_pos = detector.get_latest_position()
                        if latest_pos is not None:
                            x, y, z = (x / 1000 for x in latest_pos) # Convert mm to m for display
                            print(f"\nDetected glove at: [{x:.3f}, {y:.3f}, {z:.3f}] m")
                            print("Starting grasping sequence...")
                            
                            # 执行抓取序列
                            execute_grasp_sequence(arm, x, y, z)
                        else:
                            print("\nNo glove detected! Please keep glove in view and try again.")
                    elif key == 'q':
                        break
                    # Ignore other keys
                
                # Get latest detection for display
                latest_pos = detector.get_latest_position()
                
                # Display status
                if latest_pos is not None:
                    x, y, z = (x / 1000 for x in latest_pos) 
                    print(f"\rLatest detection: [{x:.3f}, {y:.3f}, {z:.3f}] m", end="", flush=True)
                else:
                    print(f"\rNo glove detected yet", end="", flush=True)
                    
        except KeyboardInterrupt:
            print("\nInterrupted...")
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    except Exception as e:
        print(f"Error during operation: {e}")
    finally:
        # Clean up
        print("\nShutting down...")
        detector.stop()
        arm.rm_delete_robot_arm()


def execute_grasp_sequence(arm, x, y, z):
    """Execute the complete grasping sequence"""
    try:
        # 为第二次抓取准备，切换回Arm_Tip工具坐标系
        arm.rm_change_tool_frame("Arm_Tip")
        arm.rm_set_gripper_release(500, True, 1)

        # 获取当前末端位姿（原始工具坐标系）
        state, pose = arm.rm_get_current_arm_state()
        if state != 0:
            raise RuntimeError(f"获取机械臂状态失败, 错误码: {state}")
        print(f'当前工具位姿:\n {pose["pose"]}')

        ee_pose = pose['pose']
        base_pos = convert_camera_to_base(x, y, z, ee_pose)
        print(f'物体在机械臂基座坐标系下的坐标:\n {base_pos}')

        # 设置/切换工具坐标系
        tool_name = "tcp_offset"
        tcp_pose = [0, 0, 0.33, 0, 0, math.pi/2]  # z轴+0.33m，绕z轴逆时针90°
        frame = rm_frame_t(tool_name, tcp_pose, 1, 0, 0, 0)
        try:
            ret = arm.rm_set_manual_tool_frame(frame)
        except Exception as e:
            print(f"设置工具坐标系异常: {e}")
        ret = arm.rm_change_tool_frame(tool_name)

        # 再次获取当前末端姿态（此时已切换工具坐标系）
        state2, pose2 = arm.rm_get_current_arm_state()
        if state2 != 0:
            raise RuntimeError(f"获取机械臂状态失败, 错误码: {state2}")
        ee_pose2 = pose2['pose']

        # 运动到目标点，保持当前姿态，只改位置
        target_pose = list(base_pos) + [3.141, -0.222, 3.082]
        print(f'目标TCP位姿: {target_pose}')
        move_ret = arm.rm_movej_p(target_pose, 20, 0, 0, True)
        print(f'运动到目标返回码: {move_ret}')

        # 第一次夹取
        # arm.rm_set_gripper_pick(500, 100, False, 1)

        # 修正目标位姿
        target_pose[2] = -0.532   # 固定z为-0.530
        target_pose[1] -= 0.04   # y减去0.05
        # target_pose[0] += 0.04   # x减去0.03
        last_pose = list(target_pose)

        arm.rm_movej_p(last_pose, 20, 0, 0, 1)
        
        # 抓取物体（关闭夹爪）
        arm.rm_set_gripper_pick(500, 100, True, 1)
        arm.rm_set_gripper_pick(500, 100, True, 1)

        # 回到初始位姿（工具坐标系）
        arm.rm_movej_p([-0.263,-0.0001,-0.238,3.141,-0.028,3.141], 20, 0, 0, 1)
        # arm.rm_movej_p([-0.443, 0.038, 0.339, 3.13, -0.791, -3.038], 20, 0, 0, 1)
        # arm.rm_movej_p([0.193, -0.247, 0.734, -0.559, -1.331, -3.067], 20, 0, 0, 1)
        # arm.rm_movej_p([0.199, -0.246, 0.441, 2.987, -1.177, -0.367], 20, 0, 0, 1)

        # 晃动末端确保物品掉落
        # arm.rm_moves([-0.443, 0.038, 0.339, 3.13, -0.791, -3.038], 20, 0, 1, 1)
        # arm.rm_moves([0.193, -0.247, 0.734, -0.559, -1.331, -3.067], 20, 0, 1, 1)
        # arm.rm_moves([0.199, -0.246, 0.441, 2.987, -1.177, -0.367], 20, 0, 1, 1)

        # 打开夹爪松开物体
        arm.rm_set_gripper_release(500, True, 1)
        arm.rm_set_gripper_release(500, True, 1)

        print("Grasping sequence completed!")
        
    except Exception as e:
        print(f"Error during grasping sequence: {e}")
        raise


if __name__ == '__main__':
    main()
