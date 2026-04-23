import cv2
import pyorbbecsdk as ob
from utils import frame_to_bgr_image
import time
import os

def main():
    # 初始化Orbbec相机
    pipeline = ob.Pipeline()
    config = ob.Config()

    # 获取颜色流配置
    color_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    if color_profiles is None:
        print("No color profiles found")
        return

    # 选择640x480分辨率的RGB/BGR配置
    color_profile = None
    for profile in color_profiles:
        if profile.get_format() in [ob.OBFormat.RGB, ob.OBFormat.BGR] and profile.get_width() == 640 and profile.get_height() == 480:
            color_profile = profile
            break

    if color_profile is None:
        print("No suitable color profile found for 640x480")
        return

    # 启用颜色流
    config.enable_stream(color_profile)

    # 开始管道
    pipeline.start(config)

    # 获取分辨率
    width = color_profile.get_width()
    height = color_profile.get_height()
    fps = color_profile.get_fps()

    # 创建rgbdate文件夹如果不存在
    output_dir = os.path.expanduser("~/rgbdate")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成文件名，使用时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{timestamp}.avi")

    # 创建VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Starting recording to {output_path}")
    print("Press 'q' to stop recording and exit")

    try:
        while True:
            # 获取帧
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            # 转换为BGR图像
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                continue

            # 写入视频
            out.write(color_image)

            # 显示图像（可选）
            cv2.imshow('Recording', color_image)

            # 检查按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # 停止管道
        pipeline.stop()
        # 释放VideoWriter
        out.release()
        # 关闭窗口
        cv2.destroyAllWindows()
        print(f"Recording saved to {output_path}")

if __name__ == "__main__":
    main()