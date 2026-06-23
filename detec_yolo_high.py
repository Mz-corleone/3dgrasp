# ****************************************************************************
# 3D Glove Detection with Orbbec HW D2C + YOLOv8
# ****************************************************************************

import cv2
import numpy as np
import time

import pyorbbecsdk as ob
from utils import frame_to_bgr_image
from ultralytics import YOLO


def get_hw_d2c_config(pipeline: ob.Pipeline):
    config = ob.Config()

    color_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    if color_profiles is None:
        print("No color profiles")
        return None

    for color_profile in color_profiles:
        # 要求 RGB / BGR RGB 摄像头
        if color_profile.get_format() != ob.OBFormat.RGB and color_profile.get_format() != ob.OBFormat.BGR:
            continue

        hw_d2c_profile_list = pipeline.get_d2c_depth_profile_list(color_profile, ob.OBAlignMode.HW_MODE)
        if not hw_d2c_profile_list or len(hw_d2c_profile_list) == 0:
            continue

        hw_d2c_profile = hw_d2c_profile_list[0]
        config.enable_stream(hw_d2c_profile)
        config.enable_stream(color_profile)
        config.set_align_mode(ob.OBAlignMode.HW_MODE)
        return config

    print("No HW D2C profile matched")
    return None


def align_point_2d_to_3d(x, y, depth_mm, intrinsic):
    if depth_mm <= 0 or intrinsic.fx == 0 or intrinsic.fy == 0:
        return None
    px = (x - intrinsic.cx) * depth_mm / intrinsic.fx
    py = (y - intrinsic.cy) * depth_mm / intrinsic.fy
    pz = depth_mm
    return (px, py, pz)


class OrbbecYOLOGloveDetector:
    def __init__(self, model_name, conf_threshold):
        self.conf_threshold = conf_threshold
        self.Glove_class_id = 0  # glove id：0

        self.pipeline = ob.Pipeline()
        self.config = get_hw_d2c_config(self.pipeline)
        if self.config is None:
            raise RuntimeError('No suitable HW D2C stream config found')

        self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)

        self.camera_param = self.pipeline.get_camera_param()
        if self.camera_param is None:
            raise RuntimeError('Failed to get camera param')

        self.depth_intrinsic = self.camera_param.depth_intrinsic
        self.rgb_intrinsic = self.camera_param.rgb_intrinsic

        self.model = YOLO(f'{model_name}.pt')

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None

        color_image = frame_to_bgr_image(color_frame)
        if color_image is None:
            return None, None

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
        (depth_frame.get_height(), depth_frame.get_width()))
    
        # 调试：打印缩放系数和深度数据范围
        scale = depth_frame.get_depth_scale()
        # print(f"Depth scale: {scale}")
        # print(f"Raw depth range: {depth_data.min()} - {depth_data.max()}")
        
        depth_mm = depth_data.astype(np.float32) * scale
        # print(f"Scaled depth range: {depth_mm.min()} - {depth_mm.max()}")
            
        return color_image, depth_mm

    def detect_Gloves(self, image):
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, ids):
                if cls_id == self.Glove_class_id:
                    x1, y1, x2, y2 = box
                    detections.append((x1, y1, x2, y2, float(conf), int(cls_id)))
        return detections

    def get_3d_position(self, bbox, depth_image):
        x1, y1, x2, y2, conf, cls = bbox
        h, w = depth_image.shape[:2]

        # 将 bbox 坐标转换为整数并限制在图像范围内
        x1i = int(np.clip(x1, 0, w - 1))
        y1i = int(np.clip(y1, 0, h - 1))
        x2i = int(np.clip(x2, 0, w - 1))
        y2i = int(np.clip(y2, 0, h - 1))

        # 提取整个 bbox 区域的深度值
        bbox_depth = depth_image[y1i:y2i, x1i:x2i]
        if bbox_depth.size == 0:
            return None

        # 将 bbox 划分为 2行 x 2列 = 4 个区域块
        rows, cols = 2, 2
        block_h = (y2i - y1i) // rows
        block_w = (x2i - x1i) // cols

        min_mean_depth = float('inf')
        best_center = None
        best_mean = None

        for r in range(rows):
            for c in range(cols):
                # 计算当前区域块的范围
                y_start = y1i + r * block_h
                y_end = y1i + (r + 1) * block_h if r < rows - 1 else y2i
                x_start = x1i + c * block_w
                x_end = x1i + (c + 1) * block_w if c < cols - 1 else x2i

                # 提取该区域的深度值
                block_depth = depth_image[y_start:y_end, x_start:x_end]
                if block_depth.size == 0:
                    continue

                # 过滤无效深度: 排除 0 和过远值 (>600mm)
                valid = block_depth[(block_depth > 0) & (block_depth < 600.0)]
                if valid.size == 0:
                    valid = block_depth[block_depth > 0]
                if valid.size == 0:
                    continue

                mean_depth = float(np.mean(valid))

                # 更新最小均值
                if mean_depth < min_mean_depth:
                    min_mean_depth = mean_depth
                    # 计算该区域块的中心点
                    cx = (x_start + x_end) // 2
                    cy = (y_start + y_end) // 2
                    best_center = (cx, cy)
                    best_mean = mean_depth

        if best_center is None or best_mean is None:
            return None

        cx, cy = best_center
        point_3d = align_point_2d_to_3d(cx, cy, best_mean, self.depth_intrinsic)
        if point_3d is None:
            return None
        return (point_3d, (cx, cy), (x1i, y1i, x2i, y2i, rows, cols))

    def visualize(self, image, depth_image, detections, positions, fps=0.0):
        overlay = image.copy()
        for det, pos in zip(detections, positions):
            x1, y1, x2, y2, conf, cls = det
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))

            cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            if pos is not None:
                (px, py, pz), (cx, cy), (bx1, by1, bx2, by2, rows, cols) = pos
                px, py, pz = (x / 1000 for x in (px, py, pz)) #mm2m
                text = f'Glove {conf:.2f} ({px:.3f},{py:.3f},{pz:.3f}m)'
                ty = y1i - 10 if y1i > 20 else y2i + 20
                cv2.putText(overlay, text, (x1i, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 画出 4 个区域块的网格
                block_h = (by2 - by1) // rows
                block_w = (bx2 - bx1) // cols
                for r in range(rows):
                    for c in range(cols):
                        y_start = by1 + r * block_h
                        y_end = by1 + (r + 1) * block_h if r < rows - 1 else by2
                        x_start = bx1 + c * block_w
                        x_end = bx1 + (c + 1) * block_w if c < cols - 1 else bx2
                        cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (255, 255, 0), 1)

                # 标记选中的区域块中心点（红色圆点）
                cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
                # 高亮选中的区域块（蓝色边框）
                sel_y_start = by1 + (cy - by1) // block_h * block_h
                sel_y_end = sel_y_start + block_h if (cy - by1) // block_h < rows - 1 else by2
                sel_x_start = bx1 + (cx - bx1) // block_w * block_w
                sel_x_end = sel_x_start + block_w if (cx - bx1) // block_w < cols - 1 else bx2
                cv2.rectangle(overlay, (sel_x_start, sel_y_start), (sel_x_end, sel_y_end), (255, 0, 0), 2)

        cv2.putText(overlay, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        depth_colormap = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_colormap.astype(np.uint8), cv2.COLORMAP_JET)
        if depth_colormap.shape[:2] != overlay.shape[:2]:
            depth_colormap = cv2.resize(depth_colormap, (overlay.shape[1], overlay.shape[0]))

        stacked = np.hstack((overlay, depth_colormap))
        cv2.imshow('3D Glove Detection (Orbbec + YOLO)', stacked)

    def run(self):
        try:
            prev_time = time.time()
            fps = 0.0
            while True:
                color_image, depth_image = self.get_frames()
                if color_image is None or depth_image is None:
                    continue

                detections = self.detect_Gloves(color_image)
                positions = [self.get_3d_position(d, depth_image) for d in detections]

                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
                prev_time = curr_time

                self.visualize(color_image, depth_image, detections, positions, fps)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break

        finally:
            self.stop()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = OrbbecYOLOGloveDetector(model_name='glove', conf_threshold=0.5)
    detector.run()
