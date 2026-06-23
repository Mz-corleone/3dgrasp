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

        # ROI 半径根据 bbox 大小自适应，最小 5，最大 20
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        half = int(min(max(min(bbox_w, bbox_h) / 4, 5), 20))

        # 中心 + 四个 1/4 位置采样点
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

            # 过滤无效深度: 排除 0 和过远值 (>5000mm)
            valid = roi[(roi > 0) & (roi < 5000.0)]
            if valid.size == 0:
                # 如果 < 5000mm 过滤后无数据，只排除 0
                valid = roi[roi > 0]
            if valid.size == 0:
                continue

            # 去除异常值: 保留 1.5σ 范围内
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

        # 所有采样点深度取中位数
        z = float(np.median(depths))
        cx = int(np.clip(cx_c, 0, w - 1))
        cy = int(np.clip(cy_c, 0, h - 1))
        return align_point_2d_to_3d(cx, cy, z, self.depth_intrinsic)

    def visualize(self, image, depth_image, detections, positions, fps=0.0):
        overlay = image.copy()
        for det, pos in zip(detections, positions):
            x1, y1, x2, y2, conf, cls = det
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))

            cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            if pos is not None:
                px, py, pz = (x / 1000 for x in pos) #mm2m
                text = f'Glove {conf:.2f} ({px:.3f},{py:.3f},{pz:.3f}m)'
                ty = y1i - 10 if y1i > 20 else y2i + 20
                cv2.putText(overlay, text, (x1i, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
