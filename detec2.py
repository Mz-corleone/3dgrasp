import cv2
import numpy as np
import time

import pyorbbecsdk as ob
from utils import frame_to_bgr_image
from ultralytics import YOLO


def _get_hw_d2c_config(pipeline: ob.Pipeline):
    config = ob.Config()

    color_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    if color_profiles is None:
        print("No color profiles")
        return None

    for color_profile in color_profiles: # 遍历彩色流配置，寻找支持硬件对齐的配置
        if color_profile.get_format() != ob.OBFormat.RGB and color_profile.get_format() != ob.OBFormat.BGR:
            continue

        hw_d2c_profile_list = pipeline.get_d2c_depth_profile_list(
            color_profile, ob.OBAlignMode.HW_MODE
        )
        if not hw_d2c_profile_list or len(hw_d2c_profile_list) == 0:
            continue

        hw_d2c_profile = hw_d2c_profile_list[0]
        config.enable_stream(hw_d2c_profile) # 使能深度流
        config.enable_stream(color_profile) # 使能彩色流
        config.set_align_mode(ob.OBAlignMode.HW_MODE) # 使能硬件对齐模式
        return config

    print("No HW D2C profile matched")
    return None


class YOLO2DDetector:
    def __init__(self, model_name='glove', conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.Glove_class_id = 0

        self.pipeline = ob.Pipeline()
        self.config = _get_hw_d2c_config(self.pipeline)
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
            (depth_frame.get_height(), depth_frame.get_width())
        )

        if not hasattr(self, '_resolution_printed'):
            print(f'RGB resolution: {color_image.shape[1]}x{color_image.shape[0]}')
            print(f'Depth resolution: {depth_frame.get_width()}x{depth_frame.get_height()}')
            self._resolution_printed = True

        scale = depth_frame.get_depth_scale()
        depth_mm = depth_data.astype(np.float32) * scale

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

    def visualize(self, image, depth_image, detections, fps=0.0):
        overlay = image.copy()
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))

            cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            text = f'Glove {conf:.2f}'
            ty = y1i - 10 if y1i > 20 else y2i + 20
            cv2.putText(overlay, text, (x1i, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        cv2.putText(overlay, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        depth_colormap = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_colormap.astype(np.uint8), cv2.COLORMAP_JET)
        if depth_colormap.shape[:2] != overlay.shape[:2]:
            depth_colormap = cv2.resize(depth_colormap,
                                        (overlay.shape[1], overlay.shape[0]))

        stacked = np.hstack((overlay, depth_colormap))
        cv2.imshow('2D Glove Detection (Orbbec + YOLO)', stacked)

    def run(self):
        try:
            prev_time = time.time()
            fps = 0.0
            while True:
                color_image, depth_image = self.get_frames()
                if color_image is None or depth_image is None:
                    continue

                detections = self.detect_Gloves(color_image)

                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
                prev_time = curr_time

                self.visualize(color_image, depth_image, detections, fps)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
        finally:
            self.stop()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = YOLO2DDetector(model_name='glove_v2', conf_threshold=0.5)
    detector.run()
