# ****************************************************************************
# 3D Glove Detection with Orbbec HW D2C + GroundingDINO
# Detects yellow and white gloves with thin black stripes using GroundingDINO
# ****************************************************************************

import sys
import os
import time
import threading
import numpy as np

# Offline environment settings
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import torch
import pyorbbecsdk as ob
from PIL import Image

from groundingdino.util.inference import load_model, predict
from groundingdino.datasets import transforms as T
from utils import frame_to_bgr_image


def get_hw_d2c_config(pipeline: ob.Pipeline):
    config = ob.Config()

    color_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    if color_profiles is None:
        print("No color profiles")
        return None

    for color_profile in color_profiles:
        if color_profile.get_format() not in (ob.OBFormat.RGB, ob.OBFormat.BGR):
            continue

        hw_d2c_profile_list = pipeline.get_d2c_depth_profile_list(color_profile, ob.OBAlignMode.HW_MODE)
        if not hw_d2c_profile_list:
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


class OrbbecGroundingDINOGloveDetector:
    def __init__(self, text_prompt, conf_threshold, box_threshold=0.35, text_threshold=0.25,
                 config_path=None, weight_path=None):
        self.text_prompt = text_prompt
        self.conf_threshold = conf_threshold
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Initialize Orbbec camera
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

        # Load GroundingDINO model
        if config_path is None or weight_path is None:
            import groundingdino
            groundingdino_path = groundingdino.__path__[0]
            config_path = os.path.join(groundingdino_path, "config", "GroundingDINO_SwinT_OGC.py")
            weight_path = "/home/mxzboe/.cache/autodistill/groundingdino/groundingdino_swint_ogc.pth"

        print(f"Loading GroundingDINO model...")
        print(f"Config: {config_path}")
        print(f"Weight: {weight_path}")

        self.model = load_model(config_path, weight_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("GroundingDINO model loaded successfully")

        # Thread-safe position storage
        self.latest_position = None
        self.position_lock = threading.Lock()
        self.running = False
        self.detection_thread = None

    def stop(self):
        self.running = False
        if self.detection_thread and threading.current_thread() != self.detection_thread:
            self.detection_thread.join(timeout=1.0)
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

        scale = depth_frame.get_depth_scale()
        depth_mm = depth_data.astype(np.float32) * scale

        return color_image, depth_mm

    def detect_gloves(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        image_tensor, _ = transform(image_pil, None)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device
        )

        if len(boxes) > 0:
            best_idx = torch.argmax(logits)
            best_conf = float(logits[best_idx])

            if best_conf > self.conf_threshold:
                box = boxes[best_idx]
                h, w = image.shape[:2]
                x_center, y_center, width, height = box

                x1 = (x_center - width / 2) * w
                y1 = (y_center - height / 2) * h
                x2 = (x_center + width / 2) * w
                y2 = (y_center + height / 2) * h

                return [(x1, y1, x2, y2, best_conf, "glove")]

        return []

    def get_3d_position(self, bbox, depth_image):
        x1, y1, x2, y2, conf, label = bbox
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
        result = align_point_2d_to_3d(cx, cy, z, self.depth_intrinsic)
        if result is None:
            return None
        # Convert result from mm to m
        return (result[0], result[1], result[2])

    def detect_loop(self):
        """Continuous detection loop running in background thread"""
        try:
            prev_time = time.time()
            fps = 0.0
            while self.running:
                color_image, depth_image = self.get_frames()
                if color_image is None or depth_image is None:
                    continue

                detections = self.detect_gloves(color_image)
                positions = [self.get_3d_position(d, depth_image) for d in detections] if detections else []

                # Update latest position (use first detection if multiple)
                if positions and positions[0] is not None:
                    with self.position_lock:
                        self.latest_position = positions[0]  # (x, y, z) in meters

                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
                prev_time = curr_time

                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)

        except Exception as e:
            print(f"Error in detection loop: {e}")
        finally:
            # Ensure cleanup
            self.stop()

    def start_detection(self):
        """Start the background detection thread"""
        if not self.running:
            self.running = True
            self.detection_thread = threading.Thread(target=self.detect_loop, daemon=True)
            self.detection_thread.start()
            print("Vision detection started in background")

    def get_latest_position(self):
        """Get the latest detected 3D position (thread-safe)"""
        with self.position_lock:
            return self.latest_position

    def is_detecting(self):
        """Check if detection is running"""
        return self.running


def main():
    # Detection parameters
    TEXT_PROMPT = "a yellow gloves or a white gloves with thin black stripes."
    CONF_THRESHOLD = 0.4
    BOX_THRESHOLD = 0.45
    TEXT_THRESHOLD = 0.35

    # Model paths (None for auto-detect)
    config_path = None
    weight_path = None

    # Parse command line arguments
    if len(sys.argv) > 1:
        weight_path = sys.argv[1]
    if len(sys.argv) > 2:
        config_path = sys.argv[2]

    detector = OrbbecGroundingDINOGloveDetector(
        text_prompt=TEXT_PROMPT,
        conf_threshold=CONF_THRESHOLD,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        config_path=config_path,
        weight_path=weight_path
    )
    detector.run()


def frame_to_bgr_image_fallback(frame):
    """Fallback if utils not available"""
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == ob.OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == ob.OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return None
    return image


if __name__ == '__main__':
    main()