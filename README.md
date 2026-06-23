# 3D Grasp Project

一个基于 Orbbec 相机、YOLO 手套检测和机械臂抓取的配合系统。项目主要包含两个运行模式：

- `detec.py`：二维手套检测与深度图可视化
- `grasp.py`：调用视觉检测 + 机械臂抓取流程

## 项目结构

- `detec.py`：使用 Orbbec 相机采集彩色和深度图，加载 `glove.pt` YOLO 模型进行手套检测，并在窗口中显示检测框与深度图。
- `grasp.py`：通过检测获取手套像素位置，计算 3D 坐标，转换到机械臂基座坐标系，执行抓取与放置动作。
- `utils.py`：包含 Orbbec 相机帧格式转换函数，将相机帧转换为可用于 OpenCV 的 BGR 图像。
- `libs/auxiliary.py`：检测机械臂 IP（固定为 `192.168.1.18` 和 `192.168.10.18`），并提供日志、数据文件夹等辅助功能。
- `requirements.txt`：项目依赖列表。
- `glove.pt`：YOLO 手套检测模型权重文件。

## 运行环境

建议使用 Python 3.10+（根据当前依赖的 `torch==2.12.0` 和 `ultralytics==8.4.51`）。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

### 1. 仅检测显示

运行相机与手套检测，可在屏幕上看到 RGB 图像、检测框和深度图：

```bash
python detec.py
```

按 `q` 或 `Esc` 退出。

### 2. 机械臂抓取流程

运行抓取主程序：

```bash
python grasp.py
```

程序启动后：

- 按 `s`：开始抓取当前检测到的手套
- 按 `q`：退出程序

程序会自动：

1. 初始化 Orbbec 摄像头和 YOLO 手套检测
2. 获取手套 2D 检测框和深度值
3. 计算 3D 位置并转换到机械臂基座坐标系
4. 执行机械臂运动与夹爪控制

## 依赖说明

当前 `requirements.txt` 包含：

- `opencv-python`
- `numpy`
- `torch`
- `torchvision`
- `ultralytics`
- `pyorbbecsdk2`
- `robotic-arm`
- `scipy`

## 注意事项

- 机械臂 IP 获取依赖 `libs/auxiliary.py` 中的固定 IP 地址，如果网络或 IP 不一致，请根据实际情况调整：
  - `192.168.1.18`
  - `192.168.10.18`

- 需要连接 Orbbec 相机以及支持 `pyorbbecsdk2` 的设备。
- 需要连接 `robotic-arm` SDK 支持的机械臂，并确保网络可达。

## 代码说明

### `detec.py`

主要功能：

- 初始化 Orbbec 相机流
- 获取彩色和深度帧
- 运行 YOLO 模型检测手套
- 将检测结果显示在窗口中

### `grasp.py`

主要功能：

- 使用视觉检测获取手套位置
- 对检测目标进行 3D 深度采样和坐标计算
- 将相机坐标转换为机械臂基座坐标系
- 通过 `robotic-arm` SDK 控制机械臂运动和夹爪开闭
- 提供键盘交互启动抓取

### `utils.py`

提供 Orbbec 彩色帧格式到 BGR 图像的转换，支持多种相机格式，例如：RGB、BGR、YUYV、MJPG、I420、NV12、NV21、UYVY。

## 额外建议

- 如果想用 `grasp.py` 正常运行，建议先确认 `detec.py` 检测功能正常。
- 如果需要复现或调试，先运行 `python detec.py` 检查相机和模型是否工作正常。