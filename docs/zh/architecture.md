# 架构

[English](../en/architecture.md) | [中文](architecture.md)

## 模块边界

```text
UniSim
  公共触觉场景、位姿、生命周期和字段契约
    ↓
UniTac
  模型选择与 HydroShear 实现
    ↓
HydroShear
  几何采样、深度、SDF、时间状态和位移
```

UniSim 拥有与后端无关的 `TactileSensor`、`TactileSceneDescription` 和 `TactileFieldSpec` 契约。UniTac 接收脱离后端的碰撞几何、传感器安装坐标系和世界位姿，不持有仿真后端。HydroShear 管理自身配置、CUDA 计算、缓冲区、reset、rebuild 和资源释放。`src/unitac/__init__.py` 用简单工厂分支选择模型。

## 数据流

1. `materialize(scene)` 校验物体和传感器身份，将支持的碰撞描述转换为 body 局部三角网格，采样物体表面并分配设备工作区。
2. `update_pose(body_pos, body_quat)` 更新缓存的世界位姿。位置单位为米，四元数使用单位 `wxyz` 顺序。
3. `advance()` 将物体采样点变换到各传感器坐标系，以毫米光栅化压入深度，将 marker 高度转换为米供 HydroSoft 使用，推进一次时间状态并写入 marker 位移场。
4. `read()` 返回最新位移场但不推进状态。`read_visualization_snapshot()` 显式复制诊断数据到 CPU，供可视化器使用。

## 生命周期与输出

调用者负责推进物理仿真，并调用 `materialize`、`update_pose`、`advance`、`read`、`reset` 和 `close`。几何发生变化时调用 `rebuild(scene)`。模型不接收后端对象，也不会把接触力反馈给仿真器。

`displacement` 字段为 float32 CUDA Torch 数据，形状为 `(environment, sensor, row, column, xyz)`。其 `TactileFieldSpec.units` 是 `hydroshear_model_units`：HydroSoft 的经验系数没有定义物理位移长度。`read()` 返回传感器拥有的缓冲区视图；下一次 advance、reset、rebuild 或 close 后该缓冲区可能被复用。

可视化快照包含以毫米表示的传感器局部 marker 坐标、以毫米表示且形状为 `(sensor, 175, 100)` 的相机压入深度，以及未缩放的 `displacement` 模型字段。GelSight Mini 渲染将深度毫米除以 `0.065 mm/pixel`，得到标定像素高度；再使用经验系数 `0.05 mm/model-unit` 将 HydroShear 模型单位映射为显示位移。这些显示值不是物理位移测量。只有显式创建可视化器时才会加载 RGB 标定数据和 OpenCV。

## 配置与设备

HydroShear 配置必须提供 `gel_mesh`：传感器局部坐标系中的闭合 `(triangles, 3, 3)` 三角网格，单位为米。`surface_points` 默认 2,048。`output_grid` 设置 marker 行列、间距、相机中心和像素比例。`solver` 包含 HydroSoft 参数及可选 FFT 或矩阵乘法路径。`device` 选择 CUDA 设备。

几何距离和深度计算要求 CUDA。NumPy 和 CPU Torch 位姿会复制到所选设备。CUDA Torch 位姿通过 DLPack 使用调用者当前 stream。普通 `read()` 保持在 CUDA；诊断快照和图像渲染会显式将数据复制到 CPU。

## 新增模型

新增模型专属实现，并在 UniTac 工厂中增加一个选择分支。实现 UniSim `TactileSensor` 契约，并为每个输出声明形状、单位、坐标系、设备 profile 和缓冲区生命周期。新模型可以有自己的几何要求和输出字段；UniSim 不规定 HydroShear 的网格或位移格式。
