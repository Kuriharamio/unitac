# UniTac

UniTac 是面向 [UniSim](https://github.com/unilabsim/unisim) 的独立触觉仿真插件，目前提供 HydroShear 模型。UniSim 提供脱离后端的碰撞几何、传感器坐标系、生命周期和字段元数据；UniTac 管理触觉模型，不持有仿真后端。

## 当前功能

- 在 CUDA 上计算 HydroShear 触觉位移，支持调用者提供的闭合凝胶几何。
- GelSight Mini 标定 RGB、深度和 marker 可视化。
- 已在 UniSim MuJoCo、MJWarp 和 Motrix 后端上验证 letter press 完整流程。

HydroShear 位移是经验模型输出，不是物理长度。字段形状为 `(environment, sensor, row, column, xyz)`，单位标记为 `hydroshear_model_units`。相机压入深度单位为毫米。可视化使用经验系数 `0.05 mm/model unit` 将模型单位转换为显示位移；箭头 `display_gain` 只改变箭头。

## 安装

从 Git 安装 UniTac：

```bash
uv add "unitac @ git+https://github.com/Kuriharamio/UniTac.git@main"
uv add "unitac[hydroshear] @ git+https://github.com/Kuriharamio/UniTac.git@main"
```

按需安装可选依赖：

```bash
uv add "unitac[hydroshear,visualization] @ git+https://github.com/Kuriharamio/UniTac.git@main"
uv add "unitac[hydroshear,mujoco] @ git+https://github.com/Kuriharamio/UniTac.git@main"
uv add "unitac[hydroshear,mujoco,mjwarp] @ git+https://github.com/Kuriharamio/UniTac.git@main"
uv add "unitac[hydroshear,motrix] @ git+https://github.com/Kuriharamio/UniTac.git@main"
```

HydroShear 需要可用的 CUDA 设备，以及兼容 CUDA 12 的 PyTorch 和 CuPy。在仓库检出目录中，为所选运行模式安装依赖：

```bash
# MuJoCo 物理仿真与交互/无界面可视化
uv sync --extra hydroshear --extra mujoco --extra visualization

# 使用同一 viewer 的 MJWarp 物理仿真
uv sync --extra hydroshear --extra mujoco --extra mjwarp --extra visualization

# 不加载渲染器的 Motrix 物理仿真
uv sync --extra hydroshear --extra motrix
```

`--no-render` 不需要 `visualization` extra。Motrix 录制视频或打开 viewer 时，再添加 `--extra visualization`。GelSight Mini 标定文件和 letter press 几何已包含在安装包中。

## 运行

完整的公共 API 集成示例会推进 MuJoCo 场景，并将脱离后端的几何和位姿传给 HydroShear：

```bash
uv run --extra hydroshear --extra mujoco python examples/mujoco_backend.py
```

在任一已验证的物理后端上运行相同的 letter press 序列：

```bash
uv run python examples/letter_press/run.py --backend mujoco
uv run python examples/letter_press/run.py --backend mjwarp
uv run python examples/letter_press/run.py --backend motrix --no-render
```

默认模式会打开交互 viewer。添加 `--headless` 可录制 MP4 和各阶段截图；添加 `--no-render` 则只运行物理仿真与 HydroShear，不加载 RGB 标定数据，也不创建 renderer。这两种模式都支持 `--frames`、`--device` 和 `--output-dir`；只有渲染模式支持 `--mode rgb|depth`。无渲染模式会在 CUDA 上汇总位移标量；深度和接触像素指标需要可视化快照，因此不写入该模式的日志。

选定的 UniSim 后端负责推进物理仿真，并提供 body 位姿、碰撞几何、site frame 和 playback snapshot。Viewer 使用 MuJoCo 将 snapshot 渲染为独立的可视化模型；运行 MJWarp 时不会推进 MuJoCo 物理仿真。MuJoCo、MJWarp 和 Motrix 均已通过完整周期无渲染运行。Motrix 为 letter 网格提供编译后的凸包，因此 HydroShear 使用与 Motrix 相同的碰撞表面。HydroShear 接触力不会反馈给物理后端。

## 公共 API

`unisim.factory.create_tactile_sensor("unitac", model="hydroshear", ...)` 返回 UniSim `TactileSensor`。调用者提供传感器局部坐标系中的闭合凝胶三角网格，单位为米；通过 `TactileSceneDescription` 完成物化后，依次调用 `update_pose`、`advance`、`read`、`reset` 和 `close`。`read()` 返回可复用 CUDA Torch 缓冲区的视图；如需跨越下一次 advance 或 reset 保存数据，请先复制。完整可运行的集成示例见 [examples/mujoco_backend.py](examples/mujoco_backend.py)。

HydroShear 几何、SDF 和深度计算要求 CUDA，传感器没有 CPU 计算路径。RGB 和深度渲染是显式调用的 CPU 诊断操作，`read()` 不会执行渲染。

## 文档与许可

- [Architecture](docs/en/architecture.md) · [架构说明](docs/zh/architecture.md)
- [第三方声明](THIRD_PARTY_NOTICES.md)
- [MIT 许可](LICENSE)
