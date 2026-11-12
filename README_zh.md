# UniTac

UniTac 是面向 [UniSim](https://github.com/unilabsim/unisim) 的独立触觉仿真插件，目前提供 HydroShear 模型。UniSim 提供脱离后端的碰撞几何、传感器坐标系、生命周期和字段元数据；UniTac 管理触觉模型，不持有仿真后端。

## 当前功能

- 在 CUDA 上计算 HydroShear 触觉位移，支持调用者提供的闭合凝胶几何。
- GelSight Mini 标定 RGB、深度和 marker 可视化。
- 已在 UniSim MuJoCo、MJWarp 和 Motrix 后端上验证完整示例流程。

HydroShear 位移是经验模型输出，不是物理长度。字段形状为 `(environment, sensor, row, column, xyz)`，单位标记为 `hydroshear_model_units`。相机压入深度单位为毫米。可视化使用经验系数 `0.05 mm/model unit` 将模型单位转换为显示位移。

## 安装

从 Git 安装 UniTac：

```bash
uv add "unitac @ git+https://github.com/Kuriharamio/UniTac.git@main"
uv add "unitac[hydroshear] @ git+https://github.com/Kuriharamio/UniTac.git@main"
```

HydroShear 需要可用的 CUDA 设备，以及兼容 CUDA 12 的 PyTorch 和 CuPy。在仓库检出目录中，为所选运行模式安装依赖：

```bash
# 由 MJBatch 执行的 MuJoCo 物理仿真与交互/无界面可视化
uv sync --extra hydroshear --extra mujoco --extra visualization

# 使用同一 viewer 的 MJWarp 物理仿真
uv sync --extra hydroshear --extra mujoco --extra mjwarp --extra visualization

# 使用 Motrix 原生 viewer 的物理仿真
uv sync --extra hydroshear --extra motrix --extra visualization
```


## 运行

最简公共 API 示例读取 UniSim 提供的初始位姿并计算一次  HydroShear：

```bash
uv run --extra hydroshear --extra mujoco python examples/mjbatch_backend.py
```

letter press 入口运行完整的物理仿真与触觉更新流程：

```bash
uv run python examples/letter_press/run_mjbatch.py
uv run python examples/letter_press/run_mjwarp.py
uv run python examples/letter_press/run_motrix.py
```

 `--headless` 选项会录制 MP4 和阶段截图
 
 `--no-render` 只运行物理仿真与 HydroShear，不加载 RGB 标定数据，也不创建 renderer。Motrix 示例使用 Motrix 原生 viewer，并支持 `--no-render`

## 公共 API

`unisim.factory.create_tactile_sensor(tactile_type="hydroshear", ...)` 返回 UniSim `TactileSensor`。调用者提供传感器局部坐标系中的闭合凝胶三角网格，单位为米；通过 `TactileSceneDescription` 完成物化后，依次调用 `update_pose`、`advance`、`read`、`reset` 和 `close`。`read()` 返回可复用 CUDA Torch 缓冲区的视图。最简公共 API 示例见 [examples/mjbatch_backend.py](examples/mjbatch_backend.py)。

## 文档与许可

- [Architecture](docs/en/architecture.md) · [架构说明](docs/zh/architecture.md)
- [第三方声明](THIRD_PARTY_NOTICES.md)
