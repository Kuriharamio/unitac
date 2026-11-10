# UniTac

UniTac 是面向 [UniSim](https://github.com/unilabsim/unisim) 的独立触觉仿真插件，目前提供 HydroShear 模型。UniSim 提供脱离后端的碰撞几何、传感器坐标系、生命周期和字段元数据；UniTac 管理触觉模型，不持有仿真后端。

## 当前功能

- 在 CUDA 上计算 HydroShear 触觉位移，支持调用者提供的闭合凝胶几何。
- GelSight Mini 标定 RGB、深度和 marker 可视化。
- 支持 UniSim MuJoCo 集成；MJWarp 可通过可选依赖安装。

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
```

HydroShear 需要可用的 CUDA 设备，以及兼容 CUDA 12 的 PyTorch 和 CuPy。`visualization` 安装 OpenCV；`mujoco` 安装 MuJoCo 集成依赖；使用 MJWarp 时再添加 `mjwarp`。GelSight Mini 标定文件和 letter press 几何已包含在安装包中。

## 运行

完整的公共 API 集成示例会推进 MuJoCo 场景，并将脱离后端的几何和位姿传给 HydroShear：

```bash
uv run --extra hydroshear --extra mujoco python examples/mujoco_backend.py
```

运行包含 RGB、深度和 marker 可视化的 letter press 序列：

```bash
uv run --extra hydroshear --extra visualization --extra mujoco \
  python examples/letter_press/run.py --headless --output-dir outputs/letter_press
```

示例按接近、下压、保持、旋转、抬升和返回的顺序推进 MuJoCo 位姿。HydroShear 根据物体几何和位姿计算接触，但不会将触觉接触力反馈给 MuJoCo。示例箭头增益默认为 `40`。可用 `--calibration-dir` 指定外部 GelSight Mini 标定数据。

## 公共 API

`unisim.factory.create_tactile_sensor("unitac", model="hydroshear", ...)` 返回 UniSim `TactileSensor`。调用者提供传感器局部坐标系中的闭合凝胶三角网格，单位为米；通过 `TactileSceneDescription` 完成物化后，依次调用 `update_pose`、`advance`、`read`、`reset` 和 `close`。`read()` 返回可复用 CUDA Torch 缓冲区的视图；如需跨越下一次 advance 或 reset 保存数据，请先复制。完整可运行的集成示例见 [examples/mujoco_backend.py](examples/mujoco_backend.py)。

HydroShear 几何、SDF 和深度计算要求 CUDA，传感器没有 CPU 计算路径。RGB 和深度渲染是显式调用的 CPU 诊断操作，`read()` 不会执行渲染。

## 文档与许可

- [Architecture](docs/en/architecture.md) · [架构说明](docs/zh/architecture.md)
- [第三方声明](THIRD_PARTY_NOTICES.md)
- [MIT 许可](LICENSE)
