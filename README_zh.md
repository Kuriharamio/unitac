# UniTac

UniTac 是面向 [UniSim](https://github.com/unilabsim/unisim) 的独立触觉计算插件。目前包含 HydroShear；其求解器、网格 SDF、深度光栅化、几何处理和时间状态均由本仓库维护。运行时不依赖原 HydroShear 仓库。

`unisim` 提供与模型无关的 `TactileSensor` 契约和惰性插件工厂。UniTac 管理 HydroShear 参数和 CUDA 实现。模型接收脱离后端的碰撞几何、传感器安装坐标系和世界位姿，不持有仿真后端。

## 安装

基础包不导入 Torch 或 CUDA 库。在 CUDA 12 环境中安装算法依赖：

```bash
uv add unitac
uv add unitac --extra hydroshear
```

开发仓库时递归克隆子模块并同步锁定环境：

```bash
git clone --recurse-submodules https://github.com/Kuriharamio/UniTac.git
uv sync --extra hydroshear --extra viz
```

HydroShear 的 SDF 与深度计算要求 CUDA。位姿输入支持 NumPy、CPU Torch 和 CUDA Torch，输出为传感器拥有的 CUDA Torch 位移场。调用者需提供传感器坐标系中的闭合凝胶三角网格，单位为米。仓库不包含 HydroShear 标定文件或凝胶网格。

## 使用

```python
from unisim.factory import create_tactile_sensor

sensor = create_tactile_sensor(
    "unitac",
    model="hydroshear",
    num_envs=64,
    config={
        "gel_mesh": licensed_closed_gel_triangles,  # (三角形数, 3, 3)，米
        "surface_points": 2048,
        "output_grid": {
            "rows": 9,
            "cols": 7,
            "spacing_mm": [2.145, 2.153125],
            "pixel_scale_mm": 0.065,
        },
        "solver": {"use_matmul_shear": True},
        "device": "cuda:0",
    },
)
sensor.materialize(scene_description)
sensor.update_pose(body_pos, body_quat)  # 世界坐标，米，单位 wxyz 四元数
sensor.advance()                         # 推进一次模型和历史状态
observation = sensor.read()["displacement"]
sensor.reset(env_ids=[3, 7])
sensor.close()
```

`read()` 返回可复用缓冲区的视图。如需跨越下一次 `advance`、`reset`、`rebuild` 或 `close` 保存数据，请先复制。参见[架构与验证说明](docs/zh/architecture.md)。

## 开发

```bash
uv run ruff check .
uv run pytest -q
uv run python benchmarks/bench_gpu.py --envs 1 64 1024
```

HydroShear 数值代码来源和许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [licenses/HydroShear-MIT.txt](licenses/HydroShear-MIT.txt)。
