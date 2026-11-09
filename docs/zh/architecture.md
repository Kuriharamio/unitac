# UniTac 架构与验证

[English](../en/architecture.md) | [中文](architecture.md)

## 仓库边界

UniSim 负责与模型无关的触觉契约。UniTac 是独立的 UniSim 插件，负责
HydroShear 配置、几何准备、CUDA 计算、历史状态、输出缓冲区和资源释放。
HydroShear 仓库只作为数值来源，不是运行时依赖。

```text
unisim-core
└── tactile.py                 场景、能力、生命周期和字段元数据

unitac
├── __init__.py                轻量导入和模型分发
├── geometry.py                碰撞描述 → body 局部三角网格
└── hydroshear/
    ├── config.py              HydroShear 专属配置校验
    ├── depth.py               相机光栅化与深度投影
    ├── sdf.py                 网格有符号距离
    ├── solver.py              HydroSoft 历史状态与 marker 位移
    ├── fft_dilation.py        可选网格膨胀路径
    ├── fft_shear.py           可选稠密剪切路径
    └── sensor.py              UniSim 生命周期与 CUDA 缓冲区所有权
```

## UniRay 的适用模式

UniTac 复用 UniRay 独立插件边界及其 UniSim 原则：模型包按需导入；静态
拓扑在冷路径准备；每步状态使用脱离引擎的数组或设备张量；输出所有权
和执行 profile 明确；插件不持有 `SimBackend`。几何来自 UniSim 的
`RayCollisionDescription`，安装坐标系来自 `RaySiteFrameDescription`。

触觉计算不使用 ray-caster 接口。契约允许每种模型选择自己的字段名、
形状、单位、坐标系和缓冲区生命周期。HydroShear 声明传感器局部位移网格；
其他模型可以输出节点、接触值、力、压力或图像。

## HydroShear 源码迁移

| `26403177779c4f74689761c041fa9580b075dc08` 中的来源文件 | UniTac 实现 | 保留或调整的行为 |
| --- | --- | --- |
| `src/hydroshear/depth.py` | `src/unitac/hydroshear/depth.py` | 相机投影、深度语义和三角形光栅化；持久化 renderer 工作区替代每次调用分配 z/depth 缓冲区。 |
| `src/hydroshear/mesh_sdf.py` | `src/unitac/hydroshear/sdf.py` | 融合的三角网格有符号距离和内外判定；固定批次预备查询及结果存储。 |
| `src/hydroshear/sensors/hydrosoft.py` | `src/unitac/hydroshear/solver.py` | HydroSoft 力历史、摩擦响应、膨胀、剪切和 reset 语义。历史输入在复用前复制，避免持久 CuPy 工作区改写前一帧状态。 |
| `src/hydroshear/sensors/fft_dilation.py` | `src/unitac/hydroshear/fft_dilation.py` | 可选 FFT 膨胀实现。 |
| `src/hydroshear/sensors/fft_shear.py` | `src/unitac/hydroshear/fft_shear.py` | 可选稠密剪切实现。 |
| 旧 batch、配置及 viewer 路径 | `geometry.py`、`config.py`、`sensor.py` | 改为脱离 UniSim 几何、一个校验配置、显式生命周期、预分配缓冲区和 caller-stream Torch/CuPy 互操作。旧 viewer 和实验设置不进入运行时。 |

几何层将受支持的公共 mesh 和 primitive 碰撞描述转为三角网格，以固定
随机种子一次性采样物体表面，并在每步将采样点变换到各指垫坐标系。
HydroShear 要求物体和凝胶表面闭合；调用者提供获准分发的凝胶网格。
仓库不附带 GelSight 标定文件或凝胶资产。

可选 FFT 模块保留为 HydroShear 内部算法实现。默认验证配置使用来源参数
`use_fft_dilation=False`、`use_matmul_shear=True`。UniSim 契约不暴露 FFT 抽象。

## XENSIM 架构检查

只读检查了 UniManip 适配器和其 vendored XENSIM 源码：UniManip `main`，
XENSIM 快照 `006166e9dc28d74690c69f89c8d5920f449f6b3b`。适配器使用每个物体
body 局部三角网格、环境数与 pad 数、可选 FEM 文件及求解选项构造
`TactileFemBatch`。每步组合物体和指垫世界位姿，保持 CUDA 结果，并用 FEM
静止位置计算位移，再映射到配置输出。底层 batch 可提供位移、位置、力和
接触数据；适配器目前发布单位为毫米的 `xensim` 位移字段。reset 委托给 FEM
运行时并支持选择环境。

| 可复用的 UniTac 公共契约 | 未来 XENSIM 插件专属部分 |
| --- | --- |
| 脱离引擎的物体碰撞几何与 body 身份 | FEM 文件、材料/求解参数和 FEM 静止网格 |
| 传感器安装坐标系及逐环境世界位姿 | FEM 接触/深度映射、矩阵装配和迭代求解 |
| materialize、位姿更新、advance、read、reset、close | XENSIM 通道、可选力/接触字段和网格插值 |
| 设备能力声明及各字段独立元数据 | XENSIM 缓存策略和静态 FEM 资源 |

该检查支持复用场景、位姿、生命周期、能力和字段元数据契约；不支持
现在把 FEM 参数、固定触觉网格或 XENSIM 空实现加入 UniSim 或 UniTac。
未来插件可在自己的包中管理 FEM 资源和求解器，并声明真实产生的字段形状。

## 数值参考

`tests/fixtures/hydroshear_reference.npz` 保存了来源提交在 RTX 4060 Laptop
GPU（Torch 2.9.0+cu128、CUDA 12.8）上生成的完整 marker 数组。两种实现
均使用合成闭合盒体、固定 2,048 个表面采样点、双指垫、9×7 网格、单位为
毫米的 float32 输出和相同求解参数。fixture 不包含专有标定资产。
`tests/test_hydroshear_reference.py` 使用 `rtol=2e-6, atol=2e-5` 逐项比较。

| 场景 | 来源最大绝对位移（mm） | 来源范数（mm） |
| --- | ---: | ---: |
| 无接触 | 0 | 0 |
| 压入 | 57.173409 | 427.700531 |
| 平移 | 57.245087 | 451.158966 |
| 旋转 | 57.159119 | 430.604095 |
| 滑动 | 57.631008 | 466.962219 |
| 持续接触 | 57.173409 | 427.700531 |
| 双物体 | 1.503018 | 12.737490 |
| 部分重置 | 1.009494 | 9.361657 |

单环境来源场为 `(2 pads, 9 rows, 7 columns, 3 components)`；UniTac 运行时
结果在其前增加环境维度。部分重置 fixture 记录了 reset 后被选择环境的输出。
本地 CUDA 设备上，八个 fixture 比较和位姿更新/读取缓冲区生命周期测试均通过。
