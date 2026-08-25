# OceanGraph

基于 GraphCast 的海洋时空预测代码与实验记录。

本项目在 GraphCast 图神经网络框架基础上开展海洋数据建模与预测相关实验，包含数据处理、模型实现、训练及推理示例。

> 本仓库用于论文代码整理与复现。数据集、完整训练产物和大规模预训练权重不直接上传。

## 项目结构

```text
OceanGraph/
├── graphcast/                    # GraphCast 核心模型与图网络组件
│   ├── graphcast.py              # 主模型
│   ├── typed_graph_net.py        # 类型化图神经网络
│   ├── deep_typed_graph_net.py   # 深层图网络
│   ├── data_utils.py             # 数据处理工具
│   ├── losses.py                 # 损失函数
│   ├── normalization.py          # 归一化模块
│   └── ...
├── all_graphcast_loadData.py     # 海洋/沉积物数据读取与预处理脚本
├── data_download.ipynb           # 数据下载示例
├── data_utils.ipynb              # 数据处理示例
├── train.ipynb                   # 训练实验示例
├── graphcast_demo.ipynb          # GraphCast 推理示例
├── setup.py                      # Python 依赖配置
└── LICENSE                       # Apache 2.0 许可证
```

## 环境配置

建议使用 Linux 与 NVIDIA GPU 环境，并安装与 CUDA 版本匹配的 JAX。

```bash
git clone https://github.com/BigXuuuuu/OceanGraph.git
cd OceanGraph

pip install -e .
```

如运行过程中缺少依赖，请根据报错安装相应 Python 包。核心依赖包括：

```text
jax
dm-haiku
jraph
xarray
numpy
pandas
scipy
trimesh
torch
timm
```

## 数据准备

本仓库不包含原始数据集。

- 数据读取与预处理示例见 `all_graphcast_loadData.py` 和 `data_utils.ipynb`。
- GraphCast 的公开模型权重、归一化统计量和示例输入可从 [DeepMind GraphCast 数据桶](https://console.cloud.google.com/storage/browser/dm_graphcast) 获取。
- 请根据自身数据集修改脚本中的本地数据路径。

## 使用说明

### 数据处理

```bash
python all_graphcast_loadData.py
```

运行前请在脚本中配置数据文件路径。

### 模型推理

打开并运行：

```text
graphcast_demo.ipynb
```

该 Notebook 展示 GraphCast 的数据加载、模型初始化和预测流程。

### 训练实验

训练流程示例见：

```text
train.ipynb
test/train_graphcast.py
test/pretrain_graphcast_nonpall.py
```

注意：`test/` 下部分历史脚本依赖特定的数据目录与工程模块，未作为开箱即用训练入口。复现实验时请根据本地数据路径、模型配置和运行环境调整。

## 说明

- 本项目基于 DeepMind 提出的 GraphCast 工作及其开源实现进行实验与修改。
- 原始 GraphCast 代码遵循 Apache License 2.0；本仓库保留原始许可证与版权声明。
- 模型权重可能适用独立许可；使用或分发前请确认其许可证要求。
- 本项目为研究用途，不构成官方 GraphCast 实现或服务。

## 参考文献

```bibtex
@article{lam2023learning,
  title={Learning skillful medium-range global weather forecasting},
  author={Lam, Remi and Sanchez-Gonzalez, Alvaro and Willson, Matthew and others},
  journal={Science},
  volume={382},
  number={6677},
  pages={1416--1421},
  year={2023},
  doi={10.1126/science.adi2336}
}
```

## 致谢

本项目参考了以下开源工作：

- [Google DeepMind GraphCast](https://github.com/google-deepmind/graphcast)
- [GraphCast-from-Ground-Zero](https://github.com/sfsun67/GraphCast-from-Ground-Zero)

## 许可证

本项目遵循 [Apache License 2.0](LICENSE)。
