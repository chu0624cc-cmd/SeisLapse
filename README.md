# SeiNoise
Ambient seismic noise processing: preprocessing, ACF/CCF correlation, and dv/v analysis using Python


# SeiNoise 🌍〰️

**Ambient Seismic Noise Processing & Time-Lapse Velocity Change (dv/v) Analysis Pipeline**

SeiNoise 是一个基于 Python 开发的地震背景噪声处理工作流。本项目致力于从连续地震记录中提取经验格林函数（Empirical Green's Functions），支持单台自相关（ACF）与双台互相关（CCF）计算，并可用于地下介质波速随时间变化的监测（Time-Lapse Seismic Velocity Changes, dv/v）。

本项目的底层算法逻辑深度参考了 Bensen et al. (2007) 的经典处理流程，并在计算效率和多频带自适应处理上进行了优化，其核心相关计算模块在数学上与 Julia 环境下的 `SeisNoise.jl` 保持等价。

## ✨ 核心功能 (Features)

- **高性能预处理 (`preprocess.py`)**：
  - 自动化质量控制（去除异常振幅与全零数据段）。
  - 支持多种时域归一化方法（One-bit, Clip, 标准滑动绝对平均 RAM, 单/多频带自适应 RAM）。
  - 精确的频率域谱白化（支持 Standard 与 Smoothed 方法，严格保证零相移）。
- **稳健的相关计算 (`correlate.py`)**：
  - 统一的 ACF 与 CCF 计算接口。
  - 支持基于 FFT 的高效频域计算与时域路径。
  - 内置后置谱白化（Post-whitening）功能，有效锐化高频尾波（Coda waves）信号。
- **信号叠加与处理 (`stacking.py`)**：
  - 提供线性和非线性叠加算法，快速提升信噪比（SNR）。
- **完整的工作流演示 (`notebooks`)**：
  - 提供 Jupyter Notebook 交互式教程（如 `01-完整步骤.ipynb`），开箱即用。
- **可视化模块 (`plotting/`)**：
  - 针对地震噪声数据的定制化绘图工具，方便快速查看波形与频谱特征。

## 📂 项目结构 (Repository Structure)

```text
SeiNoise/
├── preprocess.py        # 核心预处理模块（滤波、时域归一化、谱白化）
├── correlate.py         # 相关计算模块（ACF/CCF、清理与二次白化）
├── stacking.py          # 互相关/自相关函数的叠加模块
├── tool.py              # 底层工具函数（去趋势、尖灭、带通滤波等）
├── process_raw.py       # 原始连续波形数据的读取与切片
├── wxspectrum.py        # 交叉谱/功率谱分析工具
├── slide.py             # 滑动窗口分析工具
├── plotting/            # 地震学专用的可视化绘图包
│   ├── seisplot.py
│   └── seismic_module.py
└── 01-完整步骤.ipynb      # 完整流程的使用示例与测试验证
