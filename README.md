# SeiLapse
Ambient seismic noise processing: preprocessing, ACF/CCF correlation, and dv/v analysis using Python


# SeiLapse 🌍〰️

**Ambient Seismic Noise Processing & Time-Lapse Velocity Change (dv/v) Analysis Pipeline**

SeiLapse 是一个基于 Python 开发的地震背景噪声处理工作流。本项目致力于从连续地震记录中提取经验格林函数（Empirical Green's Functions），支持单台自相关（ACF）与双台互相关（CCF）计算，并可用于地下介质波速随时间变化的监测（Time-Lapse Seismic Velocity Changes, dv/v）。

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
```


## 📖 参考文献 (References)

本项目底层算法逻辑与数据处理管线主要基于以下经典文献与前沿研究：

**背景噪声处理基础 (Ambient Noise Processing):**
- Bensen, G. D., Ritzwoller, M. H., Barmin, M. P., Levshin, A. L., Lin, F., Moschetti, M. P., Shapiro, N. M., & Yang, Y. (2007). Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements. *Geophysical Journal International*, 169(3), 1239-1260. [doi:10.1111/j.1365-246X.2007.03374.x](https://doi.org/10.1111/j.1365-246X.2007.03374.x)
- Clements, T., & Denolle, M. A. (2020). SeisNoise.jl: Ambient Seismic Noise Cross Correlation on the CPU and GPU in Julia. *Seismological Research Letters*, 92(1), 517-527. [doi:10.1785/0220200192](https://doi.org/10.1785/0220200192)

**尾波干涉与走时偏移提取 (Coda Wave Interferometry & dv/v):**
- Brenguier, F., Campillo, M., Hadziioannou, C., Shapiro, N. M., Nadeau, R. M., & Vilotte, E. (2008). Postseismic relaxation along the San Andreas fault at Parkfield from continuous seismological observations. *Science*, 321(5895), 1478-1481. (经典断裂带波速监测应用)
- Clarke, D., Zaccarelli, L., Shapiro, N. M., & Brenguier, F. (2011). Assessment of resolution and accuracy of the Moving Window Cross Spectral technique for monitoring crustal temporal variations using ambient seismic noise. *Geophysical Journal International*, 186(2), 867-882. (MWCS 移动窗口互谱法基础)
- Meier, U., Shapiro, N. M., & Brenguier, F. (2010). Detecting seasonal variations in seismic velocities within Los Angeles basin from correlations of ambient seismic noise. *Geophysical Journal International*, 181(2), 985-996. (Stretching 波形拉伸法基础)
- Mao, S., Mordret, A., Campillo, M., Fang, H., & van der Hilst, R. D. (2020). On the measurement of seismic traveltime changes in the time–frequency domain with wavelet cross-spectrum analysis. Geophysical Journal International, 221(1), 550-568. doi:10.1093/gji/ggaa009 (小波交叉谱 WCS 方法核心文献)

## 💻 参考代码库 (Open Source Acknowledgements)

本项目的开发深刻受益于开源社区，特别是以下优秀的地震学软件项目。`SeiNoise` 的核心频域运算逻辑在数学上严格对齐了 `SeisNoise.jl`：

- [**SeisNoise.jl**](https://github.com/tclements/SeisNoise.jl): 高性能的 Julia 语言背景噪声互相关库 (Clements & Denolle, 2020)。本项目的 `_correlate_freq` 及谱白化模块高度参考了其底层实现。
- [**NoisePy**](https://github.com/mdenolle/NoisePy): 另一个由 Denolle 组主导的优秀 Python 背景噪声干涉流计算库，为本项目的工程架构设计提供了灵感。
- [**MSNoise**](https://github.com/ROB-Seismology/MSNoise): 基于 Python 的完整地震噪声波速变化 (dv/v) 监测软件套件，是业界广泛使用的标准化工具。
- dt-wavelet / cross-wavelet-transform: 由 Shujuan Mao 和 Aurélien Mordret 提供的开源库。本项目的 wxspectrum.py 高度参考了该代码库中利用小波交叉谱（Wavelet Cross-Spectrum）在时频域内测量走时偏移的 Python/MATLAB 实现逻辑。
