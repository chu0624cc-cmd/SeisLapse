"""
process_raw.py
==============
"""

import numpy as np
from scipy.signal import hilbert, resample_poly
from scipy.fft import fft, ifft, rfft, irfft, rfftfreq, fftshift
from scipy.signal import detrend as scipy_detrend
import warnings
import copy
from math import gcd
from tool import detrend, demean, taper, lowpass_filter

from scipy.fft import rfft, irfft, rfftfreq, fft, ifft, fftfreq, next_fast_len


# =       去除仪器响应    = = = = =


def paz_transfer_function(freqs: np.ndarray, paz: dict) -> np.ndarray:
    """计算 PAZ 传递函数 H(f)"""
    s = 2j * np.pi * freqs
    H = np.ones(len(freqs), dtype=complex)
    for z in paz['zeros']: H *= (s - z)
    for p in paz['poles']: H /= (s - p)
    return H * paz['gain']


# def remove_response(x: np.ndarray, fs: float, paz: dict,
#                     pre_filt: tuple, water_level_db: float = 60.0) -> np.ndarray:
#     """
#     PAZ 频域去仪器响应，输出 m/s。
#
#     Parameters
#     ----------
#     x             : 原始 counts 时间序列，shape (N,)
#     fs            : 采样率 (Hz)
#     paz           : 极零点字典，含 poles/zeros/gain/sensitivity/norm_freq
#     pre_filt      : 余弦锥形预滤波四角频率 (f1, f2, f3, f4)，Hz
#     water_level_db: 水位（dB），抑制除零，默认 60 dB
#
#     Returns
#     -------
#     去响应后时间序列，单位 m/s，shape (N,)
#     """
#     N     = len(x)
#     Nfft  = next_fast_len(N)
#     freqs = fftfreq(Nfft, d=1.0 / fs)
#     X     = fft(x, Nfft)
#
#     # 传递函数 + sensitivity 校正
#     H_raw    = paz_transfer_function(freqs, paz)
#     f_ref    = paz['norm_freq']
#     s_ref    = 2j * np.pi * f_ref
#     H_at_ref = paz['gain']
#     for z in paz['zeros']: H_at_ref *= (s_ref - z)
#     for p in paz['poles']: H_at_ref /= (s_ref - p)
#     H = H_raw * (paz['sensitivity'] / abs(H_at_ref))
#
#     # water-level（基于通带中位数）
#     H_abs     = np.abs(H)
#     fa        = np.abs(freqs)
#     band_mask = (fa >= 0.5) & (fa <= 10.0)
#     H_median  = np.median(H_abs[band_mask]) if band_mask.any() else H_abs.max()
#     wl        = H_median * 10.0 ** (-water_level_db / 20.0)
#
#     H_safe           = H.copy()
#     too_small        = H_abs < wl
#     H_safe[too_small] = wl * np.exp(1j * np.angle(H[too_small]))
#     X_corr           = X / H_safe
#
#     # 余弦锥形预滤波
#     f1, f2, f3, f4 = pre_filt
#     tap = np.zeros(Nfft)
#     tap[(fa >= f2) & (fa <= f3)] = 1.0
#     m12 = (fa >= f1) & (fa < f2)
#     m34 = (fa > f3) & (fa <= f4)
#     tap[m12] = 0.5 * (1.0 - np.cos(np.pi * (fa[m12] - f1) / (f2 - f1)))
#     tap[m34] = 0.5 * (1.0 + np.cos(np.pi * (fa[m34] - f3) / (f4 - f3)))
#     X_corr  *= tap
#
#     x_out = np.real(ifft(X_corr, Nfft))[:N]
#     return scipy_detrend(x_out, type='linear')



def remove_response(x: np.ndarray, fs: float, paz: dict,
                    pre_filt: tuple, water_level_db: float = 60.0) -> np.ndarray:
    """
    PAZ 频域去仪器响应，输出 m/s。
    在去响应前先对数据两端做 taper，压制滤波器瞬态响应。
    """
    N     = len(x)
    Nfft  = next_fast_len(N)
    freqs = fftfreq(Nfft, d=1.0 / fs)
    x = x - np.mean(x)

    X     = fft(x, Nfft)

    # 传递函数 + sensitivity 校正
    H_raw    = paz_transfer_function(freqs, paz)
    f_ref    = paz['norm_freq']
    s_ref    = 2j * np.pi * f_ref
    H_at_ref = paz['gain']
    for z in paz['zeros']: H_at_ref *= (s_ref - z)
    for p in paz['poles']: H_at_ref /= (s_ref - p)
    H = H_raw * (paz['sensitivity'] / abs(H_at_ref))

    # water-level
    H_abs     = np.abs(H)
    fa        = np.abs(freqs)
    band_mask = (fa >= 0.5) & (fa <= 10.0)
    H_median  = np.median(H_abs[band_mask]) if band_mask.any() else H_abs.max()
    wl        = H_median * 10.0 ** (-water_level_db / 20.0)

    H_safe            = H.copy()
    too_small         = H_abs < wl
    H_safe[too_small] = wl * np.exp(1j * np.angle(H[too_small]))
    X_corr            = X / H_safe

    # 余弦锥形预滤波
    f1, f2, f3, f4 = pre_filt
    tap = np.zeros(Nfft)
    tap[(fa >= f2) & (fa <= f3)] = 1.0
    m12 = (fa >= f1) & (fa < f2)
    m34 = (fa > f3) & (fa <= f4)
    tap[m12] = 0.5 * (1.0 - np.cos(np.pi * (fa[m12] - f1) / (f2 - f1)))
    tap[m34] = 0.5 * (1.0 + np.cos(np.pi * (fa[m34] - f3) / (f4 - f3)))
    X_corr  *= tap

    x_out = np.real(ifft(X_corr, Nfft))[:N]
    return scipy_detrend(x_out, type='linear')

#   =      去除仪器响应  =   = = = = =




def resample(x: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    """
    多相重采样（保持精度）。

    Parameters
    ----------
    x      : 时间序列，shape (N,) 或 (N, M)
    fs_old : 原始采样率 (Hz)
    fs_new : 目标采样率 (Hz)
    """
    if fs_old == fs_new:
        return x
    common = gcd(int(fs_new), int(fs_old))
    up = int(fs_new) // common
    down = int(fs_old) // common
    if x.ndim == 1:
        return resample_poly(x, up, down)
    return np.column_stack([resample_poly(x[:, i], up, down)
                            for i in range(x.shape[1])])


def phase_shift(x: np.ndarray, fs: float, t0: float,
                phi_shift: bool = True) -> np.ndarray:
    """
    相位偏移：将数据起始时间对齐到最近的采样点。。
    核心逻辑：
      dt  = 1 / fs
      off = t0 % dt
      if off <= dt/2 : off = -off       # 向前对齐
      else           : off = dt - off   # 向后对齐
      fftdata *= exp(+1j * 2π * freq * off)
    Parameters
    ----------
    x         : 时间序列，shape (N,) 或 (N, M)
    fs        : 采样率 (Hz)
    t0        : 数据起始时刻的亚秒偏移量（秒）
    phi_shift : False 时跳过频域偏移（仅对齐逻辑，与 Julia ϕshift 一致）
    """
    dt = 1.0 / fs

    # 计算亚采样周期内的偏移（Julia: mod(mod(t_us, 1e6)*1e-6, dt)）
    off = t0 % dt

    # 偏移量极小（接近整采样点），无需处理
    if dt - off <= np.finfo(np.float32).eps:
        off = 0.0

    if off == 0.0:
        return x.copy() if x.ndim == 1 else x.copy()

    # 判断偏移方向（Julia 核心逻辑）
    if off <= dt / 2.0:
        off = -off        # 偏移小：向前移
    else:
        off = dt - off    # 偏移大：向后移到下一采样点

    if not phi_shift:
        return x.copy()

    # 频域相位偏移：exp(+1j * 2π * freq * off)，一致
    n = x.shape[0]
    freq = rfftfreq(n, d=1.0 / fs)
    phase = np.exp(1j * 2.0 * np.pi * freq * off)   # ← 正号

    X = rfft(x, axis=0)
    if x.ndim == 2:
        phase = phase[:, np.newaxis]
    return irfft(X * phase, n=n, axis=0)


# ─────────────────────────────────────────────
# 2. 原始数据预处理
# ─────────────────────────────────────────────


def process_raw(x: np.ndarray, fs_old: float, fs_new: float,
                t0: float = 0.0, phi_shift: bool = True,
                paz: dict = None, pre_filt: tuple = None,
                water_level_db: float = 60.0) -> np.ndarray:
    """
    原始地震数据完整预处理流程。
    对应 Julia process_raw!(S, fs)，并在其之前增加仪器响应去除。

      [可选] 去仪器响应（PAZ → m/s）
         ↓
      去趋势 + 锥形窗
         ↓
      低通滤波（抗混叠）
         ↓
      降采样
         ↓
      锥形窗 + 相位对齐

    Parameters
    ----------
    x             : 原始时间序列 counts，shape (N,)
    fs_old        : 原始采样率 (Hz)
    fs_new        : 目标采样率 (Hz)
    t0            : 亚秒时间偏移（秒）
    phi_shift     : 是否执行相位对齐
    paz           : PAZ 字典，不为 None 时执行去仪器响应
    pre_filt      : 预滤波四角频率 (f1,f2,f3,f4)，paz 不为 None 时必填
    water_level_db: 水位（dB），默认 60
    """
    # Step 0：去仪器响应（可选）
    if paz is not None:
        if pre_filt is None:
            raise ValueError("提供 paz 时必须同时提供 pre_filt")
        x = remove_response(x, fs_old, paz, pre_filt, water_level_db)

    # Step 1-2：去趋势 + 锥形窗，对应 Julia detrend!(S) + taper!(S)
    x = detrend(x)
    x = taper(x, fs_old)

    # Step 3：抗混叠低通滤波
    if fs_new != fs_old:
        x = lowpass_filter(x, fh=fs_new / 2.0, fs=fs_old)

    # Step 4-5：降采样 + 锥形窗
    x = resample(x, fs_old, fs_new)
    x = taper(x, fs_new)

    # Step 6：相位对齐
    if phi_shift:
        x = phase_shift(x, fs_new, t0)

    return x


def process_raw_inplace(x: np.ndarray, fs_old: float, fs_new: float,
                        t0: float = 0.0, phi_shift: bool = True,
                        paz: dict = None, pre_filt: tuple = None,
                        water_level_db: float = 60.0) -> np.ndarray:
    """process_raw 原地修改版本"""
    x[:] = process_raw(x, fs_old, fs_new, t0=t0, phi_shift=phi_shift,
                       paz=paz, pre_filt=pre_filt,
                       water_level_db=water_level_db)
    return x





# def process_raw(x: np.ndarray, fs_old: float, fs_new: float,
#                 t0: float = 0.0, phi_shift: bool = True) -> np.ndarray:
#     """
#     原始地震数据预处理流程（非原地修改，返回新数组）：
#       1. 去均值 & 去趋势
#       2. 锥形窗
#       3. 低通滤波（降采样前抗混叠）
#       4. 降采样
#       5. 再次锥形窗
#       6. 相位偏移对齐整点
#
#     Parameters
#     ----------
#     x        : 原始时间序列，shape (N,) 或 (N, M)
#     fs_old   : 原始采样率 (Hz)
#     fs_new   : 目标采样率 (Hz)
#     t0       : 数据起始时间偏移（秒，用于相位对齐）
#     phi_shift: 是否执行相位偏移
#
#     Returns
#     -------
#     处理后的时间序列 np.ndarray
#     """
#     x = demean(x)
#     x = detrend(x)
#     x = taper(x, fs_old)
#
#     if fs_new != fs_old:
#         x = lowpass_filter(x, fh=fs_new / 2.0, fs=fs_old)
#
#     x = resample(x, fs_old, fs_new)
#     x = taper(x, fs_new)
#
#     if phi_shift:
#         x = phase_shift(x, fs_new, t0, phi_shift=phi_shift)
#
#     return x
#
#
# def process_raw_inplace(x: np.ndarray, fs_old: float, fs_new: float,
#                         t0: float = 0.0, phi_shift: bool = True) -> np.ndarray:
#     """
#     原始数据预处理（原地修改版本，直接覆盖输入数组）。
#     参数含义与 process_raw 相同。
#     """
#     x[:] = process_raw(x, fs_old, fs_new, t0=t0, phi_shift=phi_shift)
#     return x







# # ─────────────────────────────────────────────
# # 3. 振幅处理：静音 / 截幅 / 限幅
# # ─────────────────────────────────────────────
#
# def mute(x: np.ndarray, factor: float = 3.0) -> np.ndarray:
#     """
#     将高振幅样本置零（基于包络中位数）。
#
#     Parameters
#     ----------
#     x      : 时间序列，shape (N,) 或 (N, M)
#     factor : 超过 factor * median(envelope) 的部分置零
#
#     Returns
#     -------
#     处理后的时间序列（新数组）
#     """
#     out = x.copy()
#     envelope = np.abs(hilbert(out, axis=0))
#     levels = envelope.mean(axis=0)
#     level = factor * np.median(levels)
#     out[envelope > level] = 0.0
#     return out
#
#
# def mute_inplace(x: np.ndarray, factor: float = 3.0) -> None:
#     """mute 的原地版本。"""
#     x[:] = mute(x, factor)
#
#
# def clip(x: np.ndarray, factor: float,
#          func=np.std, axis: int = 0) -> np.ndarray:
#     """
#     按统计阈值截幅：将超过 factor * func(x) 的部分截断。
#
#     Parameters
#     ----------
#     x      : 时间序列，shape (N,) 或 (N, M)
#     factor : 阈值倍数
#     func   : 统计函数（默认 np.std，可替换为 np.var、MAD 等）
#     axis   : 统计维度
#
#     Returns
#     -------
#     截幅后的新数组
#     """
#     out = x.copy()
#     if out.ndim == 1:
#         high = func(out) * factor
#         np.clip(out, -high, high, out=out)
#     else:
#         highs = func(out, axis=axis) * factor
#         for i in range(out.shape[1]):
#             np.clip(out[:, i], -highs[i], highs[i], out=out[:, i])
#     return out
#
#
# def clip_inplace(x: np.ndarray, factor: float,
#                  func=np.std, axis: int = 0) -> None:
#     """clip 的原地版本。"""
#     x[:] = clip(x, factor, func=func, axis=axis)
#
#
# def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
#     """
#     将数组值限制在 [lo, hi] 区间内。
#
#     Parameters
#     ----------
#     x  : 输入数组
#     lo : 下限
#     hi : 上限
#     """
#     if lo >= hi:
#         raise ValueError(f"lo ({lo}) 必须小于 hi ({hi})")
#     return np.clip(x, lo, hi)
#
#
# def clamp_inplace(x: np.ndarray, lo: float, hi: float) -> None:
#     """clamp 的原地版本。"""
#     if lo >= hi:
#         raise ValueError(f"lo ({lo}) 必须小于 hi ({hi})")
#     np.clip(x, lo, hi, out=x)
#
#
# def onebit(x: np.ndarray) -> np.ndarray:
#     """
#     单比特归一化：将时间序列替换为其符号（+1 / -1 / 0）。
#
#     Parameters
#     ----------
#     x : 时间序列数组
#
#     Returns
#     -------
#     符号数组（新数组）
#     """
#     return np.sign(x)
#
#
# def onebit_inplace(x: np.ndarray) -> None:
#     """onebit 的原地版本。"""
#     x[:] = np.sign(x)
#
#
# # ─────────────────────────────────────────────
# # 4. 振幅筛选（remove_amp）
# # ─────────────────────────────────────────────
#
# def nonzero_columns(x: np.ndarray) -> np.ndarray:
#     """
#     返回包含至少一个非零元素的列索引。
#
#     Parameters
#     ----------
#     x : 2D 数组，shape (N, M)
#
#     Returns
#     -------
#     非零列的整数索引数组
#     """
#     return np.where(np.any(x != 0, axis=0))[0]
#
#
# def std_threshold_columns(x: np.ndarray, max_std: float) -> np.ndarray:
#     """
#     返回标准差不超过 max_std * global_std 的列索引。
#
#     Parameters
#     ----------
#     x       : 2D 数组，shape (N, M)
#     max_std : 标准差倍数阈值
#
#     Returns
#     -------
#     满足条件的列索引数组
#     """
#     global_std = np.std(x)
#     col_std = np.std(x, axis=0)
#     return np.where(col_std <= max_std * global_std)[0]
#
#
# def remove_amp(x: np.ndarray, t: np.ndarray,
#                max_std: float = 10.0):
#     """
#     基于振幅质量控制过滤窗口：
#       1. 删除全零列
#       2. 删除标准差超过阈值的列
#
#     Parameters
#     ----------
#     x       : 数据矩阵，shape (N, M)
#     t       : 时间戳数组，shape (M,)
#     max_std : 标准差倍数阈值
#
#     Returns
#     -------
#     (x_clean, t_clean) : 过滤后的数据和时间戳
#     """
#     # 1. 去除全零列
#     zero_ind = nonzero_columns(x)
#     if len(zero_ind) == 0:
#         raise ValueError("所有窗口数据均为零")
#     x, t = x[:, zero_ind], t[zero_ind]
#
#     # 2. 去除超标准差列
#     std_ind = std_threshold_columns(x, max_std)
#     if len(std_ind) == 0:
#         raise ValueError("所有窗口数据的标准差均超过阈值")
#     x, t = x[:, std_ind], t[std_ind]
#
#     return x, t
#
#
# # ─────────────────────────────────────────────
# # 5. 瞬时相位（Phase / Hilbert）
# # ─────────────────────────────────────────────
#
# def hilbert_transform(x: np.ndarray) -> np.ndarray:
#     """
#     计算 Hilbert 变换（虚部），与 Julia 实现保持一致。
#
#     Julia 版对 FFT 频谱手动乘以 -i / +i，等价于标准 Hilbert 变换的虚部。
#
#     Parameters
#     ----------
#     x : 实数时间序列，shape (N,) 或 (N, M)
#
#     Returns
#     -------
#     Hilbert 变换结果（复数数组）
#     """
#     n = x.shape[0]
#     f = fft(x, axis=0)
#
#     h = np.zeros(n, dtype=complex)
#     # DC 分量
#     h[0] = 0.0
#     if n % 2 == 0:
#         # 偶数长度：[1, N/2) → -i，N/2 → 0，(N/2, N) → +i
#         h[1:n // 2] = -1j
#         h[n // 2] = 0.0
#         h[n // 2 + 1:] = 1j
#     else:
#         # 奇数长度：[1, (N+1)/2) → -i，[(N+1)/2, N) → +i
#         half = (n + 1) // 2
#         h[1:half] = -1j
#         h[half:] = 1j
#
#     if x.ndim == 2:
#         h = h[:, np.newaxis]
#     return ifft(f * h, axis=0)
#
#
# def analytic_signal(x: np.ndarray) -> np.ndarray:
#     """
#     计算解析信号：S = x + i * H(x)。
#
#     Parameters
#     ----------
#     x : 实数时间序列
#
#     Returns
#     -------
#     复数解析信号
#     """
#     return x + 1j * hilbert_transform(x)
#
#
# def instantaneous_phase(x: np.ndarray) -> np.ndarray:
#     """
#     提取瞬时相位（单位模复数）：e^{iθ} = S / |S|。
#
#     用于相位互相关（PCC），参见 Ventosa et al., 2019。
#
#     Parameters
#     ----------
#     x : 实数时间序列，shape (N,) 或 (N, M)
#
#     Returns
#     -------
#     单位模复数数组，shape 与输入相同
#     """
#     s = analytic_signal(x)
#     return s / np.abs(s)
#
#
# # ─────────────────────────────────────────────
# # 6. FFT 计算
# # ─────────────────────────────────────────────
#
# def compute_rfft(x: np.ndarray) -> np.ndarray:
#     """
#     计算窗口化实数 FFT（单边谱）。
#
#     Parameters
#     ----------
#     x : 时间序列矩阵，shape (N, M)，M 为窗口数
#
#     Returns
#     -------
#     rfft 结果，shape (N//2 + 1, M)
#     """
#     return rfft(x, axis=0)
#
#
# def compute_phase_fft(x: np.ndarray) -> np.ndarray:
#     """
#     计算瞬时相位（用于 PCC 流程）。
#
#     Parameters
#     ----------
#     x : 时间序列矩阵，shape (N, M)
#
#     Returns
#     -------
#     瞬时相位数组（复数），shape (N, M)
#     """
#     return instantaneous_phase(x)