"""
vmd_hht_dvv.py
==============
基于 VMD-HHT 的地震波速变化 (dv/v) 计算模块。

算法流程：
    1. VMD 分解参考和当前波形
    2. 逐模态估算主频 f_c（低成本）
    3. 频带预筛选：f_c 不在目标频带内的模态直接跳过
    4. 精筛：对通过频带预筛的模态计算 Coherency + dv/v
    5. 能量加权融合通过精筛的模态

公开接口：
    run_vmd(trace, fs, K, alpha, verbose)
        → np.ndarray (K, N)

    compute_dvv_hht_single(ref_imf, cur_imf, fs, time, ...)
        → dict

    adaptive_imf_selection(imf_results, ...)
        → list

    dvv_vmd_hht(trace_ref, trace_cur, fs, time, ...)
        → dict

依赖：numpy, scipy, vmdpy

作者：（your name）
版本：3.0
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from vmdpy import VMD



# ══════════════════════════════════════════════════════════════════════════════
# 内部工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_fc(imf: np.ndarray,
                 fs: float) -> tuple[float, np.ndarray, np.ndarray]:
    """
    估算单个 IMF/模态的能量加权平均主频 f_c。

    Parameters
    ----------
    imf : np.ndarray
        单个模态时间序列（1D）
    fs : float
        采样率（Hz）

    Returns
    -------
    f_c : float
        能量加权平均主频（Hz）
    inst_freq : np.ndarray
        瞬时频率序列（Hz），长度与 imf 相同
    z : np.ndarray
        解析信号（复数），长度与 imf 相同

    Raises
    ------
    RuntimeError
        若瞬时频率全部超出合理范围 (0.1, fs/2)
    """
    dt        = 1.0 / fs
    z         = hilbert(imf)
    phase     = np.unwrap(np.angle(z))
    inst_freq = np.gradient(phase) / (2.0 * np.pi * dt)
    amp       = np.abs(z)

    mask = (inst_freq > 0.1) & (inst_freq < fs / 2.0)
    if mask.sum() == 0:
        raise RuntimeError("模态瞬时频率全部超出合理范围 (0.1, fs/2)")

    f_c = float(np.average(inst_freq[mask], weights=amp[mask] ** 2))
    return f_c, inst_freq, z


def _smooth_cross_spectrum(z_ref: np.ndarray,
                           z_cur: np.ndarray,
                           win_pts: int) -> tuple[np.ndarray, np.ndarray]:
    """
    计算并平滑互相关解析信号。

    Parameters
    ----------
    z_ref : np.ndarray
        参考波形解析信号
    z_cur : np.ndarray
        当前波形解析信号
    win_pts : int
        平滑窗口点数

    Returns
    -------
    z_cross_smooth : np.ndarray
        平滑后的互相关复信号
    amp_smooth : np.ndarray
        平滑后的模长（能量包络）
    """
    window        = np.ones(win_pts) / win_pts
    z_cross       = z_ref * np.conj(z_cur)
    z_cross_smooth = (
        np.convolve(np.real(z_cross), window, mode='same') +
        1j * np.convolve(np.imag(z_cross), window, mode='same')
    )
    return z_cross_smooth, np.abs(z_cross_smooth)


# def _unwrap_with_protection(z_cross_smooth: np.ndarray,
#                              amp_smooth: np.ndarray,
#                              amp_percentile: float) -> np.ndarray:
#     """
#     对平滑互相关信号进行相位解卷绕，并对低能量区域做插值保护。
#
#     先 unwrap 再对低能量点插值，避免在 wrapped 相位上插值导致
#     跨越 ±π 的 2π 信息丢失。
#
#     Parameters
#     ----------
#     z_cross_smooth : np.ndarray
#         平滑后的互相关复信号
#     amp_smooth : np.ndarray
#         平滑后的模长
#     amp_percentile : float
#         低能量保护百分位（%）
#
#     Returns
#     -------
#     dp_safe : np.ndarray
#         解卷绕并插值保护后的相位差序列（rad）
#     """
#     dp_unwrap  = np.unwrap(np.angle(z_cross_smooth))
#     amp_thresh = np.percentile(amp_smooth, amp_percentile)
#     valid      = amp_smooth > amp_thresh
#     n_inv      = (~valid).sum()
#
#     if n_inv > 0 and n_inv < len(valid):
#         vidx    = np.where(valid)[0]
#         dp_safe = np.interp(
#             np.arange(len(dp_unwrap)), vidx, dp_unwrap[vidx]
#         )
#     else:
#         dp_safe = dp_unwrap
#
#     return dp_safe


def _compute_r2_phase_integral(inst_freq: np.ndarray,
                                dp_safe: np.ndarray,
                                mask_roi: np.ndarray,
                                time_roi: np.ndarray,
                                dt: float) -> tuple[float, float]:
    """
    计算 ROI 内相位差对累积相位积分 ∫f(t)dt 的线性拟合 R²。

    Parameters
    ----------
    inst_freq : np.ndarray
        瞬时频率序列（Hz）
    dp_safe : np.ndarray
        相位差序列（rad），与 inst_freq 等长
    mask_roi : np.ndarray
        ROI 布尔掩码
    time_roi : np.ndarray
        ROI 时间轴（仅用于长度校验）
    dt : float
        采样间隔（s）

    Returns
    -------
    r2 : float
        线性拟合决定系数 R²，范围 [0, 1]
    slope : float
        拟合斜率（无量纲，物理意义约等于 dv/v）
    """
    # ∫f(t)dt → rad
    phase_integral     = 2.0 * np.pi * np.cumsum(inst_freq) * dt
    phase_integral_roi = phase_integral[mask_roi]
    dp_roi             = dp_safe[mask_roi]

    # ROI 内再次 unwrap，避免截断造成跳变
    dp_roi = np.unwrap(dp_roi)

    # ROI 相位积分归零
    phase_integral_roi = phase_integral_roi - phase_integral_roi[0]

    finite_mask = np.isfinite(dp_roi) & np.isfinite(phase_integral_roi)

    if finite_mask.sum() <= 10:
        return 0.0, 0.0

    coef   = np.polyfit(phase_integral_roi[finite_mask],
                        dp_roi[finite_mask], 1)
    dp_fit = np.polyval(coef, phase_integral_roi[finite_mask])
    ss_res = np.var(dp_roi[finite_mask] - dp_fit)
    ss_tot = np.var(dp_roi[finite_mask])
    r2     = float(1.0 - ss_res / (ss_tot + 1e-30))
    slope  = float(coef[0])
    return r2, slope


def _compute_coherency(z_cross_smooth: np.ndarray,
                        mask_roi: np.ndarray) -> float:
    """
    计算 ROI 内的相干度（Coherency）。

    定义：C = |Σ z_cross_smooth| / Σ|z_cross_smooth|
    范围：[0, 1]，C < 0.5 表示相位随机，模态不可用。

    Parameters
    ----------
    z_cross_smooth : np.ndarray
        平滑后的互相关复信号
    mask_roi : np.ndarray
        ROI 布尔掩码

    Returns
    -------
    coherency : float
        相干度，范围 [0, 1]
    """
    z_roi = z_cross_smooth[mask_roi]
    return float(
        np.abs(np.sum(z_roi)) / (np.sum(np.abs(z_roi)) + 1e-30)
    )


def _compute_dvv_scalar(dvv_series: np.ndarray,
                         amp_smooth: np.ndarray,
                         mask_roi: np.ndarray,
                         sigma_clip: float) -> tuple[float, int]:
    """
    在 ROI 内对 dv/v 时间序列做能量加权平均，返回标量 dv/v。

    Parameters
    ----------
    dvv_series : np.ndarray
        逐时刻 dv/v 序列（%）
    amp_smooth : np.ndarray
        平滑能量包络（权重）
    mask_roi : np.ndarray
        ROI 布尔掩码
    sigma_clip : float
        异常值剔除的 sigma 倍数

    Returns
    -------
    scalar : float
        ROI 内能量加权平均 dv/v（%），失败时返回 np.nan
    valid_pts : int
        参与平均的有效点数
    """
    dvv_roi = dvv_series[mask_roi]
    wgt_roi = amp_smooth[mask_roi]
    med     = np.nanmedian(dvv_roi)
    std     = np.nanstd(dvv_roi)
    ok      = (np.abs(dvv_roi - med) < sigma_clip * std) & np.isfinite(dvv_roi)

    if ok.sum() == 0:
        return np.nan, 0

    scalar = float(np.average(dvv_roi[ok], weights=wgt_roi[ok]))
    return scalar, int(ok.sum())


def mute_zero_lag(trace: np.ndarray,
                  time: np.ndarray,
                  mute_end: float = 3.0,
                  taper_len: float = 2.0) -> np.ndarray:
    """
    静音 ACF 零延迟主峰，保留尾波散射部分。

    Parameters
    ----------
    trace : np.ndarray
        输入时间序列（1D），正滞后段（t >= 0）
    time : np.ndarray
        时间轴（s），与 trace 等长
    mute_end : float
        强制清零的结束时刻（s），默认 3.0s
        建议设为主频周期的 3~5 倍：
            1Hz 信号 → 3~5s
            2Hz 信号 → 1.5~2.5s
    taper_len : float
        清零段结束后的渐变过渡长度（s），默认 2.0s
        使用半个 Tukey 窗平滑过渡，避免截断产生高频伪信号

    Returns
    -------
    trace_muted : np.ndarray
        主峰静音后的时间序列，与输入等长

    Notes
    -----
    处理示意：
        t=0 ~ mute_end         → 强制归零
        t=mute_end ~ mute_end+taper_len → 渐变（0→1 的半 Tukey 斜坡）
        t > mute_end+taper_len → 保持原始值
    """
    trace_muted = trace.copy()
    dt          = time[1] - time[0]

    # ── 强制清零段 ─────────────────────────────────────────────
    mute_mask              = time <= mute_end
    trace_muted[mute_mask] = 0.0

    # ── 渐变过渡段（半 Tukey 斜坡，0 → 1）────────────────────
    taper_start = mute_end
    taper_stop  = mute_end + taper_len
    taper_mask  = (time > taper_start) & (time <= taper_stop)

    if taper_mask.sum() > 0:
        n_taper = taper_mask.sum()
        # 0 → 1 的余弦斜坡
        ramp    = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_taper) / n_taper))
        trace_muted[taper_mask] *= ramp

    return trace_muted


# def _unwrap_with_protection(z_cross_smooth: np.ndarray,
#                              amp_smooth: np.ndarray,
#                              amp_percentile: float,
#                              mask_roi: np.ndarray = None) -> np.ndarray:
#     """
#     只在 ROI 内 unwrap，ROI 外用线性外推填充，
#     彻底避免 ROI 外的瞬态污染 ROI 内的相位积分。
#     """
#     dp_raw    = np.angle(z_cross_smooth)   # ∈ (-π, π]，wrapped
#     dp_result = dp_raw.copy()
#
#     if mask_roi is not None and mask_roi.sum() > 3:
#         # 只对 ROI 内做 unwrap
#         roi_idx           = np.where(mask_roi)[0]
#         dp_roi_wrapped    = dp_raw[roi_idx]
#         dp_roi_unwrapped  = np.unwrap(dp_roi_wrapped)
#
#         # 低能量点插值保护（仅在 ROI 内）
#         amp_roi   = amp_smooth[roi_idx]
#         thresh    = np.percentile(amp_roi, amp_percentile)
#         valid_roi = amp_roi > thresh
#
#         if valid_roi.sum() > 3 and (~valid_roi).sum() > 0:
#             vidx             = roi_idx[valid_roi]
#             dp_roi_unwrapped = np.interp(
#                 roi_idx, vidx, dp_roi_unwrapped[valid_roi]
#             )
#
#         dp_result[mask_roi] = dp_roi_unwrapped
#         # ROI 外用 ROI 边界值填充（不影响最终结果，只是避免 NaN）
#         dp_result[:roi_idx[0]]  = dp_roi_unwrapped[0]
#         dp_result[roi_idx[-1]+1:] = dp_roi_unwrapped[-1]
#     else:
#         # fallback：全局 unwrap
#         dp_unwrap  = np.unwrap(dp_raw)
#         amp_thresh = np.percentile(amp_smooth, amp_percentile)
#         valid      = amp_smooth > amp_thresh
#         if (~valid).sum() > 0 and valid.sum() > 0:
#             vidx      = np.where(valid)[0]
#             dp_result = np.interp(np.arange(len(dp_unwrap)),
#                                   vidx, dp_unwrap[vidx])
#         else:
#             dp_result = dp_unwrap
#
#     return dp_result

def _unwrap_with_protection(z_cross_smooth: np.ndarray,
                             amp_smooth: np.ndarray,
                             amp_percentile: float,
                             mask_roi: np.ndarray = None) -> np.ndarray:
    """
    鲁棒相位解卷绕。
    核心逻辑：先对高能量骨架点做逐步unwrap（跳过大间距断点），
    再插值填充低能量坑，避免噪声引发的2π累积污染。
    """
    dp_raw    = np.angle(z_cross_smooth)
    dp_result = dp_raw.copy()

    def _robust_unwrap(dp_seg: np.ndarray,
                       amp_seg: np.ndarray,
                       percentile: float) -> np.ndarray:
        """对一段序列做高能量骨架unwrap + 插值填充。"""
        thresh     = np.percentile(amp_seg, max(percentile, 20.0))
        valid_mask = amp_seg > thresh

        if valid_mask.sum() < 4:
            return np.unwrap(dp_seg)   # 退化

        valid_idx = np.where(valid_mask)[0]
        dp_valid  = dp_seg[valid_mask].copy()

        # 逐步unwrap：仅当相邻骨架点间距 ≤ 中位间距*3 时才信任2π判断
        gaps       = np.diff(valid_idx).astype(float)
        gap_thresh = np.median(gaps) * 3.0

        dp_unwrapped   = dp_valid.copy()
        cumulative_2pi = 0.0
        for j in range(1, len(valid_idx)):
            diff = dp_valid[j] - dp_valid[j - 1]
            if gaps[j - 1] <= gap_thresh:
                # 间距可信：正常累积2π修正
                if diff > np.pi:
                    cumulative_2pi -= 2.0 * np.pi
                elif diff < -np.pi:
                    cumulative_2pi += 2.0 * np.pi
            # 间距过大：不可信，不做2π修正，避免错误累积
            dp_unwrapped[j] = dp_valid[j] + cumulative_2pi

        # 插值回完整序列
        return np.interp(np.arange(len(dp_seg)), valid_idx, dp_unwrapped)

    if mask_roi is not None and mask_roi.sum() > 3:
        roi_idx = np.where(mask_roi)[0]

        dp_roi_result = _robust_unwrap(
            dp_raw[roi_idx], amp_smooth[roi_idx], amp_percentile
        )

        dp_result[mask_roi]         = dp_roi_result
        dp_result[:roi_idx[0]]      = dp_roi_result[0]
        dp_result[roi_idx[-1] + 1:] = dp_roi_result[-1]

    else:
        dp_result = _robust_unwrap(dp_raw, amp_smooth, amp_percentile)

    return dp_result



# ══════════════════════════════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════════════════════════════

def run_vmd(trace: np.ndarray,
            fs: float,
            K: int = 5,
            alpha: float = 2000,
            verbose: bool = True) -> np.ndarray:
    """
    使用 VMD（变分模态分解）对时间序列进行分解。

    Parameters
    ----------
    trace : np.ndarray
        输入时间序列（1D）
    fs : float
        采样率（Hz）
    K : int
        预设模态数量，建议 4~6，默认 5
    alpha : float
        带宽约束惩罚因子。越大则各模态频带越窄越纯粹，默认 2000
    verbose : bool
        是否打印分解结果，默认 True

    Returns
    -------
    imfs : np.ndarray, shape (K, N)
        VMD 分解出的所有模态，已逆序排列（高频→低频），
        与原 EMD 输出顺序一致，兼容下游代码逻辑。

    Notes
    -----
    VMD 固定参数：
        tau  = 0.0   噪声容忍度（干净信号）
        DC   = 0     不强制包含直流分量
        init = 1     均匀分布初始化中心频率
        tol  = 1e-7  收敛容差
    """
    trace = np.asarray(trace, dtype=float).ravel()

    # VMD 固定参数
    tau  = 0.0
    DC   = 0
    init = 1
    tol  = 1e-7

    u, u_hat, omega = VMD(trace, alpha, tau, K, DC, init, tol)

    # VMD 默认低频→高频，逆序使其高频→低频，兼容原 EMD 逻辑
    imfs = u[::-1, :]

    if verbose:
        print(f"✅ VMD 完成：提取了 {K} 个模态，信号长度 {len(trace)} 点")
        # 打印各模态中心频率（omega 最后一列为收敛后的中心频率，单位为归一化频率）
        center_freqs = omega[:, -1] * fs
        # omega 同样需要逆序对应 imfs
        center_freqs_sorted = center_freqs[::-1]
        for i, fc in enumerate(center_freqs_sorted):
            print(f"   Mode{i+1}: 中心频率 ≈ {fc:.3f} Hz")

    return imfs


from scipy.signal import butter, sosfiltfilt


def compute_dvv_hht_single(ref_imf: np.ndarray,
                            cur_imf: np.ndarray,
                            fs: float,
                            time: np.ndarray,
                            n_cycles: float = 3.0,
                            lag_band: tuple = (20.0, 80.0),
                            amp_percentile: float = 5.0,
                            sigma_clip: float = 3.0) -> dict:
    """
    对单个 VMD 模态执行 HHT dv/v 计算（纯粹版，无静音逻辑）。
    静音预处理已在 dvv_vmd_hht 的 VMD 分解之前完成。
    """
    dt       = 1.0 / fs
    time     = np.asarray(time, dtype=float).ravel()
    mask_roi = (time >= lag_band[0]) & (time <= lag_band[1])

    # 1. f_c + 解析信号
    f_c, inst_freq, z_ref = _estimate_fc(ref_imf, fs)
    z_cur                 = hilbert(cur_imf)

    # 2. 复平面平滑
    win_pts                    = max(int(n_cycles / f_c * fs), 3)
    z_cross_smooth, amp_smooth = _smooth_cross_spectrum(z_ref, z_cur, win_pts)

    # 3. Coherency
    coherency = _compute_coherency(z_cross_smooth, mask_roi)

    # # 4. unwrap + 低能量保护
    # dp_safe = _unwrap_with_protection(z_cross_smooth, amp_smooth, amp_percentile)
    dp_safe = _unwrap_with_protection(z_cross_smooth, amp_smooth,
                                      amp_percentile, mask_roi=mask_roi)

    # 5. 逐时刻 dv/v
    t_safe     = np.where(np.abs(time) > 1e-10, time, np.nan)
    dvv_series = -(dp_safe / (2.0 * np.pi * f_c)) / t_safe * 100.0

    # 6. ROI 内 3σ 标量提取
    scalar, valid_pts = _compute_dvv_scalar(
        dvv_series, amp_smooth, mask_roi, sigma_clip
    )

    return {
        'f_c'       : f_c,
        'dvv'       : scalar,
        'dvv_series': dvv_series,
        'coherency' : coherency,
        'amp_smooth': amp_smooth,
        'energy'    : float(np.sum(ref_imf ** 2)),
        'valid_pts' : valid_pts,
    }

# def compute_dvv_hht_single(ref_imf: np.ndarray,
#                             cur_imf: np.ndarray,
#                             fs: float,
#                             time: np.ndarray,
#                             n_cycles: float = 3.0,
#                             lag_band: tuple = (20.0, 80.0),
#                             amp_percentile: float = 5.0,
#                             sigma_clip: float = 3.0) -> dict:
#     """
#     对单个 VMD 模态执行 HHT dv/v 计算。
#     修复：用逐时刻瞬时频率 inst_freq 替换固定 f_c 计算 dv/v，
#     消除模态频率时变时的系统误差。
#     """
#     dt       = 1.0 / fs
#     time     = np.asarray(time, dtype=float).ravel()
#     mask_roi = (time >= lag_band[0]) & (time <= lag_band[1])
#
#     # 1. f_c + 解析信号（参考）
#     f_c, inst_freq_ref, z_ref = _estimate_fc(ref_imf, fs)
#
#     # 当前模态解析信号 + 瞬时频率
#     z_cur                          = hilbert(cur_imf)
#     phase_cur                      = np.unwrap(np.angle(z_cur))
#     inst_freq_cur                  = np.gradient(phase_cur) / (2.0 * np.pi * dt)
#
#     # 两路瞬时频率取平均作为局部频率估计（更稳健）
#     inst_freq_mean = 0.5 * (inst_freq_ref + inst_freq_cur)
#
#     # 异常频率保护：超出合理范围的点用 f_c 替代
#     freq_valid = (inst_freq_mean > 0.1) & (inst_freq_mean < fs / 2.0)
#     inst_freq_safe = np.where(freq_valid, inst_freq_mean, f_c)
#
#     # 2. 复平面平滑
#     win_pts                    = max(int(n_cycles / f_c * fs), 3)
#     z_cross_smooth, amp_smooth = _smooth_cross_spectrum(z_ref, z_cur, win_pts)
#
#     # 3. Coherency
#     coherency = _compute_coherency(z_cross_smooth, mask_roi)
#
#     # 4. unwrap + 低能量保护（仅在ROI内）
#     dp_safe = _unwrap_with_protection(z_cross_smooth, amp_smooth,
#                                       amp_percentile, mask_roi=mask_roi)
#
#     # 5. 逐时刻 dv/v（使用逐时刻瞬时频率，消除固定f_c的系统误差）
#     t_safe     = np.where(np.abs(time) > 1e-10, time, np.nan)
#     # 核心修改：分母由固定 f_c 改为逐时刻 inst_freq_safe
#     dvv_series = -(dp_safe / (2.0 * np.pi * inst_freq_safe)) / t_safe * 100.0
#
#     # 6. ROI 内 3σ 标量提取
#     scalar, valid_pts = _compute_dvv_scalar(
#         dvv_series, amp_smooth, mask_roi, sigma_clip
#     )
#
#     return {
#         'f_c'       : f_c,
#         'dvv'       : scalar,
#         'dvv_series': dvv_series,
#         'coherency' : coherency,
#         'amp_smooth': amp_smooth,
#         'energy'    : float(np.sqrt(np.sum(ref_imf**2) * np.sum(cur_imf**2))),  # 几何平均能量
#         'valid_pts' : valid_pts,
#     }


def adaptive_imf_selection(imf_results: list,
                            dvv_abs_max: float = 1.0,
                            coherency_min: float = 0.7,
                            verbose: bool = True) -> list:
    """
    对已通过频带预筛的模态做精筛（Coherency + |dv/v|）。

    Parameters
    ----------
    imf_results : list of dict
        compute_dvv_hht_single 的返回值列表，
        每个元素须额外包含 'idx' 键（模态索引，从 0 起）
    dvv_abs_max : float
        |dv/v| 最大阈值（%），默认 1.0
    coherency_min : float
        Coherency 最低阈值，默认 0.7
    verbose : bool
        是否打印筛选结果，默认 True

    Returns
    -------
    valid_imfs : list of dict
        通过精筛的模态结果列表
    """
    valid_imfs = []

    if verbose:
        print(f"\n精筛条件: |dv/v| <= {dvv_abs_max}%  |  C >= {coherency_min}")
        print("-" * 60)

    for r in imf_results:
        pass_dvv = (not np.isnan(r['dvv'])) and (abs(r['dvv']) <= dvv_abs_max)
        pass_coh = r['coherency'] >= coherency_min
        passed   = pass_dvv and pass_coh

        if passed:
            valid_imfs.append(r)

        if verbose:
            flag = "✅ PASS" if passed else "❌ FAIL"
            print(f"  Mode{r['idx']+1} [{flag}]  "
                  f"C={r['coherency']:.4f}{'✅' if pass_coh else '❌'}  "
                  f"|dv/v|={abs(r['dvv']):.4f}%{'✅' if pass_dvv else '❌'}  "
                  f"→ dv/v={r['dvv']:.5f}%")

    if verbose:
        print(f"\n精筛结果: {len(valid_imfs)} / {len(imf_results)} 个模态通过")

    return valid_imfs


def dvv_vmd_hht(trace_ref: np.ndarray,
                trace_cur: np.ndarray,
                fs: float,
                time: np.ndarray,
                freq_band: tuple = (1.0, 3.0),
                n_cycles: float = 3.0,
                lag_band: tuple = (20.0, 80.0),
                amp_percentile: float = 5.0,
                sigma_clip: float = 3.0,
                dvv_abs_max: float = 1.0,
                coherency_min: float = 0.5,
                K: int = 5,
                alpha: float = 2000,
                mode: str = 'coda',
                mute_end: float = 3.0,
                taper_len: float = 2.0,
                verbose: bool = True) -> dict:
    """
    完整流程：VMD → 频带预筛 → 精筛 → 能量加权融合。

    Parameters
    ----------
    trace_ref : np.ndarray
        参考波形（1D）
    trace_cur : np.ndarray
        当前波形（1D）
    fs : float
        采样率（Hz）
    time : np.ndarray
        时间轴（s），为尾波的滞后时间
    freq_band : tuple
        目标频带 (f_min, f_max)（Hz），用于频带预筛
    n_cycles : float
        各模态平滑窗口的主周期数，默认 3.0
    lag_band : tuple
        尾波提取窗口 (t_min, t_max)（s）
    amp_percentile : float
        低能量保护百分位（%），默认 5.0
    sigma_clip : float
        异常值剔除 sigma 倍数，默认 3.0
    dvv_abs_max : float
        精筛 |dv/v| 上限（%），默认 1.0
    coherency_min : float
        精筛 Coherency 阈值，默认 0.5
    K : int
        VMD 模态数量，默认 5
    alpha : float
        VMD 带宽约束惩罚因子，默认 2000
    verbose : bool
        是否打印过程信息，默认 True

    Returns
    -------
    dict，包含：
        dvv          : float   最终融合 dv/v（%），失败时为 np.nan
        dvv_series   : list    各有效模态的逐时刻 dv/v 曲线
        imf_results  : list    进入精筛的模态完整结果
        valid_imfs   : list    通过精筛的模态结果
        imfs_ref     : ndarray 参考波形 VMD 分解结果 (K, N)
        imfs_cur     : ndarray 当前波形 VMD 分解结果 (K, N)
        n_candidate  : int     通过频带预筛的模态数量
        n_valid      : int     通过精筛的模态数量

    Example
    -------
    >>> result = dvv_vmd_hht(
    ...     trace_ref, trace_cur,
    ...     fs        = 500.0,
    ...     time      = time,
    ...     freq_band = (1.0, 3.0),
    ...     lag_band  = (20.0, 80.0),
    ...     K         = 5,
    ...     alpha     = 2000,
    ... )
    >>> print(f"dv/v = {result['dvv']:.4f} %")
    >>> print(f"有效模态: {[r['idx']+1 for r in result['valid_imfs']]}")
    """
    trace_ref = np.asarray(trace_ref, dtype=float).ravel()
    trace_cur = np.asarray(trace_cur, dtype=float).ravel()
    time      = np.asarray(time,      dtype=float).ravel()
    assert len(trace_ref) == len(trace_cur) == len(time), \
        "trace_ref、trace_cur、time 长度必须一致"

    _sep = "=" * 60

    # ── Step 1: 预处理 + VMD 分解 ────────────────────────────
    if verbose:
        print(_sep)
        print(f"VMD-HHT dv/v  |  目标频带: {freq_band[0]}–{freq_band[1]} Hz  "
              f"|  K={K}, α={alpha}")
        print(_sep)
        print("\n[1/3] 预处理 + VMD 分解...")

    # ACF 模式：VMD 之前先静音主峰，避免主峰污染 VMD 分解
    if mode == 'acf':
        if verbose:
            print(f"  ACF 模式：VMD 前静音 0~{mute_end}s，过渡段 {taper_len}s")
        trace_ref_vmd = mute_zero_lag(trace_ref, time,
                                      mute_end=mute_end,
                                      taper_len=taper_len)
        trace_cur_vmd = mute_zero_lag(trace_cur, time,
                                      mute_end=mute_end,
                                      taper_len=taper_len)
    else:
        trace_ref_vmd = trace_ref
        trace_cur_vmd = trace_cur

    if verbose:
        print("  参考波形:")
    imfs_ref = run_vmd(trace_ref_vmd, fs, K=K, alpha=alpha, verbose=verbose)

    if verbose:
        print("  当前波形:")
    imfs_cur = run_vmd(trace_cur_vmd, fs, K=K, alpha=alpha, verbose=verbose)

    # ── VMD 输出长度修正 ──────────────────────────────────────────
    n_imfs_to_test = min(imfs_ref.shape[0], imfs_cur.shape[0])

    N_imf = imfs_ref.shape[1]
    if N_imf != len(time):
        if verbose:
            print(f"  ⚠️  VMD 输出长度 {N_imf} ≠ time 长度 {len(time)}，自动对齐")
        time = time[:N_imf]
        imfs_ref = imfs_ref[:, :N_imf]
        imfs_cur = imfs_cur[:, :N_imf]

    # ── Step 2: 频带预筛 ──────────────────────────────────────
    if verbose:
        print(f"\n[2/3] 频带预筛 → 精筛（共 {n_imfs_to_test} 个模态）...")
        print("-" * 60)

    imf_results = []
    n_candidate = 0

    for idx in range(n_imfs_to_test):

        # 频带预筛：只估算 f_c，成本极低
        try:
            f_c, inst_freq, z_ref = _estimate_fc(imfs_ref[idx], fs)
        except RuntimeError as e:
            if verbose:
                print(f"Mode{idx+1}: ⏭️  跳过（{e}）")
            continue

        in_band = freq_band[0] <= f_c <= freq_band[1]

        if not in_band:
            if verbose:
                print(f"Mode{idx+1}: ⏭️  跳过  "
                      f"f_c={f_c:.3f} Hz（频带外）")
            continue

        # 通过频带预筛，进入精筛计算
        n_candidate += 1
        if verbose:
            print(f"Mode{idx+1}: ✔️  进入精筛  f_c={f_c:.3f} Hz ✅")

        try:
            res = compute_dvv_hht_single(
                ref_imf        = imfs_ref[idx],
                cur_imf        = imfs_cur[idx],
                fs             = fs,
                time           = time,
                n_cycles       = n_cycles,
                lag_band       = lag_band,
                amp_percentile = amp_percentile,
                sigma_clip     = sigma_clip,
            )
            res['idx'] = idx
            imf_results.append(res)

        except Exception as e:
            if verbose:
                print(f"Mode{idx+1}: ⚠️  精筛计算失败（{e}）")

    # ── Step 3: 精筛 + 融合 ───────────────────────────────────
    if verbose:
        print(f"\n[3/3] 精筛 + 融合...")

    valid_imfs = adaptive_imf_selection(
        imf_results,
        dvv_abs_max   = dvv_abs_max,
        coherency_min = coherency_min,
        verbose       = verbose,
    )

    # 融合
    if len(valid_imfs) == 0:
        final_dvv = np.nan
        if verbose:
            print("⚠️  没有模态通过精筛，final_dvv = NaN")
            print("   建议：降低 coherency_min 或扩大 freq_band 或调整 K/alpha")

    elif len(valid_imfs) == 1:
        final_dvv = valid_imfs[0]['dvv']
        if verbose:
            print(f"\n单模态直接使用: Mode{valid_imfs[0]['idx']+1}")

    else:
        dvv_vals  = np.array([r['dvv']    for r in valid_imfs])
        wgts      = np.array([r['energy'] for r in valid_imfs])
        final_dvv = float(np.average(dvv_vals, weights=wgts))
        if verbose:
            print(f"\n能量加权融合 {len(valid_imfs)} 个模态: "
                  f"Mode{[r['idx']+1 for r in valid_imfs]}")

    if verbose:
        print(f"\n{_sep}")
        print(f"  频带预筛通过: {n_candidate} 个模态")
        print(f"  精筛通过:     {len(valid_imfs)} 个模态  "
              f"Mode{[r['idx']+1 for r in valid_imfs]}")
        print(f"  最终 dv/v:    {final_dvv:.5f} %")
        print(_sep)

    return {
        'dvv'        : final_dvv,
        'dvv_series' : [r['dvv_series'] for r in valid_imfs],
        'imf_results': imf_results,
        'valid_imfs' : valid_imfs,
        'imfs_ref'   : imfs_ref,
        'imfs_cur'   : imfs_cur,
        'n_candidate': n_candidate,
        'n_valid'    : len(valid_imfs),
    }

