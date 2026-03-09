"""
wxspectrum.py
=============
小波互谱法测量地震走时变化（dv/v）。
严格对照 Mao et al. (2020) GJI 的 MATLAB 代码 My_Wxspectrum_TO.m 编写。
"""

import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import convolve2d
import pywt


# ─────────────────────────────────────────────
# 1. 平滑函数，对应 MATLAB smoothCFS()
# ─────────────────────────────────────────────

def _smooth_cfs(cfs: np.ndarray, scales: np.ndarray, dt: float,
                ns: int, nt: float) -> np.ndarray:
    """
    对 CWT 系数做时间方向（Gaussian）+ 尺度方向（boxcar）平滑。
    """
    n_scales, N = cfs.shape
    npad = int(2 ** np.ceil(np.log2(N)))

    # 构造频域 omega 轴
    omega_pos = np.arange(1, npad // 2 + 1) * (2 * np.pi / npad)
    n_neg = int((npad - 1) / 2)
    omega_neg = -omega_pos[n_neg - 1:: -1]
    omega = np.concatenate([[0.], omega_pos, omega_neg])

    # 【核心修复点】: pywt 计算出的 scales 已经是样本点数 (samples)。
    # MATLAB 因为 scale 是秒，所以需要 scale/dt。在 Python 中直接使用即可！
    normscales = scales

    cfs_out = cfs.copy().astype(complex)

    # 时间方向：Gaussian 平滑（逐尺度）
    for k_s in range(n_scales):
        F = np.exp(-nt * (normscales[k_s] ** 2) * omega ** 2)
        smooth = ifft(F * fft(cfs_out[k_s, :], npad))
        cfs_out[k_s, :] = smooth[:N]

    # 尺度方向：boxcar 平滑
    if ns > 1:
        H = np.ones((ns, 1)) / ns
        cfs_out = convolve2d(cfs_out, H, mode='same')

    return cfs_out


# ─────────────────────────────────────────────
# 2. CWT 计算，调用 pywt
# ─────────────────────────────────────────────

def _compute_cwt(x: np.ndarray, fs: float, wname: str,
                 freq_limits: tuple, voices_per_octave: int,
                 extend_sig: bool) -> tuple:
    N = len(x)
    dt = 1.0 / fs

    # 对称延拓
    if extend_sig:
        x_ext = np.concatenate([x[::-1], x, x[::-1]])
        pad = N
    else:
        x_ext = x.copy()
        pad = 0

    fmin, fmax = freq_limits
    wavelet_map = {
        'amor': 'cmor1.5-1.0',
        'morse': 'cmor1.5-1.0',
        'bump': 'cmor1.0-0.5',
    }
    wavelet = wavelet_map.get(wname, 'cmor1.5-1.0')
    center_freq = pywt.central_frequency(wavelet)

    # 频率范围 → 尺度
    n_octaves = np.log2(fmax / fmin)
    n_scales = int(np.round(n_octaves * voices_per_octave))
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)[::-1]

    # pywt 中的 scale 本身就是相当于周期/dt 的样本点数量
    scales = center_freq / (freqs * dt)

    # 执行 CWT
    coefs_ext, _ = pywt.cwt(x_ext, scales, wavelet, sampling_period=dt)
    coefs = coefs_ext[:, pad:pad + N] if extend_sig else coefs_ext

    # COI（影响锥）
    coi_time = np.minimum(np.arange(1, N + 1), np.arange(N, 0, -1)) * dt
    coi = center_freq / (np.sqrt(2) * coi_time)
    coi = np.clip(coi, fmin, fmax)

    return coefs, freqs, scales, coi


# ─────────────────────────────────────────────
# 3. 主函数，对应 MATLAB My_Wxspectrum_TO()
# ─────────────────────────────────────────────

def wxspectrum(x_reference: np.ndarray,
               x_current: np.ndarray,
               fs: float,
               wname: str = 'amor',
               freq_limits: tuple = (0.5, 5.0),
               smoothing: bool = False,
               ns: int = 3,
               nt: float = 0.25,
               voices_per_octave: int = 10,
               extend_sig: bool = True,
               compute_coherence: bool = True,
               time_axis: np.ndarray = None) -> dict:
    x_reference = np.asarray(x_reference, dtype=float).ravel()
    x_current = np.asarray(x_current, dtype=float).ravel()
    N = len(x_reference)
    dt = 1.0 / fs

    cwt_ref, freqs, scales, coi = _compute_cwt(
        x_reference, fs, wname, freq_limits, voices_per_octave, extend_sig)
    cwt_cur, _, _, _ = _compute_cwt(
        x_current, fs, wname, freq_limits, voices_per_octave, extend_sig)

    invscales = (1.0 / scales)[:, np.newaxis]

    # 【核心修复点】: 恢复 ref * conj(cur)，确保 WXdt = t_cur - t_ref 符号正确
    if (not smoothing) or (ns == 1 and nt == 0):
        crossCFS = cwt_ref * np.conj(cwt_cur)
        WXamp = np.abs(crossCFS)
        WXspec = crossCFS

        if compute_coherence:
            cfs1 = _smooth_cfs(invscales * np.abs(cwt_cur) ** 2, scales, dt, ns, nt)
            cfs2 = _smooth_cfs(invscales * np.abs(cwt_ref) ** 2, scales, dt, ns, nt)
            cross_sm = _smooth_cfs(invscales * crossCFS, scales, dt, ns, nt)
            Wcoh = np.abs(cross_sm) ** 2 / (np.real(cfs1) * np.real(cfs2) + 1e-30)
            Wcoh = np.clip(Wcoh, 0.0, 1.0)  # 截断保护
        else:
            Wcoh = None
    else:
        cfs1 = _smooth_cfs(invscales * np.abs(cwt_cur) ** 2, scales, dt, ns, nt)
        cfs2 = _smooth_cfs(invscales * np.abs(cwt_ref) ** 2, scales, dt, ns, nt)
        crossCFS = cwt_ref * np.conj(cwt_cur)
        WXamp = np.abs(crossCFS)

        cross_sm = _smooth_cfs(invscales * crossCFS, scales, dt, ns, nt)

        # WXspec = cross_sm / (np.sqrt(np.real(cfs1) + 1e-30) * np.sqrt(np.real(cfs2) + 1e-30))
        # Wcoh = np.abs(cross_sm) ** 2 / (np.real(cfs1) * np.real(cfs2) + 1e-30)

        # 使用 np.abs() 消除浮点数负数误差
        WXspec = cross_sm / (np.sqrt(np.abs(cfs1) + 1e-30) * np.sqrt(np.abs(cfs2) + 1e-30))
        Wcoh = np.abs(cross_sm) ** 2 / (np.abs(cfs1) * np.abs(cfs2) + 1e-30)
        Wcoh = np.clip(Wcoh, 0.0, 1.0)

    # 走时变化 WXdt
    WXangle = np.angle(WXspec)
    WXdt = WXangle / (2.0 * np.pi * freqs[:, np.newaxis])

    if time_axis is not None:
        time = np.asarray(time_axis, dtype=float)
    else:
        time = np.arange(N) * dt

    result = {
        'WXspec': WXspec,
        'WXdt': WXdt,
        'WXamp': WXamp,
        'freq': freqs,
        'coi': coi,
        'time': time,
    }
    if Wcoh is not None:
        result['Wcoh'] = Wcoh

    return result


# ─────────────────────────────────────────────
# 4. dv/v 提取
# ─────────────────────────────────────────────

def dvv_from_wxdt(wxdt: np.ndarray,
                  wxamp: np.ndarray,
                  freqs: np.ndarray,
                  time: np.ndarray,
                  coi: np.ndarray,
                  freq_band: tuple = None,
                  lag_band: tuple = None,
                  amp_thresh: float = 0.0,
                  coi_mask: bool = True) -> dict:


    n_freq, n_time = wxdt.shape

    fmask = (freqs >= freq_band[0]) & (freqs <= freq_band[1]) if freq_band else np.ones(n_freq, dtype=bool)
    tmask = (np.abs(time) >= lag_band[0]) & (np.abs(time) <= lag_band[1]) if lag_band else np.ones(n_time, dtype=bool)

    if coi_mask:
        coi_valid = freqs[:, np.newaxis] >= coi[np.newaxis, :]
    else:
        coi_valid = np.ones((n_freq, n_time), dtype=bool)

    amp_valid = wxamp >= amp_thresh
    mask = (fmask[:, np.newaxis] & tmask[np.newaxis, :] & coi_valid & amp_valid)

    t_2d = time[np.newaxis, :]
    t_safe = np.where(np.abs(t_2d) > 1e-10, t_2d, np.nan)
    dvv_tf = -wxdt / t_safe * 100.0  # 单位 %

    weight = np.where(mask, wxamp, 0.0)
    weight_sum = weight.sum(axis=0)

    dvv_mean = np.where(
        weight_sum > 0,
        np.nansum(dvv_tf * weight, axis=0) / (weight_sum + 1e-30),
        np.nan
    )

    return {
        'dvv': dvv_mean,
        'dvv_freq': dvv_tf,
        'weight': weight,
        'mask': mask,
    }