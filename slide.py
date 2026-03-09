"""
slide.py
========
将预处理后的连续波形按滑动窗口切割，构建用于互相关计算的数据字典。
对应 Julia SeisNoise: slide() + RawData(S, cc_len, cc_step)
"""

import numpy as np
import obspy
from tool import detrend, taper


# ─────────────────────────────────────────────
# 1. 核心切窗函数
# ─────────────────────────────────────────────

def slide(x: np.ndarray, fs: float, cc_len: float, cc_step: float,
          starttime: float, endtime: float) -> tuple:
    """
    将连续波形切割为滑动窗口矩阵。
    对应 Julia SeisNoise: slide(A, cc_len, cc_step, fs, starttime, endtime)

    Parameters
    ----------
    x         : 预处理后连续波形，shape (N,)
    fs        : 采样率 (Hz)
    cc_len    : 窗口长度（秒）
    cc_step   : 滑动步长（秒）
    starttime : 数据起始 epoch 秒
    endtime   : 数据结束 epoch 秒

    Returns
    -------
    out    : 切窗矩阵，shape (window_samples, n_windows)
    starts : 每个窗口起始 epoch 秒，shape (n_windows,)
    """
    window_samples = int(cc_len * fs)

    # 对应 Julia: starts = range(starttime, stop=endtime, step=cc_step)
    starts = np.arange(starttime, endtime, cc_step)
    ends   = starts + cc_len - 1.0 / fs

    # 对应 Julia: ind = findlast(x -> x <= endtime, ends)
    ind    = np.searchsorted(ends, endtime, side='right')
    starts = starts[:ind]
    ends   = ends[:ind]

    if len(starts) == 0:
        raise ValueError(
            f"没有完整窗口：数据长度 {endtime - starttime:.1f}s < cc_len {cc_len}s"
        )

    # 无重叠快速路径，对应 Julia reshape 分支
    if cc_step == cc_len:
        n_complete = len(x) // window_samples
        out = (x[:n_complete * window_samples]
               .reshape(window_samples, n_complete, order='F')
               .copy())
        return out, starts[:n_complete]

    # 通用重叠路径，对应 Julia 逐窗口切割
    # 对应 Julia: s = round.((hcat(starts,ends) .- starttime) .* fs .+ 1.)
    s   = np.round((np.column_stack([starts, ends]) - starttime) * fs).astype(int)
    out = np.zeros((window_samples, len(starts)), dtype=np.float64)

    for ii in range(len(starts)):
        out[:, ii] = x[s[ii, 0] : s[ii, 1] + 1]

    return out, starts


# ─────────────────────────────────────────────
# 2. 单台数据字典
# ─────────────────────────────────────────────

def make_raw_data(x: np.ndarray, fs: float, cc_len: float, cc_step: float,
                  starttime: float, endtime: float, freqmin: float = 0.0,
                  freqmax: float = 0.0, net: str = "", sta: str = "",
                  loc: str = "", cha: str = "", detrend_win: bool = True,
                  taper_win: bool = True) -> dict:
    """
    slide 切窗 + 每窗口 detrend/taper，返回单台数据字典。
    对应 Julia: RawData(S, cc_len, cc_step) + detrend!.(R) + taper!.(R)

    Parameters
    ----------
    x           : 预处理后连续波形，shape (N,)
    fs          : 采样率 (Hz)
    cc_len      : 窗口长度（秒）
    cc_step     : 滑动步长（秒）
    starttime   : 数据起始 epoch 秒
    endtime     : 数据结束 epoch 秒
    freqmin/max : 分析频带（记录用，Hz）
    net/sta/loc/cha : 台站信息
    detrend_win : 每窗口去趋势
    taper_win   : 每窗口锥形窗

    Returns
    -------
    dict : 'x' (window_samples, n_windows) | 'fs' | 'cc_len' | 'cc_step'
           't' (n_windows,) | 'freqmin' | 'freqmax' | 'net','sta','loc','cha'
    """
    out, starts = slide(x, fs, cc_len, cc_step, starttime, endtime)

    for i in range(out.shape[1]):
        if detrend_win:
            out[:, i] = detrend(out[:, i])
        if taper_win:
            out[:, i] = taper(out[:, i], fs)

    return {'x': out, 'fs': fs, 'cc_len': cc_len, 'cc_step': cc_step,
            't': starts, 'freqmin': freqmin, 'freqmax': freqmax,
            'net': net, 'sta': sta, 'loc': loc, 'cha': cha}


# ─────────────────────────────────────────────
# 3. 双台切窗（CCF 专用）
# ─────────────────────────────────────────────

def make_raw_data_pair(x1: np.ndarray, x2: np.ndarray, fs: float,
                       cc_len: float, cc_step: float,
                       starttime1: float, endtime1: float,
                       starttime2: float, endtime2: float,
                       freqmin: float = 0.0, freqmax: float = 0.0,
                       net1: str = "", sta1: str = "", loc1: str = "", cha1: str = "",
                       net2: str = "", sta2: str = "", loc2: str = "", cha2: str = "",
                       detrend_win: bool = True, taper_win: bool = True) -> tuple:
    """
    对两台波形数据分别切窗，保留时间戳对齐的共同窗口。
    对应 Julia: R = RawData.([S1, S2], cc_len, cc_step) + sync()

    两台数据窗口起始时间取交集，确保互相关时窗口严格对应。

    Parameters
    ----------
    x1/x2             : 两台预处理后波形，shape (N,)
    fs                : 采样率 (Hz)，两台必须相同
    cc_len            : 窗口长度（秒）
    cc_step           : 滑动步长（秒）
    starttime1/2      : 两台起始 epoch 秒
    endtime1/2        : 两台结束 epoch 秒
    freqmin/max       : 分析频带（记录用，Hz）
    net/sta/loc/cha 1/2 : 两台台站信息
    detrend_win       : 每窗口去趋势
    taper_win         : 每窗口锥形窗

    Returns
    -------
    raw1, raw2 : 两个时间对齐的数据字典，窗口数相同，t 完全一致
    """
    out1, starts1 = slide(x1, fs, cc_len, cc_step, starttime1, endtime1)
    out2, starts2 = slide(x2, fs, cc_len, cc_step, starttime2, endtime2)

    # 时间戳取交集，对应 Julia sync()
    t1_set   = set(np.round(starts1, 4))
    t2_set   = set(np.round(starts2, 4))
    common_t = sorted(t1_set & t2_set)

    if len(common_t) == 0:
        raise ValueError(
            f"两台数据无共同时间窗口：\n"
            f"  {net1}.{sta1}.{cha1}: "
            f"{obspy.UTCDateTime(starttime1)} ~ {obspy.UTCDateTime(endtime1)}\n"
            f"  {net2}.{sta2}.{cha2}: "
            f"{obspy.UTCDateTime(starttime2)} ~ {obspy.UTCDateTime(endtime2)}"
        )

    # 筛选共同窗口
    common_set = set(common_t)
    idx1 = [i for i, t in enumerate(np.round(starts1, 4)) if t in common_set]
    idx2 = [i for i, t in enumerate(np.round(starts2, 4)) if t in common_set]

    out1 = out1[:, idx1]
    out2 = out2[:, idx2]
    t    = np.array(common_t)

    # 每窗口 detrend + taper
    for i in range(out1.shape[1]):
        if detrend_win:
            out1[:, i] = detrend(out1[:, i])
            out2[:, i] = detrend(out2[:, i])
        if taper_win:
            out1[:, i] = taper(out1[:, i], fs)
            out2[:, i] = taper(out2[:, i], fs)

    print(f"  {net1}.{sta1}.{cha1} 原始窗口: {len(starts1)}")
    print(f"  {net2}.{sta2}.{cha2} 原始窗口: {len(starts2)}")
    print(f"  对齐后共同窗口: {len(t)}")

    raw1 = {'x': out1, 'fs': fs, 'cc_len': cc_len, 'cc_step': cc_step,
            't': t, 'freqmin': freqmin, 'freqmax': freqmax,
            'net': net1, 'sta': sta1, 'loc': loc1, 'cha': cha1}
    raw2 = {'x': out2, 'fs': fs, 'cc_len': cc_len, 'cc_step': cc_step,
            't': t, 'freqmin': freqmin, 'freqmax': freqmax,
            'net': net2, 'sta': sta2, 'loc': loc2, 'cha': cha2}

    return raw1, raw2