"""
correlate.py
============
计算自相关函数（ACF）或互相关函数（CCF）并进行后处理。

支持两种输入模式：
  模式A（频域输入）：接收 process_fft / step2_normalize_whiten 返回的
                    FFT 字典，在频域直接相乘后 irfft，与 Julia 完全一致。

  模式B（时域输入）：接收 step2_normalize_whiten 返回的时域白化信号，
                    通过 FFT 加速的时域相关计算，数学等价于模式A。

对应 Julia SeisNoise:
  correlate(FFT1, FFT2, maxlag) → clean_up!(C) → abs_max!(C)
"""

import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from tool import detrend, taper, bandpass_filter
from preprocess import whiten_fft
from scipy.fft import rfft, irfft, rfftfreq, next_fast_len
# ═══════════════════════════════════════════════════════════════════
# 1. 核心相关内核（频域）
#    严格对应 Julia correlate()
# ═══════════════════════════════════════════════════════════════════

# def _correlate_freq(F1: np.ndarray, F2: np.ndarray,
#                     N: int, maxlag_pts: int) -> np.ndarray:
#     """
#     频域互相关内核，严格对应 Julia:
#       corrT = irfft(conj(FFT1) .* FFT2, N, 1)
#       t     = vcat(0:N/2-1, -N/2:-1)
#       ind   = findall(abs.(t) .<= maxlag)
#       return corrT[fftshift(ind), :]
#
#     Parameters
#     ----------
#     F1, F2     : 白化后 FFT，shape (Nfft//2+1, n_windows)
#     N          : irfft 目标长度 = int(round(cc_len * fs))
#     maxlag_pts : 最大滞后点数 = int(round(maxlag * fs))
#
#     Returns
#     -------
#     corr : shape (2*maxlag_pts+1, n_windows)，时间轴单调递增（-maxlag→+maxlag）
#     """
#
#
#     corrT = irfft(np.conj(F1) * F2, N, axis=0)   # shape (N, n_windows)
#
#     t   = np.concatenate([np.arange(0, N // 2),
#                           np.arange(-N // 2, 0)])
#
#     ind = np.where(np.abs(t) <= maxlag_pts)[0]
#     pos_start = maxlag_pts + 1
#     newind    = np.concatenate([ind[pos_start:],    # 负滞后
#                                 ind[:pos_start]])    # 正滞后（含零点）
#
#     return corrT[newind, :]

def _correlate_freq(F1: np.ndarray, F2: np.ndarray,
                    N: int, maxlag_pts: int) -> np.ndarray:
    """
    频域互相关内核，严格对应 Julia:
      corrT = irfft(conj(FFT1) .* FFT2, N, 1)

    修复：irfft 必须用 Nfft（与上游 rfft 时一致），不能用原始点数 N。
    F1/F2 是基于 Nfft=next_fast_len(N) 做的 rfft，
    若 irfft 用 N 则频率轴错位，导致系统性相位偏差。
    """
    # ✅ 从 F1 形状反推 Nfft，无需额外传参
    Nfft = (F1.shape[0] - 1) * 2

    # ✅ 第二参数改为 Nfft
    corrT = irfft(np.conj(F1) * F2, Nfft, axis=0)   # shape (Nfft, n_windows)

    # ✅ t 向量也基于 Nfft
    t   = np.concatenate([np.arange(0, Nfft // 2),
                          np.arange(-Nfft // 2, 0)])

    ind = np.where(np.abs(t) <= maxlag_pts)[0]
    pos_start = maxlag_pts + 1
    newind    = np.concatenate([ind[pos_start:],
                                ind[:pos_start]])

    return corrT[newind, :]



# ═══════════════════════════════════════════════════════════════════
# 2. 核心相关内核（时域）
#    与频域版本数学等价
# ═══════════════════════════════════════════════════════════════════

def _correlate_time(x1: np.ndarray, x2: np.ndarray,
                    N: int, maxlag_pts: int,
                    Nfft: int = None) -> np.ndarray:
    """
    时域互相关内核。

    关键：Nfft 必须与 step2 白化时用的 n_fft 一致，
    即 next_fast_len(N)，而不是 next_fast_len(2*N)。
    """
    if Nfft is None:
        Nfft = next_fast_len(N)  # 与 compute_fft 保持一致

    F1 = rfft(x1, Nfft, axis=0)
    F2 = rfft(x2, Nfft, axis=0)
    corrT = irfft(np.conj(F1) * F2, N, axis=0)  # shape (N, n_windows)

    t = np.concatenate([np.arange(0, N // 2),
                        np.arange(-N // 2, 0)])
    ind = np.where(np.abs(t) <= maxlag_pts)[0]

    pos_start = maxlag_pts + 1
    newind = np.concatenate([ind[pos_start:],
                             ind[:pos_start]])

    return corrT[newind, :]





# ═══════════════════════════════════════════════════════════════════
# 3. 后处理   对应  clean_up + abs_max
# ═══════════════════════════════════════════════════════════════════

def clean_up(corr_data: dict, freqmin: float, freqmax: float,
             max_length: float = 20.0) -> dict:
    """
    每列做 detrend + taper + bandpass。
    对应 Julia clean_up!(C, freqmin, freqmax; max_length=20.)
    """
    x  = corr_data['corr'].copy()
    fs = corr_data['fs']

    for i in range(x.shape[1]):
        x[:, i] = detrend(x[:, i])
        x[:, i] = taper(x[:, i], fs, max_length=max_length)
        x[:, i] = bandpass_filter(x[:, i], freqmin, freqmax, fs)

    return {**corr_data, 'corr': x,
            'freqmin': freqmin, 'freqmax': freqmax}


def abs_max(corr_data: dict) -> dict:
    """
    每列归一化到 [-1, 1]。
    对应 Julia abs_max!(C)
    """
    x      = corr_data['corr'].copy()
    maxabs = np.max(np.abs(x), axis=0)
    maxabs[maxabs == 0] = 1.0
    x     /= maxabs
    return {**corr_data, 'corr': x}


def whiten_acf(corr_data: dict,
               freqmin:       float = None,
               freqmax:       float = None,
               method:        str   = 'smoothed',
               smooth_half_win: int = 5,
               pad:           int   = 50) -> dict:
    """
    对ACF/CCF结果做谱白化。
    对应 clean_up 的位置，在相关计算后、abs_max 前调用。

    Parameters
    ----------
    corr_data : correlate() 返回的字典，含 'corr'/'fs'/'freqmin'/'freqmax'
    freqmin   : 白化频带下限，None 时使用 corr_data['freqmin']
    freqmax   : 白化频带上限，None 时使用 corr_data['freqmax']
    method    : 'standard' 或 'smoothed'
    """


    _fmin  = freqmin if freqmin is not None else corr_data['freqmin']
    _fmax  = freqmax if freqmax is not None else corr_data['freqmax']
    fs     = corr_data['fs']
    x      = corr_data['corr']          # shape (n_lags, n_windows)
    n_lags = x.shape[0]
    Nfft   = next_fast_len(n_lags)
    freq   = rfftfreq(Nfft, d=1.0 / fs)

    F = rfft(x, Nfft, axis=0)

    fft_tmp = {
        'fft'  : F,
        'freq' : freq,
        'fs'   : fs,
        'n_pts': n_lags,
        'n_fft': Nfft,
    }
    fft_tmp = whiten_fft(fft_tmp, _fmin, _fmax,
                         method          = method,
                         smooth_half_win = smooth_half_win,
                         pad             = pad)

    x_white = irfft(fft_tmp['fft'], Nfft, axis=0)[:n_lags, :]

    print(f"  whiten_acf: {method}  {_fmin}~{_fmax} Hz")

    return {**corr_data,
            'corr'    : x_white,
            'freqmin' : _fmin,
            'freqmax' : _fmax}


# ═══════════════════════════════════════════════════════════════════
# 4. 统一相关入口
# ═══════════════════════════════════════════════════════════════════

def correlate(data1: dict,
              maxlag: float,
              data2:           dict  = None,
              use_freq_domain: bool  = True,
              do_clean_up:     bool  = True,
              do_abs_max:      bool  = True,
              max_length:      float = 20.0,
              do_whiten_acf:      bool  = False,    # ← 新增：对ACF结果是否再做白化
              acf_freqmin:     float = None,     # ← 新增：ACF白化频带下限
              acf_freqmax:     float = None,     # ← 新增：ACF白化频带上限
              acf_whiten_method: str = 'smoothed', # ← 新增：ACF白化方法
              ) -> dict:
    """
    统一相关计算入口，支持 ACF 和 CCF，支持频域/时域两种路径。

    Parameters
    ----------
    data1           : process_fft 或 step2_normalize_whiten 返回的字典
                      频域路径需含 'fft' 字段
                      时域路径需含 'x' 字段（白化后时域信号）
    maxlag          : 最大滞后时间（秒）
    data2           : 第二台站数据字典，None 时做自相关（ACF）
    use_freq_domain : True  → 频域相乘（模式A，对应原 correlate.py，Julia 一致）
                      False → 时域相关（模式B，数学等价，需 'x' 字段）
    do_clean_up     : 是否执行 clean_up（detrend+taper+bandpass）
    do_abs_max      : 是否执行 abs_max（归一化到 [-1,1]）
    max_length      : clean_up taper 最大长度（秒）

    Returns
    -------
    CorrData 字典：
      'corr'   : shape (2*maxlag_pts+1, n_windows)
      'lags'   : shape (2*maxlag_pts+1,)，单位秒
      'fs'     : 采样率
      'maxlag' : 最大滞后时间
      't'      : 各窗口起始时间
      'type'   : 'ACF' 或 'CCF'
      ...其余元数据字段
    """
    fs         = data1['fs']
    cc_len     = data1['cc_len']
    N          = int(round(cc_len * fs))
    maxlag_pts = int(round(maxlag * fs))
    is_acf     = (data2 is None)
    corr_type  = 'ACF' if is_acf else 'CCF'

    # ── 相关计算 ──────────────────────────────────────────────────────
    if use_freq_domain:
        # 兼容两种上游输出：
        assert 'F_white' in data1 or 'fft' in data1, \
            "use_freq_domain=True 需要 'F_white'（preprocess_2）或 'fft'（process_fft）字段"
        F1   = data1.get('F_white', data1.get('fft'))
        F2   = F1 if is_acf else data2.get('F_white', data2.get('fft'))
        corr = _correlate_freq(F1, F2, N, maxlag_pts)

    else:
        assert 'x_white' in data1 or 'x' in data1, \
            "use_freq_domain=False 需要 'x_white' 或 'x' 字段"
        _x1 = data1.get('x_white', data1.get('x'))
        _x2 = _x1 if is_acf else data2.get('x_white', data2.get('x'))
        x1 = _x1[:N, :]
        x2 = _x2[:N, :]
        Nfft = data1.get('n_fft', next_fast_len(N))  # ← 用 step2 的 n_fft
        corr = _correlate_time(x1, x2, N, maxlag_pts, Nfft=Nfft)

    lags = np.arange(-maxlag_pts, maxlag_pts + 1) / fs

    print(f"  {corr_type}  mode={'freq' if use_freq_domain else 'time'}")
    print(f"  N={N}  maxlag_pts={maxlag_pts}  corr shape={corr.shape}")

    corr_data = {
        'corr'   : corr,
        'lags'   : lags,
        'fs'     : fs,
        'maxlag' : maxlag,
        't'      : data1['t'],
        'cc_len' : cc_len,
        'cc_step': data1['cc_step'],
        'freqmin': data1['freqmin'],
        'freqmax': data1['freqmax'],
        'net'    : data1['net'],
        'sta'    : data1['sta'],
        'loc'    : data1['loc'],
        'cha'    : data1['cha'],
        'type'   : corr_type,
    }

    # ── 后处理 ────────────────────────────────────────────────────────
    if do_clean_up:
        corr_data = clean_up(corr_data,
                             corr_data['freqmin'],
                             corr_data['freqmax'],
                             max_length=max_length)
        print(f"  clean_up: detrend+taper+bandpass"
              f"({corr_data['freqmin']}~{corr_data['freqmax']} Hz)")

    if do_whiten_acf:  # ← 新增开关参数
        corr_data = whiten_acf(corr_data,
                               freqmin=acf_freqmin,
                               freqmax=acf_freqmax,
                               method=acf_whiten_method)

    if do_abs_max:
        corr_data = abs_max(corr_data)
        print(f"  abs_max : 归一化到 [-1,1]")



    return corr_data


# ═══════════════════════════════════════════════════════════════════
# 5. 便捷函数（保持向后兼容）
# ═══════════════════════════════════════════════════════════════════

def process_acf(fft_data: dict, maxlag: float,
                freqmin: float, freqmax: float,
                do_clean_up: bool  = True,
                do_abs_max:  bool  = True,
                max_length:  float = 20.0) -> dict:
    """
    向后兼容接口，对应原 correlate.py 的 process_acf()。
    内部调用 correlate()，频域模式，结果与原版完全一致。

    Parameters
    ----------
    fft_data : process_fft 返回的字典（含 'fft' 字段）
    """
    return correlate(
        data1           = fft_data,
        maxlag          = maxlag,
        data2           = None,
        use_freq_domain = True,
        do_clean_up     = do_clean_up,
        do_abs_max      = do_abs_max,
        max_length      = max_length,
    )
