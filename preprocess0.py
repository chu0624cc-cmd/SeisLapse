"""
preprocess.py
=============
  Step1：质量控制 + 带通滤波
  Step2：时域归一化 + 谱白化 → 返回时域白化信号
数学等价性：
  irfft(conj(F1)*F2) = xcorr(irfft(F1), irfft(F2))
  所以先白化再 irfft 回时域，再做相关，结果与原流程完全一致。
"""

import numpy as np
from scipy.fft import rfft, irfft, rfftfreq, next_fast_len
from tool import bandpass_filter, detrend, taper

# ─────────────────────────────────────────────
# 1. 振幅质量控制，对应 Julia remove_amp! + std_threshold
# ─────────────────────────────────────────────

def remove_amp(raw: dict, max_std: float = 10.0) -> dict:
    """
    剔除异常窗口。对应 Julia remove_amp!(R, max_std=10.)
    步骤：① 剔除全零窗口  ② 剔除 max(|col|)/std(col) > max_std 的窗口
    """
    x, t = raw['x'].copy(), raw['t'].copy()

    # 对应 Julia nonzero()
    nonzero_idx = np.where(np.any(x != 0, axis=0))[0]
    if len(nonzero_idx) == 0:
        raise ValueError("所有窗口均为零")
    x, t = x[:, nonzero_idx], t[nonzero_idx]

    # 对应 Julia std_threshold(A, max_std)
    stds = np.std(x, axis=0)
    maxs = np.max(np.abs(x), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(stds > 0, maxs / stds, np.inf)
    good_idx  = np.where(ratio < max_std)[0]

    if len(good_idx) == 0:
        raise ValueError("所有窗口振幅均超过阈值，请放宽 max_std")
    n_removed = x.shape[1] - len(good_idx)
    if n_removed > 0:
        print(f"  remove_amp：剔除 {n_removed} 个异常窗口，剩余 {len(good_idx)} 个")

    return {**raw, 'x': x[:, good_idx], 't': t[good_idx]}


# ─────────────────────────────────────────────
# 2. 时域归一化（可选，Julia 标准流程无此步）
# ─────────────────────────────────────────────

def onebit(raw: dict) -> dict:
    """单比特归一化，对应 Julia onebit!(R)"""
    return {**raw, 'x': np.sign(raw['x'])}


def clip(raw: dict, factor: float = 3.0) -> dict:
    """
    截幅归一化，对应 Julia clip!(R, factor)
    Julia clip! 用 std 作为基准（f=std，默认）
    """
    x    = raw['x'].copy()
    high = np.std(x, axis=0) * factor
    for i in range(x.shape[1]):
        np.clip(x[:, i], -high[i], high[i], out=x[:, i])
    return {**raw, 'x': x}


def running_abs_mean(raw: dict, half_win: int = 50,
                     weight_src: np.ndarray = None) -> dict:
    """
    运行绝对均值归一化（Bensen et al., 2007）。

    标准模式（weight_src=None）：
      w_n = mean(|x[n-N : n+N+1]|)
      x_norm[n] = x[n] / w_n

    地震频带压制模式（weight_src 不为 None，对应 Bensen Figure 5）：
      w_n = mean(|x_filtered[n-N : n+N+1]|)   ← 权重来自滤波后数据
      x_norm[n] = x[n] / w_n                  ← 归一化施加在原始数据

    Parameters
    ----------
    raw        : 数据字典，'x' shape (N, M)
    half_win   : 滑动窗口半长（采样点数）
                 推荐 int(fs / freqmin / 2)
    weight_src : 用于计算权重的外部数据（如滤波到地震频带的副本）
                 shape 须与 raw['x'] 相同
                 为 None 时退化为标准 RAM
    """
    x      = raw['x'].copy()
    out    = np.zeros_like(x)
    w_data = weight_src if weight_src is not None else x

    for i in range(x.shape[1]):
        col   = x[:, i]
        col_w = w_data[:, i]
        N     = len(col)

        # cumsum 差分实现滑动绝对均值，O(N) 复杂度
        csum = np.cumsum(np.concatenate([[0.0], np.abs(col_w)]))
        weight = np.zeros(N)
        for j in range(N):
            lo         = max(0, j - half_win)
            hi         = min(N, j + half_win + 1)
            weight[j]  = (csum[hi] - csum[lo]) / (hi - lo)

        weight[weight < 1e-30] = 1e-30
        out[:, i] = col / weight

    return {**raw, 'x': out}



def running_abs_mean_multiband(raw: dict, bands: list, ram_half_win: int = None) -> dict:
    """
    多频带运行绝对均值归一化（Multi-band RAM）。

    对每个子频带分别计算滑动绝对均值权重，取所有频带权重的平均值，
    再将平均权重施加到 bandpass 后的数据上。

    优势：
      - 比单频带 RAM 更均匀地压制不同频段的地震信号
      - 避免单一频带权重过弱导致的除零问题
      - 对宽目标频带（如 0.1~10 Hz）效果更稳定

    Parameters
    ----------
    raw          : 数据字典，'x' 已经过 bandpass 到目标频带
    bands        : 子频带列表，每个元素为 (fmin, fmax)，单位 Hz
                   所有子频带必须是目标频带的子集
                   例如 [(0.5, 1.0), (1.0, 2.0)] 覆盖目标频带 0.5~2 Hz
    ram_half_win : 滑动半窗（采样点），None 时对每个频带自动计算
                   = int(fs / fmin / 2)，即半个最低频率周期

    Returns
    -------
    归一化后的数据字典
    """
    x   = raw['x'].copy()
    fs  = raw['fs']
    out = np.zeros_like(x)
    N   = x.shape[0]

    # 对每个子频带计算权重，累加
    weight_sum = np.zeros_like(x)

    for fmin_b, fmax_b in bands:
        # Nyquist 保护
        if fmax_b >= fs / 2:
            print(f"  警告：子频带 {fmin_b}~{fmax_b} Hz 超过 Nyquist，跳过")
            continue

        # 自动计算该频带的最佳窗长
        hw = ram_half_win if ram_half_win is not None else int(fs / fmin_b / 2)

        # 生成该频带的滤波副本
        filtered = bandpass_filter(x, fmin_b, fmax_b, fs)

        # 对每列计算滑动绝对均值
        for i in range(x.shape[1]):
            col_w = filtered[:, i]
            csum  = np.cumsum(np.concatenate([[0.0], np.abs(col_w)]))
            for j in range(N):
                lo = max(0, j - hw)
                hi = min(N, j + hw + 1)
                weight_sum[j, i] += (csum[hi] - csum[lo]) / (hi - lo)

    # 取所有频带权重的平均
    n_bands      = len(bands)
    weight_mean  = weight_sum / n_bands
    weight_mean[weight_mean < 1e-30] = 1e-30

    out = x / weight_mean

    return {**raw, 'x': out}


def time_norm(raw: dict, method: str = 'onebit', clip_factor: float = 3.0,
              ram_half_win: int = 50, eq_band: tuple = None,
              multi_bands: list = None) -> dict:
    """
    时域归一化统一入口。

    method:
      'onebit'     → 单比特归一化
      'clip'       → 截幅归一化
      'ram'        → 标准 RAM，权重来自自身
      'ram_eq'     → 单频带地震压制 RAM（Bensen Figure 5）
                     half_win 根据 eq_band 自动计算
      'ram_multi'  → 多频带 RAM，对多个子频带分别算权重取平均
                     需指定 multi_bands

    Parameters
    ----------
    ram_half_win : 'ram' 模式的滑动半窗
    eq_band      : 'ram_eq' 的地震频带 (fmin, fmax)，Hz
    multi_bands  : 'ram_multi' 的子频带列表
                   例如 [(0.5, 1.0), (1.0, 2.0)]
                   建议按目标频带均匀划分，覆盖完整

    Examples
    --------
    # 目标频带 0.5~2 Hz，均匀划分为两个子带
    raw = time_norm(raw, method='ram_multi',
                    multi_bands=[(0.5, 1.0), (1.0, 2.0)])

    # 目标频带 0.5~2 Hz，三个子带
    raw = time_norm(raw, method='ram_multi',
                    multi_bands=[(0.5, 0.8), (0.8, 1.3), (1.3, 2.0)])
    """
    if method == 'onebit':
        return onebit(raw)

    elif method == 'clip':
        return clip(raw, factor=clip_factor)

    elif method == 'ram':
        return running_abs_mean(raw, half_win=ram_half_win)

    elif method == 'ram_eq':
        if eq_band is None:
            return running_abs_mean(raw, half_win=ram_half_win)

        fmin_eq, fmax_eq = eq_band
        fs = raw['fs']

        if fmax_eq >= fs / 2:
            raise ValueError(
                f"eq_band 上限 {fmax_eq:.4f} Hz >= Nyquist {fs/2} Hz"
            )

        ram_half_win_eq = int(fs / fmin_eq / 2)
        filtered_x      = bandpass_filter(raw['x'], fmin_eq, fmax_eq, fs)

        print(f"  ram_eq：地震频带 {fmin_eq}~{fmax_eq} Hz  "
              f"half_win={ram_half_win_eq} pts={ram_half_win_eq/fs:.1f}s")

        return running_abs_mean(raw, half_win=ram_half_win_eq,
                                weight_src=filtered_x)

    elif method == 'ram_multi':
        if multi_bands is None:
            raise ValueError("method='ram_multi' 需指定 multi_bands 参数")

        fs = raw['fs']
        print(f"  ram_multi：{len(multi_bands)} 个子频带")
        for fb in multi_bands:
            hw = int(fs / fb[0] / 2)
            print(f"    {fb[0]}~{fb[1]} Hz  half_win={hw} pts={hw/fs:.2f}s")

        return running_abs_mean_multiband(raw, bands=multi_bands,
                                          ram_half_win=None)  # 每频带自动计算

    else:
        raise ValueError(
            f"未知方法: '{method}'，"
            f"可选 'onebit'/'clip'/'ram'/'ram_eq'/'ram_multi'"
        )


# ─────────────────────────────────────────────
# 3. 滑动平均，对应 Julia smooth!(A, half_win)
# ─────────────────────────────────────────────

def smooth(x: np.ndarray, half_win: int = 3) -> np.ndarray:
    """
    cumsum 差分实现滑动平均，边缘用变长窗口。
    对应 Julia smooth!(A, half_win)，结果与 Julia 一致。
    """
    n    = len(x)
    csum = np.cumsum(np.concatenate([[0.0], x]))
    out  = np.zeros(n)
    for i in range(n):
        lo      = max(0, i - half_win)
        hi      = min(n, i + half_win + 1)
        out[i]  = (csum[hi] - csum[lo]) / (hi - lo)
    return out

# ─────────────────────────────────────────────
# 4. rfft，对应 Julia FFT = rfft.(R)
# ─────────────────────────────────────────────

def compute_fft(raw: dict) -> dict:
    """
    计算单边 rfft，返回 FFTData 字典。
    对应 Julia: FFT = rfft.(R)，rfft(R.x, dims=1)
    """
    x    = raw['x']
    N    = x.shape[0]
    Nfft = next_fast_len(N)
    freq = rfftfreq(Nfft, d=1.0 / raw['fs'])
    F    = rfft(x, Nfft, axis=0)    # shape (Nfft//2+1, n_windows)

    return {
        'fft'          : F,
        'freq'         : freq,
        'fs'           : raw['fs'],
        'n_pts'        : N,
        'n_fft'        : Nfft,
        't'            : raw['t'],
        'cc_len'       : raw['cc_len'],
        'cc_step'      : raw['cc_step'],
        'freqmin'      : raw['freqmin'],
        'freqmax'      : raw['freqmax'],
        'net'          : raw['net'],
        'sta'          : raw['sta'],
        'loc'          : raw['loc'],
        'cha'          : raw['cha'],
        'whitened'     : False,
        'whiten_method': 'none',
        'time_norm'    : raw.get('time_norm', 'none'),
    }


# ─────────────────────────────────────────────
# 5. 谱白化，对应 Julia whiten!(FFT, freqmin, freqmax, fs, N; pad=50)
# ─────────────────────────────────────────────

def whiten_fft(fft_data: dict, freqmin: float, freqmax: float,
               method: str = 'standard', smooth_half_win: int = 5,
               pad: int = 50) -> dict:
    """
    在 FFT 域直接谱白化，对应 Julia whiten!(FFT, freqmin, freqmax)

    Julia 实现细节（严格对应）：
      - 频带外置零
      - 左右各 pad 个点做余弦过渡
      - 通带内：F = exp(im * angle(F))，即只保留相位，振幅归一
      - method='smoothed'：除以滑动平均振幅谱（非 Julia 标准，扩展选项）

    Parameters
    ----------
    fft_data        : compute_fft 返回的字典
    freqmin/max     : 白化频带 (Hz)
    method          : 'standard'（Julia默认）或 'smoothed'
    smooth_half_win : smoothed 方法的滑动平均半窗（频率点数）
    pad             : 过渡带点数，对应 Julia pad=50
    """
    F    = fft_data['fft'].copy()         # shape (Nfft//2+1, n_windows)
    freq = fft_data['freq']
    fs   = fft_data['fs']
    N    = fft_data['n_pts']

    # 对应 Julia: freqvec = rfftfreq(N, fs)
    freqvec = rfftfreq(N, d=1.0 / fs)

    # 对应 Julia: left = findfirst(x->x>=freqmin), right = findfirst(freqmax<=freqvec)
    left_arr  = np.where(freqvec >= freqmin)[0]
    right_arr = np.where(freqvec >= freqmax)[0]
    if len(left_arr) == 0 or len(right_arr) == 0:
        raise ValueError(f"freqmin={freqmin} 或 freqmax={freqmax} 超出频率范围")
    left, right = left_arr[0], right_arr[0]

    # 对应 Julia: low = left-pad, high = right+pad（边界保护）
    low  = max(0,              left  - pad)
    high = min(F.shape[0] - 1, right + pad)
    # 如果 low/high 被截断，相应调整 left/right
    left  = low  + pad
    right = high - pad

    padarr = np.arange(pad, dtype=float)   # 0,1,...,pad-1

    for i in range(F.shape[1]):
        col  = F[:, i]
        Fabs = np.abs(col)

        if method == 'smoothed':
            Fabs = smooth(Fabs, half_win=smooth_half_win)
        Fabs[Fabs < 1e-30] = 1e-30

        # 对应 Julia: A[1:low-1] = 0  （频带外置零，1-indexed → 0-indexed）
        col[:low] = 0.0

        # 对应 Julia 左过渡：cos(π/2 + π/2 * padarr/pad)^2 * exp(im*angle)
        taper_left = np.cos(np.pi / 2.0 + np.pi / 2.0 * padarr / pad) ** 2
        col[low:left] = taper_left * np.exp(1j * np.angle(col[low:left]))

        # 对应 Julia 通带：exp(im * angle(A))，振幅归一
        if method == 'standard':
            col[left:right] = np.exp(1j * np.angle(col[left:right]))
        else:  # smoothed：除以平滑振幅
            col[left:right] = col[left:right] / Fabs[left:right]

        # 对应 Julia 右过渡：cos(π/2 * padarr/pad)^2 * exp(im*angle)
        taper_right = np.cos(np.pi / 2.0 * padarr / pad) ** 2
        col[right:high] = taper_right * np.exp(1j * np.angle(col[right:high]))

        # 对应 Julia: A[high:end] = 0
        col[high:] = 0.0
        F[:, i] = col

    return {**fft_data, 'fft': F, 'freqmin': freqmin, 'freqmax': freqmax,
            'whitened': True, 'whiten_method': method}




# ═══════════════════════════════════════════════════════════════════
# STEP 1：质量控制 + 带通滤波
# ═══════════════════════════════════════════════════════════════════
def step1_preprocess(raw: dict,
                     freqmin: float = 0.5,
                     freqmax: float = 2.0,
                     max_std: float = 10.0) -> dict:
    """
    Step1：质量控制 + 带通滤波。
    复用 remove_amp() + bandpass_filter()。

    Parameters
    ----------
    raw     : make_raw_data 返回的字典
    freqmin : 带通下限 (Hz)
    freqmax : 带通上限 (Hz)
    max_std : 振幅异常阈值

    Returns
    -------
    raw 字典，'x' 已滤波，新增 'freqmin'/'freqmax'
    """
    # ── 质量控制，复用 remove_amp() ───────────────────────────────────
    raw = remove_amp(raw, max_std=max_std)

    # ── 带通滤波 ──────────────────────────────────────────────────────
    x_bp = bandpass_filter(raw['x'], freqmin, freqmax, raw['fs'])
    raw  = {**raw, 'x': x_bp, 'freqmin': freqmin, 'freqmax': freqmax}

    print(f"  [Step1] bandpass {freqmin}~{freqmax} Hz  "
          f"n_windows={raw['x'].shape[1]}")
    return raw



def step2_time_norm(raw: dict,
                    time_norm_method: str   = 'ram',
                    ram_half_win:     int   = 50,
                    eq_band:          tuple = None,
                    multi_bands:      list  = None,
                    clip_factor:      float = 3.0) -> dict:
    """
    Step2：时域归一化。

    对应 Julia 可选步骤：onebit!(R) / clip!(R) / RAM 等。
    time_norm_method=None 时跳过，直接透传。

    Parameters
    ----------
    raw              : step1_preprocess 返回的字典
    time_norm_method : None/'onebit'/'clip'/'ram'/'ram_eq'/'ram_multi'
    ram_half_win     : RAM 半窗（采样点），推荐 int(fs/freqmin/2)
    eq_band          : ram_eq 地震频带 (fmin, fmax) Hz
    multi_bands      : ram_multi 子频带列表
    clip_factor      : clip 截幅倍数

    Returns
    -------
    raw 字典，'x' 更新为归一化后信号，新增 'time_norm' 字段
    """
    if time_norm_method is not None:
        raw = time_norm(raw,
                        method       = time_norm_method,
                        clip_factor  = clip_factor,
                        ram_half_win = ram_half_win,
                        eq_band      = eq_band,
                        multi_bands  = multi_bands)
        raw['time_norm'] = time_norm_method
    else:
        raw['time_norm'] = 'none'

    print(f"  [Step2] time_norm={raw['time_norm']}")
    return raw


def step3_whiten(raw: dict,
                 whiten_method:   str = 'standard',
                 smooth_half_win: int = 5,
                 pad:             int = 50,
                 freqmin: float = None,  # ← 新增
                 freqmax: float = None  # ← 新增
                 ) -> dict:

    """
    Step3：谱白化 → 同时输出频域和时域白化信号。

    流程：
      x(时域) → rfft → whiten(频域) → F_white  ← 频域输出
                                     ↓ irfft
                                   x_white      ← 时域输出

    对应 Julia whiten!(FFT, freqmin, freqmax; pad=50)

    Parameters
    ----------
    raw              : step2_time_norm 返回的字典
    whiten_method    : 'standard'（Julia默认）/'smoothed'
    smooth_half_win  : smoothed 滑动平均半窗（频率点数）
    pad              : 谱白化过渡带点数，对应 Julia pad=50

    Returns
    -------
    字典，新增字段：
      'x'            : 更新为白化后时域信号（链式调用保持一致）
      'x_white'      : 白化后时域信号，shape (N, n_windows)
      'F_white'      : 白化后频域信号，shape (Nfft//2+1, n_windows)，复数
      'freq'         : 频率轴，shape (Nfft//2+1,)，Hz
      'n_fft'        : FFT 点数 Nfft
      'n_pts'        : 原始信号点数 N
      'whitened'     : True
      'whiten_method': 谱白化方法名
    """

    # 未传入时从 raw 字典取，保证单独调用也能正常工作
    _freqmin = freqmin if freqmin is not None else raw['freqmin']
    _freqmax = freqmax if freqmax is not None else raw['freqmax']

    # ── rfft → 谱白化 ─────────────────────────────────────────────────
    fft_data = compute_fft(raw)
    fft_data = whiten_fft(fft_data,
                          freqmin         = _freqmin,
                          freqmax         = _freqmax,
                          method          = whiten_method,
                          smooth_half_win = smooth_half_win,
                          pad             = pad)

    F_white = fft_data['fft']
    N       = fft_data['n_pts']
    Nfft    = fft_data['n_fft']

    # ── irfft → 时域白化信号 ──────────────────────────────────────────
    x_white = irfft(F_white, Nfft, axis=0)[:N, :]

    print(f"  [Step3] whiten={whiten_method}")
    print(f"  [Step3] F_white shape={F_white.shape}  (频域)")
    print(f"  [Step3] x_white shape={x_white.shape}  (时域)")

    return {**raw,
            'x'            : x_white,
            'x_white'      : x_white,
            'F_white'      : F_white,
            'freq'         : fft_data['freq'],
            'n_fft'        : Nfft,
            'n_pts'        : N,
            'whitened'     : True,
            'whiten_method': whiten_method}


def preprocess(raw: dict,
               freqmin:          float = 0.5,
               freqmax:          float = 2.0,
               max_std:          float = 10.0,
               time_norm_method: str   = 'ram',
               ram_half_win:     int   = 50,
               eq_band:          tuple = None,
               multi_bands:      list  = None,
               clip_factor:      float = 3.0,
               whiten_method:    str   = 'standard',
               smooth_half_win:  int   = 5,
               pad:              int   = 50) -> dict:
    """
    完整预处理入口：Step1 + Step2 + Step3。

      Step1：质量控制 + 带通滤波
      Step2：时域归一化
      Step3：谱白化 → x_white（时域）+ F_white（频域）

    Parameters
    ----------
    raw    : make_raw_data 返回的原始字典
    其余参数见各步骤函数

    Returns
    -------
    step3_whiten 的返回字典，含 'x_white' 和 'F_white'
    """
    print("── Step1：质量控制 + 带通滤波 ──────────────────────────")
    d = step1_preprocess(raw,
                         freqmin = freqmin,
                         freqmax = freqmax,
                         max_std = max_std)

    print("── Step2：时域归一化 ────────────────────────────────────")
    d = step2_time_norm(d,
                        time_norm_method = time_norm_method,
                        ram_half_win     = ram_half_win,
                        eq_band          = eq_band,
                        multi_bands      = multi_bands,
                        clip_factor      = clip_factor)

    print("── Step3：谱白化 ────────────────────────────────────────")
    d = step3_whiten(d,
                     whiten_method   = whiten_method,
                     smooth_half_win = smooth_half_win,
                     pad             = pad,
                     freqmin         = freqmin,
                     freqmax         = freqmax)
    return d