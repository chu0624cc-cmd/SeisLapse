import numpy as np
from scipy.signal import detrend as scipy_detrend
from scipy.ndimage import uniform_filter1d
import warnings

# ─────────────────────────────────────────────
# 1. 辅助函数
# ─────────────────────────────────────────────

def detrend(x: np.ndarray) -> np.ndarray:
    """去均值 + 去线性趋势"""
    return scipy_detrend(x, type='linear', axis=0)

def demean(x: np.ndarray) -> np.ndarray:
    """去均值"""
    return x - np.mean(x, axis=0)


# def taper(x: np.ndarray, fs: float,
#           max_percentage: float = 0.05,
#           max_length: float = 20.0) -> np.ndarray:
#     """
#     对时间序列两端施加 Hanning 锥形窗。
#     Parameters
#     ----------
#     x              : 时间序列，shape (N,) 或 (N, M)
#     fs             : 采样率 (Hz)
#     max_percentage : 单端锥形窗占总长度的最大比例（默认 0.05）
#     max_length     : 单端锥形窗最大长度（秒，默认 20.0）
#     """
#     npts = x.shape[0]
#
#     wlen = int(min(
#         np.floor(npts * max_percentage),
#         np.floor(max_length * fs),
#         np.floor(npts / 2)
#     ))
#
#     if wlen == 0:
#         return x.copy()
#
#     # Hanning 窗：sin(π * i / wlen)^2，与 Julia hanningwindow 一致
#     i = np.linspace(0, wlen, wlen, endpoint=False)
#     taper_win = np.sin(np.pi * i / wlen) ** 2   # shape (wlen,)
#
#     out = x.copy()
#     if x.ndim == 1:
#         out[:wlen]  *= taper_win
#         out[-wlen:] *= taper_win[::-1]
#     else:
#         out[:wlen,  :] *= taper_win[:, np.newaxis]
#         out[-wlen:, :] *= taper_win[::-1, np.newaxis]
#     return out

def taper(x: np.ndarray, fs: float,
          max_percentage: float = 0.05,
          max_length: float = 20.0) -> np.ndarray:
    npts = x.shape[0]
    wlen = int(min(
        np.floor(npts * max_percentage),
        np.floor(max_length * fs),
        np.floor(npts / 2)
    ))
    if wlen == 0:
        return x.copy()

    i = np.linspace(0, wlen, wlen, endpoint=False)
    taper_win = np.sin(np.pi * i / (2 * wlen)) ** 2   # ← 修正：2*wlen

    out = x.copy()
    if x.ndim == 1:
        out[:wlen]  *= taper_win
        out[-wlen:] *= taper_win[::-1]
    else:
        out[:wlen,  :] *= taper_win[:, np.newaxis]
        out[-wlen:, :] *= taper_win[::-1, np.newaxis]
    return out

def cosine_taper(N: int, p: float = 0.05) -> np.ndarray:
    """
    余弦锥形窗，对应 obspy cosine_taper(N, p)。
    p : 两端各做 cosine 过渡的比例，总过渡占 p（每端 p/2）
    """
    tap    = np.ones(N)
    n_tap  = int(N * p / 2)
    t      = np.arange(n_tap)
    window = 0.5 * (1.0 - np.cos(np.pi * t / n_tap))
    tap[:n_tap]  = window
    tap[-n_tap:] = window[::-1]
    return tap



def lowpass_filter(x: np.ndarray, fh: float, fs: float,
                   corners: int = 4, zerophase: bool = True) -> np.ndarray:
    """
    低通滤波器（Butterworth）。

    Parameters
    ----------
    x         : 时间序列，shape (N,) 或 (N, M)
    fh        : 截止频率 (Hz)
    fs        : 采样率 (Hz)
    corners   : 滤波器阶数
    zerophase : 是否零相位（前向+后向滤波）
    """
    from scipy.signal import butter, sosfiltfilt, sosfilt
    nyq = fs / 2.0
    if fh >= nyq:
        warnings.warn(f"截止频率 {fh:.4f} Hz >= 奈奎斯特 {nyq:.4f} Hz，跳过滤波")
        return x.copy()
    sos = butter(corners, fh / nyq, btype='low', output='sos')
    func = sosfiltfilt if zerophase else sosfilt
    return func(sos, x, axis=0)


def highpass_filter(x: np.ndarray, fl: float, fs: float,
                    corners: int = 4, zerophase: bool = True) -> np.ndarray:
    """
    高通滤波器（Butterworth）。

    Parameters
    ----------
    x         : 时间序列，shape (N,) 或 (N, M)
    fl        : 截止频率 (Hz)
    fs        : 采样率 (Hz)
    corners   : 滤波器阶数
    zerophase : 是否零相位（前向+后向滤波）
    """
    from scipy.signal import butter, sosfiltfilt, sosfilt
    nyq = fs / 2.0
    if fl >= nyq:
        raise ValueError(f"截止频率 {fl} Hz 超过奈奎斯特 {nyq} Hz")
    sos = butter(corners, fl / nyq, btype='high', output='sos')
    func = sosfiltfilt if zerophase else sosfilt
    return func(sos, x, axis=0)


def bandpass_filter(x: np.ndarray, freqmin: float, freqmax: float, fs: float,
                    corners: int = 4, zerophase: bool = True) -> np.ndarray:
    """
    带通滤波器（Butterworth）。

    Parameters
    ----------
    x        : 时间序列，shape (N,) 或 (N, M)
    freqmin  : 低截止频率 (Hz)
    freqmax  : 高截止频率 (Hz)
    fs       : 采样率 (Hz)
    corners  : 滤波器阶数
    zerophase: 是否零相位
    """
    from scipy.signal import butter, sosfiltfilt, sosfilt
    nyq = fs / 2.0
    if freqmin >= nyq:
        raise ValueError(f"低截止频率 {freqmin} Hz 超过奈奎斯特 {nyq} Hz")
    if freqmax >= nyq - 1e-6:
        warnings.warn(
            f"高截止频率 {freqmax:.4f} Hz 接近或超过奈奎斯特 {nyq:.4f} Hz，"
            f"自动降级为高通滤波"
        )
        return highpass_filter(x, freqmin, fs, corners=corners, zerophase=zerophase)
    sos = butter(corners, [freqmin / nyq, freqmax / nyq], btype='band', output='sos')
    func = sosfiltfilt if zerophase else sosfilt
    return func(sos, x, axis=0)



def smooth_acf(acf, fs, fmin, fmax, method='uniform', cycles=1.0):
    """
    对ACF做时域平滑，窗口长度由目标频带自动计算。

    平滑窗口 = cycles / f_center * fs  (点数)
    其中 f_center = (fmin + fmax) / 2

    物理意义：窗口长度 = 目标频带中心频率的 cycles 个周期
    → 压制频带内振荡，保留趋势
    → cycles=1.0 时对应作者的处理方式（最优smooth_pts=31≈1/1.5*50）

    Parameters
    ----------
    acf    : ACF序列，shape (N,) 或 (N, M)
    fs     : 采样率 (Hz)
    fmin   : 目标频带下限 (Hz)
    fmax   : 目标频带上限 (Hz)
    method : 'uniform'  → 均值滤波（与作者一致）
             'gaussian' → 高斯平滑
             'hanning'  → Hanning窗卷积
    cycles : 平滑窗口覆盖的周期数（默认1.0）

    Returns
    -------
    平滑后的ACF，shape与输入相同
    """
    f_center   = (fmin + fmax) / 2.0
    T_center   = 1.0 / f_center             # 中心频率周期（秒）
    win_s      = cycles * T_center          # 平滑窗口长度（秒）
    smooth_pts = max(3, int(win_s * fs))
    if smooth_pts % 2 == 0:
        smooth_pts += 1                     # 保持奇数，保证对称

    print(f"smooth_acf: fmin={fmin}  fmax={fmax}  "
          f"f_center={f_center:.3f}Hz  "
          f"T_center={T_center:.3f}s  "
          f"cycles={cycles}  "
          f"win={win_s:.3f}s  "
          f"pts={smooth_pts}")

    def _smooth_1d(x):
        if method == 'uniform':
            return uniform_filter1d(x, size=smooth_pts)
        elif method == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            sigma = smooth_pts / 6.0
            return gaussian_filter1d(x, sigma=sigma)
        elif method == 'hanning':
            win = np.hanning(smooth_pts)
            win /= win.sum()
            return np.convolve(x, win, mode='same')
        else:
            raise ValueError(f"未知method: {method}")

    if acf.ndim == 1:
        return _smooth_1d(acf)

    out = np.zeros_like(acf)
    for i in range(acf.shape[1]):
        out[:, i] = _smooth_1d(acf[:, i])
    return out


def sliding_mean(stacks, n_sm):
    """
    时间序列方向的滑动平均（沿天轴/窗口轴）。

    Parameters
    ----------
    stacks : shape (n_lags, n_windows)
             每列是一个时间单元的ACF
             时间单元可以是：1天、1小时、30分钟等
             由调用者决定，函数本身只认列数
    n_sm   : 平滑窗口大小（单位=列数=时间单元数）
             n_sm=1  → 不平滑
             n_sm=3  → 以当前列为中心，取前后共3列求均值
             n_sm=7  → 取前后共7列求均值

    Returns
    -------
    out : shape (n_lags, n_windows)，与输入相同
    """
    if n_sm <= 1:
        return stacks.copy()

    n_lags, n_windows = stacks.shape
    out = np.zeros_like(stacks)
    half = n_sm // 2

    for i in range(n_windows):
        lo = max(0, i - half)
        hi = min(n_windows, i + half + 1)
        out[:, i] = np.mean(stacks[:, lo:hi], axis=1)

    return out


def recommend_smooth_hw_shallow(fs, cc_len,
                                depth_min_km=0.05,
                                depth_max_km=1.0,
                                vp_kms=0.5,
                                safety=0.2):
    from scipy.fft import next_fast_len

    Nfft    = next_fast_len(int(cc_len * fs))
    delta_f = fs / Nfft

    tau_shallow    = 2 * depth_min_km / vp_kms
    tau_deep       = 2 * depth_max_km / vp_kms
    fringe_shallow = 1.0 / tau_shallow
    fringe_deep    = 1.0 / tau_deep

    hw_strict = fringe_shallow / (2 * delta_f)   # 保护最浅层
    hw_loose  = fringe_deep    / (2 * delta_f)   # 保护最深层

    # ── Bug fix：推荐值应基于 hw_loose（最深目标约束），
    #            而不是 hw_strict（最浅目标约束）
    #            hw_strict 只是告诉你"绝对不能超过多少"
    hw_rec = max(1, int(hw_loose * safety))

    print("=" * 55)
    print(f"  fs={fs} Hz   cc_len={cc_len} s   Nfft={Nfft}")
    print(f"  delta_f = {delta_f*1000:.4f} mHz/pt")
    print("-" * 55)
    print(f"  Shallow target : {depth_min_km} km  "
          f"tau={tau_shallow:.2f}s  fringe={fringe_shallow:.3f} Hz")
    print(f"  Deep    target : {depth_max_km} km  "
          f"tau={tau_deep:.2f}s  fringe={fringe_deep:.3f} Hz")
    print("-" * 55)
    print(f"  HW absolute upper limit : {hw_strict:.0f} pts  "
          f"← never exceed (kills {depth_min_km} km signal)")
    print(f"  HW safe    upper limit  : {hw_loose:.0f} pts  "
          f"← based on deepest target ({depth_max_km} km)")
    print(f"  HW recommended          : {hw_rec} pts  "
          f"(= hw_loose × safety={safety})")
    print("-" * 55)

    # 打印每个候选值保护的最浅深度
    print("  Candidate HW values and protected depth:")
    for hw in [5, 10, 30, 60, 90, hw_rec, hw_loose, hw_strict]:
        hw = int(hw)
        df_smooth    = 2 * hw * delta_f
        if df_smooth > 0:
            tau_protect  = 1.0 / df_smooth
            depth_protect = tau_protect * vp_kms / 2
            safe = "OK" if hw <= hw_loose else \
                   "RISKY" if hw <= hw_strict else "DANGER"
            print(f"    hw={hw:>5d} pts  "
                  f"smooth={df_smooth*1000:.2f} mHz  "
                  f"protects > {depth_protect*1000:.0f} m  [{safe}]")
    print("=" * 55)
    return hw_rec, int(hw_strict), int(hw_loose)