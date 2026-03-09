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


def vmd_parameter_sweep(trace_ref: np.ndarray,
                         time: np.ndarray,
                         fs: float,
                         freq_band: tuple = (1.0, 2.0),
                         mute_end: float = 3.0,
                         taper_len: float = 2.0,
                         K_list: list = None,
                         alpha_list: list = None,
                         plot: bool = True,
                         verbose: bool = True) -> dict:
    """
    VMD-HHT 参数敏感性测试（Parameter Sweeping）。

    对输入的参考波形，遍历 K × alpha 网格组合，评估每组参数在
    目标频带内的模态分离质量，返回最优参数并可视化结果。

    Parameters
    ----------
    trace_ref : np.ndarray
        参考波形（1D），正滞后段
    time : np.ndarray
        时间轴（s），与 trace_ref 等长
    fs : float
        采样率（Hz）
    freq_band : tuple
        目标频带 (f_min, f_max)（Hz），默认 (1.0, 2.0)
    mute_end : float
        ACF 主峰静音结束时刻（s），默认 3.0
    taper_len : float
        静音过渡段长度（s），默认 2.0
    K_list : list of int
        待测模态数列表，默认 [3, 4, 5, 6]
    alpha_list : list of int/float
        待测带宽惩罚因子列表，默认 [200, 500, 1000, 2000]
    plot : bool
        是否绘制热力图和模态分布图，默认 True
    verbose : bool
        是否打印逐组结果，默认 True

    Returns
    -------
    dict，包含：
        best_K       : int     最优模态数
        best_alpha   : float   最优惩罚因子
        best_score   : float   最优综合评分
        best_fc      : float   最优模态中心频率（Hz）
        best_n_inband: int     最优参数下频带内模态数
        best_energy_ratio: float 最优参数下频带内能量占比
        results      : dict    所有 (K, alpha) 组合的详细结果
        score_matrix : ndarray 综合评分矩阵，shape (len(K_list), len(alpha_list))

    Example
    -------
    >>> out = vmd_parameter_sweep(
    ...     trace_ref = trace_ref_pos,
    ...     time      = time_pos,
    ...     fs        = 50.0,
    ...     freq_band = (1.0, 2.0),
    ...     mute_end  = 3.0,
    ... )
    >>> print(f"最优参数: K={out['best_K']}, alpha={out['best_alpha']}")
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    from vmd_hht_dvv import run_vmd, mute_zero_lag, _estimate_fc

    # ── 默认参数网格 ──────────────────────────────────────────
    if K_list     is None: K_list     = [3, 4, 5, 6]
    if alpha_list is None: alpha_list = [200, 500, 1000, 2000]

    trace_ref = np.asarray(trace_ref, dtype=float).ravel()
    time      = np.asarray(time,      dtype=float).ravel()
    assert len(trace_ref) == len(time), "trace_ref 与 time 长度必须一致"

    nK = len(K_list)
    nA = len(alpha_list)
    f_center = (freq_band[0] + freq_band[1]) / 2.0

    # ── 预处理：静音主峰 ──────────────────────────────────────
    tr_muted = mute_zero_lag(trace_ref, time,
                              mute_end=mute_end, taper_len=taper_len)

    if verbose:
        print(f"VMD 参数敏感性测试")
        print(f"  目标频带 : {freq_band[0]}~{freq_band[1]} Hz  (中心 {f_center:.2f} Hz)")
        print(f"  K 候选   : {K_list}")
        print(f"  α 候选   : {alpha_list}")
        print(f"  fs={fs}Hz  mute_end={mute_end}s")
        print("=" * 65)

    # ── 遍历网格 ──────────────────────────────────────────────
    results = {}

    for K in K_list:
        for alpha in alpha_list:
            try:
                imfs = run_vmd(tr_muted, fs, K=K, alpha=alpha, verbose=False)
            except Exception as e:
                results[(K, alpha)] = {'error': str(e)}
                if verbose:
                    print(f"  K={K}, α={alpha:>5}: ❌ {e}")
                continue

            fc_list, energy_list = [], []
            inband_fc, inband_energy = [], []

            for k in range(K):
                try:
                    f_c, _, _ = _estimate_fc(imfs[k], fs)
                    eng = float(np.sum(imfs[k] ** 2))
                    fc_list.append(f_c)
                    energy_list.append(eng)
                    if freq_band[0] <= f_c <= freq_band[1]:
                        inband_fc.append(f_c)
                        inband_energy.append(eng)
                except:
                    fc_list.append(np.nan)
                    energy_list.append(0.0)

            total_energy = sum(energy_list) + 1e-30
            inband_n     = len(inband_fc)
            energy_ratio = sum(inband_energy) / total_energy

            if inband_n > 0:
                fc_dev  = np.mean([abs(f - f_center) / f_center
                                   for f in inband_fc])
                best_fc = inband_fc[np.argmin([abs(f - f_center)
                                               for f in inband_fc])]
            else:
                fc_dev, best_fc = 1.0, np.nan

            # 综合评分：
            #   +2.0  频带内恰好1个模态（最理想）
            #   +1.0  频带内有模态但不止1个（可用但次优）
            #   +energy_ratio  频带内能量占比（0~1）
            #   +（1-fc_dev）  f_c越接近频带中心越好（0~1）
            if inband_n == 1:
                score = 2.0
            elif inband_n > 1:
                score = 1.0
            else:
                score = 0.0
            score += energy_ratio
            score += max(0.0, 1.0 - fc_dev)

            results[(K, alpha)] = {
                'n_inband'    : inband_n,
                'best_fc'     : best_fc,
                'energy_ratio': energy_ratio,
                'fc_dev'      : fc_dev,
                'fc_list'     : fc_list,
                'energy_list' : energy_list,
                'score'       : score,
            }

            if verbose:
                status = ("✅" if inband_n == 1 else
                          ("⚠️ " if inband_n > 1 else "❌"))
                print(f"  K={K}, α={alpha:>5}: {status}  "
                      f"频带内={inband_n}个  "
                      f"能量占比={energy_ratio*100:.1f}%  "
                      f"f_c偏差={fc_dev*100:.1f}%  "
                      f"评分={score:.2f}  "
                      f"各模态={[f'{f:.2f}Hz' for f in fc_list]}")

    if verbose:
        print("=" * 65)

    # ── 构建矩阵 ──────────────────────────────────────────────
    mat_ninband  = np.zeros((nK, nA))
    mat_engratio = np.zeros((nK, nA))
    mat_fcdev    = np.zeros((nK, nA))
    mat_score    = np.zeros((nK, nA))

    for i, K in enumerate(K_list):
        for j, alpha in enumerate(alpha_list):
            r = results.get((K, alpha), {})
            if 'error' in r:
                mat_ninband[i,j]  = -1
                mat_engratio[i,j] = 0
                mat_fcdev[i,j]    = 100
                mat_score[i,j]    = 0
            else:
                mat_ninband[i,j]  = r['n_inband']
                mat_engratio[i,j] = r['energy_ratio'] * 100
                mat_fcdev[i,j]    = r['fc_dev'] * 100
                mat_score[i,j]    = r['score']

    best_i, best_j = np.unravel_index(np.argmax(mat_score), mat_score.shape)
    best_K         = K_list[best_i]
    best_alpha     = alpha_list[best_j]
    r_best         = results[(best_K, best_alpha)]

    if verbose:
        print(f"\n  推荐参数 → K={best_K},  α={best_alpha}")
        print(f"  综合评分      : {r_best['score']:.3f}")
        print(f"  频带内模态数  : {r_best['n_inband']}")
        print(f"  频带内能量占比: {r_best['energy_ratio']*100:.1f}%")
        print(f"  最优模态 f_c  : {r_best['best_fc']:.4f} Hz")

    # ── 可视化 ────────────────────────────────────────────────
    if plot:
        alpha_labels = [str(a) for a in alpha_list]
        colors_K     = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                        '#ff7f00', '#a65628'][:nK]

        fig = plt.figure(figsize=(5 * nA, 4 * 3))
        gs  = gridspec.GridSpec(3, nA, figure=fig,
                                hspace=0.55, wspace=0.4)

        def _heatmap(ax, data, title, fmt, cmap, vmin=None, vmax=None):
            im = ax.imshow(data, cmap=cmap, aspect='auto',
                           vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.85)
            ax.set_xticks(range(nA))
            ax.set_xticklabels(alpha_labels, fontsize=9)
            ax.set_yticks(range(nK))
            ax.set_yticklabels([str(k) for k in K_list], fontsize=9)
            ax.set_xlabel('alpha (α)', fontsize=9)
            ax.set_ylabel('K', fontsize=9)
            ax.set_title(title, fontsize=10)
            for ii in range(nK):
                for jj in range(nA):
                    ax.text(jj, ii, fmt.format(data[ii, jj]),
                            ha='center', va='center',
                            fontsize=9, fontweight='bold')

        # 4张热力图合并到第1行（横跨4列）
        gs_top = gridspec.GridSpec(1, 4, figure=fig,
                                   left=0.05, right=0.95,
                                   top=0.92, bottom=0.67,
                                   wspace=0.4)
        ax1 = fig.add_subplot(gs_top[0, 0])
        ax2 = fig.add_subplot(gs_top[0, 1])
        ax3 = fig.add_subplot(gs_top[0, 2])
        ax4 = fig.add_subplot(gs_top[0, 3])

        _heatmap(ax1, mat_ninband,
                 '频带内模态数\n（理想=1）',
                 '{:.0f}', 'RdYlGn', 0, 3)
        _heatmap(ax2, mat_engratio,
                 '频带内能量占比(%)\n（越高越好）',
                 '{:.1f}', 'YlOrRd', 0, 100)
        _heatmap(ax3, mat_fcdev,
                 'f_c偏差(%)\n（越小越好）',
                 '{:.1f}', 'RdYlGn_r', 0, 50)
        _heatmap(ax4, mat_score,
                 '综合评分\n（越高越好）',
                 '{:.2f}', 'YlGn', 0, 4)
        ax4.add_patch(plt.Rectangle(
            (best_j - 0.5, best_i - 0.5), 1, 1,
            fill=False, edgecolor='red', lw=3))
        ax4.set_title(f'综合评分\n最优: K={best_K}, α={best_alpha}',
                      fontsize=10, color='red')

        # 模态频率分布（第2、3行，每列对应一个alpha）
        gs_bot = gridspec.GridSpec(2, nA, figure=fig,
                                   left=0.05, right=0.95,
                                   top=0.62, bottom=0.05,
                                   hspace=0.5, wspace=0.4)

        K_groups = [K_list[:max(1, nK//2)],
                    K_list[max(1, nK//2):]]

        for j, alpha in enumerate(alpha_list):
            for row, K_subset in enumerate(K_groups):
                if not K_subset:
                    continue
                ax = fig.add_subplot(gs_bot[row, j])
                ax.axvspan(freq_band[0], freq_band[1],
                           alpha=0.15, color='green')
                ax.axvline(f_center, color='green',
                           lw=1.0, ls='--', alpha=0.6)

                for K in K_subset:
                    r = results.get((K, alpha), {})
                    if 'error' in r or 'fc_list' not in r:
                        continue
                    ci       = K_list.index(K)
                    fc_arr   = np.array(r['fc_list'])
                    eng_arr  = np.array(r['energy_list'])
                    eng_norm = eng_arr / (eng_arr.sum() + 1e-30) * 100

                    for k_idx, (fc, eng) in enumerate(
                            zip(fc_arr, eng_norm)):
                        if not np.isnan(fc):
                            in_b = freq_band[0] <= fc <= freq_band[1]
                            ax.scatter(fc, eng,
                                       c=colors_K[ci],
                                       s=80 if in_b else 40,
                                       marker='D' if in_b else 'o',
                                       zorder=3 if in_b else 2,
                                       alpha=1.0 if in_b else 0.5)
                            ax.annotate(
                                f'M{k_idx+1}\n{fc:.2f}Hz',
                                xy=(fc, eng), xytext=(0, 6),
                                textcoords='offset points',
                                fontsize=6, ha='center',
                                color=colors_K[ci])

                handles = [
                    Line2D([0],[0], marker='o', color='w',
                           markerfacecolor=colors_K[K_list.index(K)],
                           markersize=7, label=f'K={K}')
                    for K in K_subset
                ]
                ax.legend(handles=handles, fontsize=7, loc='upper right')
                ax.set_xlim([0, fs / 4])
                ax.set_ylim([0, 105])
                ax.set_xlabel('f_c (Hz)', fontsize=8)
                ax.set_ylabel('能量占比(%)', fontsize=8)
                ax.set_title(
                    f'α={alpha}  K={",".join(str(k) for k in K_subset)}',
                    fontsize=9)
                ax.grid(alpha=0.3)

        fig.suptitle(
            f"VMD 参数敏感性测试\n"
            f"目标频带 {freq_band[0]}~{freq_band[1]} Hz  "
            f"fs={fs}Hz  mute={mute_end}s  "
            f"◆=频带内  ●=频带外",
            fontsize=12)
        plt.show()

    return {
        'best_K'           : best_K,
        'best_alpha'       : best_alpha,
        'best_score'       : r_best['score'],
        'best_fc'          : r_best['best_fc'],
        'best_n_inband'    : r_best['n_inband'],
        'best_energy_ratio': r_best['energy_ratio'],
        'results'          : results,
        'score_matrix'     : mat_score,
    }