# """
# wxspectrum.py
# =============
# 小波互谱法测量地震走时变化（dv/v）。
# 严格对照 Mao et al. (2020) GJI 的 MATLAB 代码 My_Wxspectrum_TO.m 编写。
#
# Reference:
#     Mao, S., Mordret, A., Campillo, M., Fang, H., & van der Hilst, R. D. (2020).
#     On the measurement of seismic traveltime changes in the time-frequency domain
#     with wavelet cross-spectrum analysis. GJI, 221(1), 550-568.
# """
#
# import numpy as np
# from scipy.fft import fft, ifft, next_fast_len
# import pywt
#
#
# # ─────────────────────────────────────────────
# # 1. 平滑函数，对应 MATLAB smoothCFS()
# # ─────────────────────────────────────────────
#
# def _smooth_cfs(cfs: np.ndarray, scales: np.ndarray, dt: float,
#                 ns: int, nt: float) -> np.ndarray:
#     """
#     对 CWT 系数做时间方向（Gaussian）+ 尺度方向（boxcar）平滑。
#     严格对应 MATLAB smoothCFS()。
#
#     Parameters
#     ----------
#     cfs    : 待平滑系数，shape (n_scales, n_time)
#     scales : 尺度向量，shape (n_scales,)
#     dt     : 采样间隔（秒）= 1/fs
#     ns     : boxcar 窗长（尺度方向），对应 NumScalesToSmooth
#     nt     : Gaussian 窗宽度参数，对应 DegTimeToSmooth
#     """
#     n_scales, N = cfs.shape
#
#     # 对应 MATLAB: npad = 2^nextpow2(N)
#     npad = next_fast_len(N)
#
#     # 对应 MATLAB: omega = [0, 1..fix(npad/2), -fix((npad-1)/2)..-1] * 2pi/npad
#     k      = np.arange(1, npad // 2 + 1)
#     omega  = (2 * np.pi / npad) * np.concatenate([
#         [0.],
#         k,
#         -np.arange(int((npad - 1) / 2), 0, -1)
#     ])                                                  # shape (npad,)
#
#     # 时间方向：Gaussian 平滑（逐尺度）
#     # 对应 MATLAB: normscales = scales./dt
#     normscales = scales / dt                            # shape (n_scales,)
#
#     cfs_out = cfs.copy().astype(complex)
#     for k_s in range(n_scales):
#         # 对应 MATLAB: F = exp(-nt*(normscales(kk)^2)*omega.^2)
#         F      = np.exp(-nt * (normscales[k_s] ** 2) * omega ** 2)
#         smooth = ifft(F * fft(cfs_out[k_s, :], npad))
#         cfs_out[k_s, :] = smooth[:N]
#
#     # 尺度方向：boxcar 平滑（移动平均）
#     # 对应 MATLAB: H = 1/ns*ones(ns,1); cfs = conv2(cfs,H,'same')
#     if ns > 1:
#         from scipy.ndimage import uniform_filter1d
#         cfs_out = uniform_filter1d(cfs_out.real, size=ns, axis=0) + \
#                   1j * uniform_filter1d(cfs_out.imag, size=ns, axis=0)
#
#     return cfs_out
#
#
# # ─────────────────────────────────────────────
# # 2. CWT 计算，调用 pywt
# # ─────────────────────────────────────────────
#
# def _compute_cwt(x: np.ndarray, fs: float, wname: str,
#                  freq_limits: tuple, voices_per_octave: int,
#                  extend_sig: bool) -> tuple:
#     """
#     计算连续小波变换，对应 MATLAB cwt()。
#
#     Parameters
#     ----------
#     x                : 输入信号，shape (N,)
#     fs               : 采样率（Hz）
#     wname            : 小波类型 'amor'（Morlet）/ 'morse' / 'bump'
#     freq_limits      : (fmin, fmax) Hz
#     voices_per_octave: 每倍频程声部数
#     extend_sig       : 是否对称延拓信号
#
#     Returns
#     -------
#     coefs  : 复数 CWT 系数，shape (n_scales, N)
#     freqs  : 频率向量，shape (n_scales,)
#     scales : 尺度向量，shape (n_scales,)
#     coi    : 影响锥（Cone of Influence），shape (N,)
#     """
#     N  = len(x)
#     dt = 1.0 / fs
#
#     # 对称延拓（对应 MATLAB ExtendSignal=true）
#     if extend_sig:
#         x_ext = np.concatenate([x[::-1], x, x[::-1]])
#         pad   = N
#     else:
#         x_ext = x.copy()
#         pad   = 0
#
#     # 频率范围 → 尺度范围
#     fmin, fmax = freq_limits
#
#     # pywt 小波名称映射
#     wavelet_map = {
#         'amor' : 'cmor1.5-1.0',    # Morlet（近似 amor）
#         'morse': 'cmor1.5-1.0',    # 近似，pywt 无 Morse
#         'bump' : 'cmor1.0-0.5',    # 近似 bump
#     }
#     if wname not in wavelet_map:
#         raise ValueError(f"不支持的小波: '{wname}'，可选 'amor'/'morse'/'bump'")
#
#     wavelet = wavelet_map[wname]
#
#     # 计算尺度：从 fmax 到 fmin，每倍频程 voices_per_octave 个点
#     n_octaves = np.log2(fmax / fmin)
#     n_scales  = int(np.round(n_octaves * voices_per_octave))
#     freqs     = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)[::-1]
#
#     # pywt 中心频率转尺度：scale = center_freq / (freq * dt)
#     center_freq = pywt.central_frequency(wavelet)
#     scales      = center_freq / (freqs * dt)
#
#     # 执行 CWT
#     coefs_ext, _ = pywt.cwt(x_ext, scales, wavelet, sampling_period=dt)
#     # coefs_ext shape (n_scales, N_ext)
#
#     # 去掉延拓部分
#     if extend_sig:
#         coefs = coefs_ext[:, pad:pad + N]
#     else:
#         coefs = coefs_ext
#
#     # 计算影响锥（COI）
#     # 对应 MATLAB Coi：Morlet 小波的 COI = sqrt(2) * scale / fs
#     e_folding = np.sqrt(2)
#     coi_scales = e_folding * scales
#     coi_time   = np.minimum(
#         np.arange(1, N + 1),
#         np.arange(N, 0, -1)
#     ) * dt
#     # COI 以频率表示
#     coi = center_freq / (coi_time * dt) / e_folding * dt
#     # 限制在频率范围内
#     coi = np.clip(coi, fmin, fmax)
#
#     return coefs, freqs, scales, coi
#
#
# # ─────────────────────────────────────────────
# # 3. 主函数，对应 MATLAB My_Wxspectrum_TO()
# # ─────────────────────────────────────────────
#
#
# def wxspectrum(x_reference:  np.ndarray,
#                x_current:    np.ndarray,
#                fs:           float,
#                wname:        str        = 'amor',
#                freq_limits:  tuple      = (0.5, 5.0),
#                smoothing:    bool       = False,
#                ns:           int        = 3,
#                nt:           float      = 0.25,
#                voices_per_octave: int   = 10,
#                extend_sig:   bool       = True,
#                compute_coherence: bool  = True,
#                time_axis:    np.ndarray = None) -> dict:
#     """
#     小波互谱法计算时频域走时变化 dt(t, f)。
#     严格对应 MATLAB My_Wxspectrum_TO()。
#
#     Parameters
#     ----------
#     x_reference       : 参考波形，shape (N,)
#     x_current         : 当前波形，shape (N,)
#     fs                : 采样率（Hz）
#     wname             : 小波类型 'amor'（推荐）/ 'morse' / 'bump'
#     freq_limits       : (fmin, fmax) Hz，CWT 频率范围
#     smoothing         : 是否平滑（对应 MATLAB SmoothingFlag）
#                         推荐 False；True 仅适用于 Morlet 小波
#     ns                : boxcar 平滑窗长（尺度方向，对应 NumScalesToSmooth）
#     nt                : Gaussian 平滑宽度（对应 DegTimeToSmooth，默认 0.25）
#     voices_per_octave : 每倍频程声部数（推荐 ≥10，对应 VoicesPerOctave）
#     extend_sig        : 是否对称延拓（对应 ExtendSigFlag）
#     compute_coherence : 是否计算小波相干
#     time_axis         : 外部传入的时间轴，shape (N,)，单位秒。
#                         - 处理单边波形（如理论格林函数）：传 None，
#                           自动生成 time = np.arange(N) * dt
#                         - 处理 ACF/CCF（含负延迟）：传入对应的 lags 数组，
#                           例如 lags = np.arange(-maxlag_pts, maxlag_pts+1) / fs
#                           这样 t=0 在数组正中间，物理意义正确
#
#     Returns
#     -------
#     dict，包含：
#       'WXspec' : 小波互谱，shape (n_scales, N)，复数
#       'WXdt'   : 时间差（走时变化），shape (n_scales, N)，单位秒
#       'WXamp'  : 振幅乘积，shape (n_scales, N)
#       'Wcoh'   : 小波相干（若 compute_coherence=True），shape (n_scales, N)
#       'freq'   : 频率向量，shape (n_scales,)，Hz
#       'coi'    : 影响锥，shape (N,)，Hz
#       'time'   : 时间轴，shape (N,)，秒
#     """
#     # ── 输入检查 ──────────────────────────────────────────────────────
#     assert len(x_reference) == len(x_current), \
#         "x_reference 和 x_current 长度必须相同"
#     assert len(x_reference) >= 4, \
#         "信号长度至少为 4"
#     assert 4 <= voices_per_octave <= 48 and voices_per_octave % 2 == 0, \
#         "voices_per_octave 必须是 4~48 之间的偶数"
#
#     x_reference = np.asarray(x_reference, dtype=float).ravel()
#     x_current   = np.asarray(x_current,   dtype=float).ravel()
#     N  = len(x_reference)
#     dt = 1.0 / fs
#
#     # ── CWT ──────────────────────────────────────────────────────────
#     cwt_ref, freqs, scales, coi = _compute_cwt(
#         x_reference, fs, wname, freq_limits, voices_per_octave, extend_sig)
#     cwt_cur, _, _, _ = _compute_cwt(
#         x_current, fs, wname, freq_limits, voices_per_octave, extend_sig)
#     # cwt_ref / cwt_cur shape: (n_scales, N)
#
#     invscales = (1.0 / scales)[:, np.newaxis]   # shape (n_scales, 1)
#
#     # ── 互谱计算 ──────────────────────────────────────────────────────
#     if (not smoothing) or (ns == 1 and nt == 0):
#         # 对应 MATLAB Without Smoothing 分支
#         # crossCFS = cwt_reference .* conj(cwt_current)
#         crossCFS = cwt_ref * np.conj(cwt_cur)
#         WXamp    = np.abs(crossCFS)
#         WXspec   = crossCFS
#
#         if compute_coherence:
#             # 相干需要平滑，即使主互谱不平滑
#             cfs1     = _smooth_cfs(invscales * np.abs(cwt_cur) ** 2,
#                                    scales, dt, ns, nt)
#             cfs2     = _smooth_cfs(invscales * np.abs(cwt_ref) ** 2,
#                                    scales, dt, ns, nt)
#             cross_sm = _smooth_cfs(invscales * crossCFS,
#                                    scales, dt, ns, nt)
#             Wcoh     = np.abs(cross_sm) ** 2 / (
#                            np.real(cfs1) * np.real(cfs2) + 1e-30)
#         else:
#             Wcoh = None
#
#     else:
#         # 对应 MATLAB With Smoothing 分支
#         cfs1     = _smooth_cfs(invscales * np.abs(cwt_cur) ** 2,
#                                scales, dt, ns, nt)
#         cfs2     = _smooth_cfs(invscales * np.abs(cwt_ref) ** 2,
#                                scales, dt, ns, nt)
#         crossCFS = cwt_ref * np.conj(cwt_cur)
#         WXamp    = np.abs(crossCFS)
#
#         cross_sm = _smooth_cfs(invscales * crossCFS, scales, dt, ns, nt)
#
#         # 对应 MATLAB: WXspec = crossCFS ./ (sqrt(cfs1) .* sqrt(cfs2))
#         WXspec   = cross_sm / (
#                        np.sqrt(np.real(cfs1) + 1e-30) *
#                        np.sqrt(np.real(cfs2) + 1e-30))
#
#         # 对应 MATLAB: Wcoh = abs(crossCFS).^2 ./ (cfs1.*cfs2)
#         Wcoh     = np.abs(cross_sm) ** 2 / (
#                        np.real(cfs1) * np.real(cfs2) + 1e-30)
#
#     # ── 走时变化 WXdt ─────────────────────────────────────────────────
#     # 对应 MATLAB:
#     #   WXangle = angle(WXspec)
#     #   WXdt    = WXangle ./ repmat(2.*pi.*Freq, 1, nx)
#     WXangle = np.angle(WXspec)                               # 包裹相位差，shape (n_scales, N)
#     WXdt    = WXangle / (2.0 * np.pi * freqs[:, np.newaxis]) # shape (n_scales, N)，单位秒
#
#     # ── 时间轴 ────────────────────────────────────────────────────────
#     if time_axis is not None:
#         assert len(time_axis) == N, \
#             f"time_axis 长度 {len(time_axis)} 与波形长度 {N} 不一致"
#         time = np.asarray(time_axis, dtype=float)
#     else:
#         # 默认：从 t=0 开始的单边波形（如理论格林函数）
#         # 注意：处理 ACF/CCF 时必须从外部传入 lags 数组
#         time = np.arange(N) * dt
#
#     # ── 返回 ──────────────────────────────────────────────────────────
#     result = {
#         'WXspec': WXspec,   # 小波互谱，复数，shape (n_scales, N)
#         'WXdt'  : WXdt,     # 走时变化，秒，shape (n_scales, N)
#         'WXamp' : WXamp,    # 振幅乘积，shape (n_scales, N)
#         'freq'  : freqs,    # 频率向量，Hz，shape (n_scales,)
#         'coi'   : coi,      # 影响锥，Hz，shape (N,)
#         'time'  : time,     # 时间轴，秒，shape (N,)
#     }
#     if Wcoh is not None:
#         result['Wcoh'] = Wcoh   # 小波相干，shape (n_scales, N)
#
#     return result
#
#
# # ─────────────────────────────────────────────
# # 4. dv/v 提取，从 WXdt 计算平均 dv/v
# # ─────────────────────────────────────────────
#
# def dvv_from_wxdt(wxdt:        np.ndarray,
#                   wxamp:       np.ndarray,
#                   freqs:       np.ndarray,
#                   time:        np.ndarray,
#                   coi:         np.ndarray,
#                   freq_band:   tuple = None,
#                   lag_band:    tuple = None,
#                   amp_thresh:  float = 0.0,
#                   coi_mask:    bool  = True) -> dict:
#     """
#     从 WXdt 提取 dv/v。
#
#     dv/v = -dt/t，在时频域对满足条件的点做加权平均。
#
#     Parameters
#     ----------
#     wxdt       : 走时变化矩阵，shape (n_freq, n_time)，单位秒
#     wxamp      : 振幅矩阵，shape (n_freq, n_time)，用于加权
#     freqs      : 频率向量，shape (n_freq,)，Hz
#     time       : 时间轴，shape (n_time,)，秒
#     coi        : 影响锥频率，shape (n_time,)，Hz
#     freq_band  : (fmin, fmax) 提取 dv/v 的频带，None=全频带
#     lag_band   : (tmin, tmax) 提取 dv/v 的时间窗，None=全时窗
#     amp_thresh : 振幅阈值，低于此值的点不参与计算
#     coi_mask   : 是否剔除 COI 以外的点（受边界效应影响）
#
#     Returns
#     -------
#     dict：
#       'dvv'     : dv/v 时间序列（对频率平均），shape (n_time,)，单位 %
#       'dvv_freq': dv/v 时频矩阵，shape (n_freq, n_time)，单位 %
#       'weight'  : 权重矩阵，shape (n_freq, n_time)
#     """
#     n_freq, n_time = wxdt.shape
#
#     # ── 频带掩膜 ──────────────────────────────────────────────────────
#     if freq_band is not None:
#         fmask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
#     else:
#         fmask = np.ones(n_freq, dtype=bool)
#
#     # ── 时间窗掩膜 ────────────────────────────────────────────────────
#     if lag_band is not None:
#         tmask = (np.abs(time) >= lag_band[0]) & (np.abs(time) <= lag_band[1])
#     else:
#         tmask = np.ones(n_time, dtype=bool)
#
#     # ── COI 掩膜 ──────────────────────────────────────────────────────
#     # 对应 MATLAB 图中白色虚线内侧有效
#     # freqs[i] >= coi[j] 表示该点在 COI 内（有效）
#     if coi_mask:
#         coi_2d  = coi[np.newaxis, :]                    # (1, n_time)
#         freq_2d = freqs[:, np.newaxis]                  # (n_freq, 1)
#         coi_valid = freq_2d >= coi_2d                   # True=有效
#     else:
#         coi_valid = np.ones((n_freq, n_time), dtype=bool)
#
#     # ── 振幅掩膜 ──────────────────────────────────────────────────────
#     amp_valid = wxamp >= amp_thresh
#
#     # ── 综合掩膜 ──────────────────────────────────────────────────────
#     mask = (fmask[:, np.newaxis] &
#             tmask[np.newaxis, :] &
#             coi_valid &
#             amp_valid)
#
#     # ── dv/v = -dt/t，单位 % ──────────────────────────────────────────
#     t_2d        = time[np.newaxis, :]                   # (1, n_time)
#     # 避免 t=0 除零
#     t_safe      = np.where(np.abs(t_2d) > 1e-10, t_2d, np.nan)
#     dvv_tf      = -wxdt / t_safe * 100.0                # 单位 %
#
#     # ── 加权平均（权重=振幅）─────────────────────────────────────────
#     weight      = np.where(mask, wxamp, 0.0)
#     weight_sum  = weight.sum(axis=0)                    # shape (n_time,)
#
#     dvv_mean    = np.where(
#         weight_sum > 0,
#         np.nansum(dvv_tf * weight, axis=0) / (weight_sum + 1e-30),
#         np.nan
#     )
#
#     return {
#         'dvv'     : dvv_mean,    # shape (n_time,)，单位 %
#         'dvv_freq': dvv_tf,      # shape (n_freq, n_time)，单位 %
#         'weight'  : weight,      # shape (n_freq, n_time)
#         'mask'    : mask,
#     }


# """
# wxspectrum.py
# =============
# 小波互谱法测量地震走时变化（dv/v）。
# 严格对照 Mao et al. (2020) GJI 的 MATLAB 代码 My_Wxspectrum_TO.m 编写。
#
# 修复说明：
#     pywt.cwt 的复数共轭约定与 MATLAB cwt 相反，
#     互谱计算改为 conj(cwt_ref) * cwt_cur 以保证
#     WXdt = (t_cur - t_ref) 符号与 MATLAB 一致。
# """
#
# import numpy as np
# from scipy.fft import fft, ifft
# from scipy.signal import convolve2d
# import pywt
#
#
# # ─────────────────────────────────────────────
# # 1. 平滑函数，对应 MATLAB smoothCFS()
# # ─────────────────────────────────────────────
#
# def _smooth_cfs(cfs: np.ndarray, scales: np.ndarray, dt: float,
#                 ns: int, nt: float) -> np.ndarray:
#     """
#     对 CWT 系数做时间方向（Gaussian）+ 尺度方向（boxcar）平滑。
#     严格对应 MATLAB smoothCFS()。
#     """
#     n_scales, N = cfs.shape
#
#     # 对应 MATLAB: npad = 2^nextpow2(N)，必须是 2 的幂次
#     npad = int(2 ** np.ceil(np.log2(N)))
#
#     # 严格对应 MATLAB omega 构造：
#     # omega = 1:fix(npad/2); omega = omega*(2pi/npad);
#     # omega = [0, omega, -omega(fix((npad-1)/2):-1:1)]
#     omega_pos = np.arange(1, npad // 2 + 1) * (2 * np.pi / npad)
#     n_neg     = int((npad - 1) / 2)
#     omega_neg = -omega_pos[n_neg - 1::-1]
#     omega     = np.concatenate([[0.], omega_pos, omega_neg])  # 长度 = npad
#
#     normscales = scales / dt
#
#     cfs_out = cfs.copy().astype(complex)
#
#     # 时间方向：Gaussian 平滑（逐尺度，频域乘法）
#     for k_s in range(n_scales):
#         F      = np.exp(-nt * (normscales[k_s] ** 2) * omega ** 2)
#         smooth = ifft(F * fft(cfs_out[k_s, :], npad))
#         cfs_out[k_s, :] = smooth[:N]
#
#     # 尺度方向：boxcar 平滑
#     # convolve2d(..., 'same') 与 MATLAB conv2(...,'same') 完全等价，支持复数
#     if ns > 1:
#         H       = np.ones((ns, 1)) / ns
#         cfs_out = convolve2d(cfs_out, H, mode='same')
#
#     return cfs_out
#
#
# # ─────────────────────────────────────────────
# # 2. CWT 计算，调用 pywt
# # ─────────────────────────────────────────────
#
# def _compute_cwt(x: np.ndarray, fs: float, wname: str,
#                  freq_limits: tuple, voices_per_octave: int,
#                  extend_sig: bool) -> tuple:
#     """
#     计算连续小波变换，对应 MATLAB cwt()。
#     """
#     N  = len(x)
#     dt = 1.0 / fs
#
#     # 对称延拓（对应 MATLAB ExtendSignal=true）
#     if extend_sig:
#         x_ext = np.concatenate([x[::-1], x, x[::-1]])
#         pad   = N
#     else:
#         x_ext = x.copy()
#         pad   = 0
#
#     fmin, fmax = freq_limits
#
#     wavelet_map = {
#         'amor' : 'cmor1.5-1.0',
#         'morse': 'cmor1.5-1.0',
#         'bump' : 'cmor1.0-0.5',
#     }
#     if wname not in wavelet_map:
#         raise ValueError(f"不支持的小波: '{wname}'，可选 'amor'/'morse'/'bump'")
#
#     wavelet     = wavelet_map[wname]
#     center_freq = pywt.central_frequency(wavelet)
#
#     # 频率范围 → 对数均匀尺度，从高频到低频
#     n_octaves = np.log2(fmax / fmin)
#     n_scales  = int(np.round(n_octaves * voices_per_octave))
#     freqs     = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)[::-1]
#     scales    = center_freq / (freqs * dt)
#
#     # 执行 CWT
#     coefs_ext, _ = pywt.cwt(x_ext, scales, wavelet, sampling_period=dt)
#
#     # 去掉延拓部分
#     coefs = coefs_ext[:, pad:pad + N] if extend_sig else coefs_ext
#
#     # COI：Morlet 小波，单位 Hz
#     coi_time = np.minimum(np.arange(1, N + 1), np.arange(N, 0, -1)) * dt
#     coi      = center_freq / (np.sqrt(2) * coi_time)
#     coi      = np.clip(coi, fmin, fmax)
#
#     return coefs, freqs, scales, coi
#
#
# # ─────────────────────────────────────────────
# # 3. 主函数，对应 MATLAB My_Wxspectrum_TO()
# # ─────────────────────────────────────────────
#
# def wxspectrum(x_reference:  np.ndarray,
#                x_current:    np.ndarray,
#                fs:           float,
#                wname:        str        = 'amor',
#                freq_limits:  tuple      = (0.5, 5.0),
#                smoothing:    bool       = False,
#                ns:           int        = 3,
#                nt:           float      = 0.25,
#                voices_per_octave: int   = 10,
#                extend_sig:   bool       = True,
#                compute_coherence: bool  = True,
#                time_axis:    np.ndarray = None) -> dict:
#     """
#     小波互谱法计算时频域走时变化 dt(t, f)。
#     对应 MATLAB My_Wxspectrum_TO()。
#
#     WXdt 物理意义：t_current - t_reference（正值=当前波走时更长）
#     dv/v = -mean(WXdt / t)
#     """
#     assert len(x_reference) == len(x_current), \
#         "x_reference 和 x_current 长度必须相同"
#     assert len(x_reference) >= 4, \
#         "信号长度至少为 4"
#     assert 4 <= voices_per_octave <= 48 and voices_per_octave % 2 == 0, \
#         "voices_per_octave 必须是 4~48 之间的偶数"
#
#     x_reference = np.asarray(x_reference, dtype=float).ravel()
#     x_current   = np.asarray(x_current,   dtype=float).ravel()
#     N  = len(x_reference)
#     dt = 1.0 / fs
#
#     # ── CWT ──────────────────────────────────────────────────────────
#     cwt_ref, freqs, scales, coi = _compute_cwt(
#         x_reference, fs, wname, freq_limits, voices_per_octave, extend_sig)
#     cwt_cur, _, _, _ = _compute_cwt(
#         x_current, fs, wname, freq_limits, voices_per_octave, extend_sig)
#
#     invscales = (1.0 / scales)[:, np.newaxis]
#
#     # ── 互谱计算 ──────────────────────────────────────────────────────
#     # 注意：pywt 复数约定与 MATLAB 相反，
#     # 此处用 conj(cwt_ref)*cwt_cur 保证 WXdt = t_cur - t_ref 符号正确
#     if (not smoothing) or (ns == 1 and nt == 0):
#         crossCFS = np.conj(cwt_ref) * cwt_cur
#         WXamp    = np.abs(crossCFS)
#         WXspec   = crossCFS
#
#         if compute_coherence:
#             cfs1     = _smooth_cfs(invscales * np.abs(cwt_cur) ** 2,
#                                    scales, dt, ns, nt)
#             cfs2     = _smooth_cfs(invscales * np.abs(cwt_ref) ** 2,
#                                    scales, dt, ns, nt)
#             cross_sm = _smooth_cfs(invscales * crossCFS,
#                                    scales, dt, ns, nt)
#             Wcoh     = np.abs(cross_sm) ** 2 / (
#                            np.real(cfs1) * np.real(cfs2) + 1e-30)
#         else:
#             Wcoh = None
#
#     else:
#         cfs1     = _smooth_cfs(invscales * np.abs(cwt_cur) ** 2,
#                                scales, dt, ns, nt)
#         cfs2     = _smooth_cfs(invscales * np.abs(cwt_ref) ** 2,
#                                scales, dt, ns, nt)
#         crossCFS = np.conj(cwt_ref) * cwt_cur
#         WXamp    = np.abs(crossCFS)
#
#         cross_sm = _smooth_cfs(invscales * crossCFS, scales, dt, ns, nt)
#
#         WXspec   = cross_sm / (
#                        np.sqrt(np.real(cfs1) + 1e-30) *
#                        np.sqrt(np.real(cfs2) + 1e-30))
#         Wcoh     = np.abs(cross_sm) ** 2 / (
#                        np.real(cfs1) * np.real(cfs2) + 1e-30)
#
#     # ── 走时变化 WXdt ─────────────────────────────────────────────────
#     WXangle = np.angle(WXspec)
#     WXdt    = WXangle / (2.0 * np.pi * freqs[:, np.newaxis])
#
#     # ── 时间轴 ────────────────────────────────────────────────────────
#     if time_axis is not None:
#         assert len(time_axis) == N, \
#             f"time_axis 长度 {len(time_axis)} 与波形长度 {N} 不一致"
#         time = np.asarray(time_axis, dtype=float)
#     else:
#         time = np.arange(N) * dt
#
#     result = {
#         'WXspec': WXspec,
#         'WXdt'  : WXdt,
#         'WXamp' : WXamp,
#         'freq'  : freqs,
#         'coi'   : coi,
#         'time'  : time,
#     }
#     if Wcoh is not None:
#         result['Wcoh'] = Wcoh
#
#     return result
#
#
# # ─────────────────────────────────────────────
# # 4. dv/v 提取
# # ─────────────────────────────────────────────
#
# def dvv_from_wxdt(wxdt:        np.ndarray,
#                   wxamp:       np.ndarray,
#                   freqs:       np.ndarray,
#                   time:        np.ndarray,
#                   coi:         np.ndarray,
#                   freq_band:   tuple = None,
#                   lag_band:    tuple = None,
#                   amp_thresh:  float = 0.0,
#                   coi_mask:    bool  = True) -> dict:
#     """
#     从 WXdt 提取 dv/v = -dt/t，加权平均。
#     """
#     n_freq, n_time = wxdt.shape
#
#     if freq_band is not None:
#         fmask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
#     else:
#         fmask = np.ones(n_freq, dtype=bool)
#
#     if lag_band is not None:
#         tmask = (np.abs(time) >= lag_band[0]) & (np.abs(time) <= lag_band[1])
#     else:
#         tmask = np.ones(n_time, dtype=bool)
#
#     if coi_mask:
#         coi_2d    = coi[np.newaxis, :]
#         freq_2d   = freqs[:, np.newaxis]
#         coi_valid = freq_2d >= coi_2d
#     else:
#         coi_valid = np.ones((n_freq, n_time), dtype=bool)
#
#     amp_valid = wxamp >= amp_thresh
#
#     mask = (fmask[:, np.newaxis] &
#             tmask[np.newaxis, :] &
#             coi_valid &
#             amp_valid)
#
#     t_2d   = time[np.newaxis, :]
#     t_safe = np.where(np.abs(t_2d) > 1e-10, t_2d, np.nan)
#     dvv_tf = -wxdt / t_safe * 100.0   # 单位 %
#
#     weight     = np.where(mask, wxamp, 0.0)
#     weight_sum = weight.sum(axis=0)
#
#     dvv_mean = np.where(
#         weight_sum > 0,
#         np.nansum(dvv_tf * weight, axis=0) / (weight_sum + 1e-30),
#         np.nan
#     )
#
#     return {
#         'dvv'     : dvv_mean,
#         'dvv_freq': dvv_tf,
#         'weight'  : weight,
#         'mask'    : mask,
#     }



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
    严格对应 MATLAB smoothCFS()。
    """
    n_scales, N = cfs.shape

    # 对应 MATLAB: npad = 2^nextpow2(N)  ← 必须是 2 的幂次
    npad = int(2 ** np.ceil(np.log2(N)))

    # 严格对应 MATLAB omega 构造
    # omega = 1:fix(npad/2); omega = omega*(2pi/npad);
    # omega = [0, omega, -omega(fix((npad-1)/2):-1:1)]
    omega_pos = np.arange(1, npad // 2 + 1) * (2 * np.pi / npad)
    n_neg     = int((npad - 1) / 2)                   # fix((npad-1)/2)
    omega_neg = -omega_pos[n_neg - 1:: -1]            # 逆序取前 n_neg 个
    omega     = np.concatenate([[0.], omega_pos, omega_neg])  # 长度 = npad

    # 归一化尺度（对应 MATLAB normscales = scales./dt）
    normscales = scales / dt

    cfs_out = cfs.copy().astype(complex)

    # 时间方向：Gaussian 平滑（逐尺度，频域乘法）
    for k_s in range(n_scales):
        F      = np.exp(-nt * (normscales[k_s] ** 2) * omega ** 2)
        smooth = ifft(F * fft(cfs_out[k_s, :], npad))
        cfs_out[k_s, :] = smooth[:N]

    # 尺度方向：boxcar 平滑
    # 对应 MATLAB: H = 1/ns*ones(ns,1); cfs = conv2(cfs,H,'same')
    # convolve2d 与 MATLAB conv2(...,'same') 完全等价，且支持复数
    if ns > 1:
        H       = np.ones((ns, 1)) / ns
        cfs_out = convolve2d(cfs_out, H, mode='same')

    return cfs_out


# ─────────────────────────────────────────────
# 2. CWT 计算，调用 pywt
# ─────────────────────────────────────────────

def _compute_cwt(x: np.ndarray, fs: float, wname: str,
                 freq_limits: tuple, voices_per_octave: int,
                 extend_sig: bool) -> tuple:
    """
    计算连续小波变换，对应 MATLAB cwt()。
    """
    N  = len(x)
    dt = 1.0 / fs

    # 对称延拓（对应 MATLAB ExtendSignal=true）
    if extend_sig:
        x_ext = np.concatenate([x[::-1], x, x[::-1]])
        pad   = N
    else:
        x_ext = x.copy()
        pad   = 0

    fmin, fmax = freq_limits

    wavelet_map = {
        'amor' : 'cmor1.5-1.0',
        'morse': 'cmor1.5-1.0',
        'bump' : 'cmor1.0-0.5',
    }
    if wname not in wavelet_map:
        raise ValueError(f"不支持的小波: '{wname}'，可选 'amor'/'morse'/'bump'")

    wavelet     = wavelet_map[wname]
    center_freq = pywt.central_frequency(wavelet)

    # 频率范围 → 对数均匀尺度，从高频到低频（对应 MATLAB cwt 输出顺序）
    n_octaves = np.log2(fmax / fmin)
    n_scales  = int(np.round(n_octaves * voices_per_octave))
    freqs     = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)[::-1]
    scales    = center_freq / (freqs * dt)

    # 执行 CWT
    coefs_ext, _ = pywt.cwt(x_ext, scales, wavelet, sampling_period=dt)

    # 去掉延拓部分
    coefs = coefs_ext[:, pad:pad + N] if extend_sig else coefs_ext

    # ── COI：Morlet 小波，对应 MATLAB Coi 输出（单位 Hz）────────────
    # 距两端的时间距离（秒）
    coi_time = np.minimum(np.arange(1, N + 1), np.arange(N, 0, -1)) * dt
    # f_coi = center_freq / (sqrt(2) * coi_time)
    coi = center_freq / (np.sqrt(2) * coi_time)
    coi = np.clip(coi, fmin, fmax)

    return coefs, freqs, scales, coi


# ─────────────────────────────────────────────
# 3. 主函数，对应 MATLAB My_Wxspectrum_TO()
# ─────────────────────────────────────────────

def wxspectrum(x_reference:  np.ndarray,
               x_current:    np.ndarray,
               fs:           float,
               wname:        str        = 'amor',
               freq_limits:  tuple      = (0.5, 5.0),
               smoothing:    bool       = False,
               ns:           int        = 3,
               nt:           float      = 0.25,
               voices_per_octave: int   = 10,
               extend_sig:   bool       = True,
               compute_coherence: bool  = True,
               time_axis:    np.ndarray = None) -> dict:
    """
    小波互谱法计算时频域走时变化 dt(t, f)。
    严格对应 MATLAB My_Wxspectrum_TO()。
    """
    assert len(x_reference) == len(x_current), \
        "x_reference 和 x_current 长度必须相同"
    assert len(x_reference) >= 4, \
        "信号长度至少为 4"
    assert 4 <= voices_per_octave <= 48 and voices_per_octave % 2 == 0, \
        "voices_per_octave 必须是 4~48 之间的偶数"

    x_reference = np.asarray(x_reference, dtype=float).ravel()
    x_current   = np.asarray(x_current,   dtype=float).ravel()
    N  = len(x_reference)
    dt = 1.0 / fs

    # ── CWT ──────────────────────────────────────────────────────────
    cwt_ref, freqs, scales, coi = _compute_cwt(
        x_reference, fs, wname, freq_limits, voices_per_octave, extend_sig)
    cwt_cur, _, _, _ = _compute_cwt(
        x_current, fs, wname, freq_limits, voices_per_octave, extend_sig)

    invscales = (1.0 / scales)[:, np.newaxis]   # shape (n_scales, 1)

    # ── 互谱计算 ──────────────────────────────────────────────────────
    if (not smoothing) or (ns == 1 and nt == 0):
        # 对应 MATLAB Without Smoothing 分支
        crossCFS = cwt_ref * np.conj(cwt_cur)
        WXamp    = np.abs(crossCFS)
        WXspec   = crossCFS

        if compute_coherence:
            cfs1     = _smooth_cfs(invscales * np.abs(cwt_cur) ** 2,
                                   scales, dt, ns, nt)
            cfs2     = _smooth_cfs(invscales * np.abs(cwt_ref) ** 2,
                                   scales, dt, ns, nt)
            cross_sm = _smooth_cfs(invscales * crossCFS,
                                   scales, dt, ns, nt)
            Wcoh     = np.abs(cross_sm) ** 2 / (
                           np.real(cfs1) * np.real(cfs2) + 1e-30)
        else:
            Wcoh = None

    else:
        # 对应 MATLAB With Smoothing 分支
        cfs1     = _smooth_cfs(invscales * np.abs(cwt_cur) ** 2,
                               scales, dt, ns, nt)
        cfs2     = _smooth_cfs(invscales * np.abs(cwt_ref) ** 2,
                               scales, dt, ns, nt)
        crossCFS = cwt_ref * np.conj(cwt_cur)
        WXamp    = np.abs(crossCFS)

        cross_sm = _smooth_cfs(invscales * crossCFS, scales, dt, ns, nt)

        WXspec   = cross_sm / (
                       np.sqrt(np.real(cfs1) + 1e-30) *
                       np.sqrt(np.real(cfs2) + 1e-30))
        Wcoh     = np.abs(cross_sm) ** 2 / (
                       np.real(cfs1) * np.real(cfs2) + 1e-30)

    # ── 走时变化 WXdt ─────────────────────────────────────────────────
    # 对应 MATLAB: WXdt = angle(WXspec) ./ (2*pi*Freq)
    WXangle = np.angle(WXspec)
    WXdt    = WXangle / (2.0 * np.pi * freqs[:, np.newaxis])

    # ── 时间轴 ────────────────────────────────────────────────────────
    if time_axis is not None:
        assert len(time_axis) == N, \
            f"time_axis 长度 {len(time_axis)} 与波形长度 {N} 不一致"
        time = np.asarray(time_axis, dtype=float)
    else:
        time = np.arange(N) * dt

    # ── 返回 ──────────────────────────────────────────────────────────
    result = {
        'WXspec': WXspec,
        'WXdt'  : WXdt,
        'WXamp' : WXamp,
        'freq'  : freqs,
        'coi'   : coi,
        'time'  : time,
    }
    if Wcoh is not None:
        result['Wcoh'] = Wcoh

    return result


# ─────────────────────────────────────────────
# 4. dv/v 提取
# ─────────────────────────────────────────────

def dvv_from_wxdt(wxdt:        np.ndarray,
                  wxamp:       np.ndarray,
                  freqs:       np.ndarray,
                  time:        np.ndarray,
                  coi:         np.ndarray,
                  freq_band:   tuple = None,
                  lag_band:    tuple = None,
                  amp_thresh:  float = 0.0,
                  coi_mask:    bool  = True) -> dict:
    """
    从 WXdt 提取 dv/v = -dt/t，加权平均。

    注意：lag_band 对 time 取绝对值后匹配，适用于单边和双边波形。
    若需要区分正负延迟，请在调用前手动构造 tmask 后传入。
    """
    n_freq, n_time = wxdt.shape

    if freq_band is not None:
        fmask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    else:
        fmask = np.ones(n_freq, dtype=bool)

    if lag_band is not None:
        # 对 |time| 做窗选，正负延迟都包含
        tmask = (np.abs(time) >= lag_band[0]) & (np.abs(time) <= lag_band[1])
    else:
        tmask = np.ones(n_time, dtype=bool)

    if coi_mask:
        coi_2d    = coi[np.newaxis, :]
        freq_2d   = freqs[:, np.newaxis]
        coi_valid = freq_2d >= coi_2d
    else:
        coi_valid = np.ones((n_freq, n_time), dtype=bool)

    amp_valid = wxamp >= amp_thresh

    mask = (fmask[:, np.newaxis] &
            tmask[np.newaxis, :] &
            coi_valid &
            amp_valid)

    t_2d   = time[np.newaxis, :]
    t_safe = np.where(np.abs(t_2d) > 1e-10, t_2d, np.nan)
    dvv_tf = -wxdt / t_safe * 100.0   # 单位 %

    weight     = np.where(mask, wxamp, 0.0)
    weight_sum = weight.sum(axis=0)

    dvv_mean = np.where(
        weight_sum > 0,
        np.nansum(dvv_tf * weight, axis=0) / (weight_sum + 1e-30),
        np.nan
    )

    return {
        'dvv'     : dvv_mean,
        'dvv_freq': dvv_tf,
        'weight'  : weight,
        'mask'    : mask,
    }