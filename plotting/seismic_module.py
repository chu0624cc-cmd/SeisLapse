import numpy as np
from numpy.fft import fft, ifft, fftfreq
from dataclasses import dataclass
from plotting.seisplot import Extent


# ==========================================================
# 1. SeisLinearEvents — 生成线性事件
# ==========================================================
def SeisLinearEvents(
    nt=256, dt=0.004,
    nx1=64, dx1=10.0,
    p1=None, tau=None, amp=None,
    f0=20.0,
):
    """
    生成多个线性 moveout 地震事件（Ricker 子波）

    参数
    ----
    nt   : 时间采样点数
    dt   : 时间采样间隔 (s)
    nx1  : 空间道数
    dx1  : 道间距 (m)
    p1   : 各事件慢度列表 (s/m)
    tau  : 各事件截距时间列表 (s)
    amp  : 各事件振幅列表
    f0   : Ricker 子波主频 (Hz)

    返回
    ----
    d    : np.ndarray, shape (nt, nx1)
    ext  : Extent 对象
    """
    if p1  is None: p1  = [0.0]
    if tau is None: tau = [0.3]
    if amp is None: amp = [1.0]

    t = np.arange(nt) * dt
    x = np.arange(nx1) * dx1
    d = np.zeros((nt, nx1))

    for slope, t0, a in zip(p1, tau, amp):
        for ix in range(nx1):
            t_shift = t0 + slope * x[ix]
            # Ricker 子波
            u = np.pi * f0 * (t - t_shift)
            wavelet = a * (1.0 - 2.0 * u**2) * np.exp(-(u**2))
            d[:, ix] += wavelet

    ext = Extent(
        title="Linear Events",
        label1="Time",   unit1="s",
        label2="Offset", unit2="m",
        o1=0.0,  d1=dt,
        o2=0.0,  d2=dx1,
    )
    return d, ext


# ==========================================================
# 2. SeisRadonFreqFor — 频率域线性 Radon 正演
#    d(t, h) = ∫ m(τ, p) δ(t - τ - p·h) dp
# ==========================================================
def SeisRadonFreqFor(
    m, nt,
    dt=0.004, h=None, p=None,
    flow=2, fhigh=80,
):
    """
    频率域线性 Radon 正演：τ-p → t-x

    参数
    ----
    m     : np.ndarray, shape (nt_m, np_)  Radon 域数据
    nt    : 输出时间采样点数
    dt    : 时间采样间隔 (s)
    h     : offset 数组 (m), shape (nx,)
    p     : 慢度数组 (s/m), shape (np_,)
    flow  : 最低处理频率 (Hz)
    fhigh : 最高处理频率 (Hz)

    返回
    ----
    d : np.ndarray, shape (nt, nx)
    """
    if h is None: h = np.array([0.0])
    if p is None: p = np.array([0.0])

    nx   = len(h)
    np_  = len(p)
    nt_m = m.shape[0]

    # 补零到输出长度
    nfft = max(nt, nt_m)
    freq = fftfreq(nfft, d=dt)
    nf   = nfft // 2 + 1

    M = fft(m, n=nfft, axis=0)   # (nfft, np_)
    D = np.zeros((nfft, nx), dtype=complex)

    for ifreq in range(nf):
        f = freq[ifreq]
        if abs(f) < flow or abs(f) > fhigh:
            continue
        # 相移矩阵 L : (nx, np_)
        L = np.exp(1j * 2 * np.pi * f * np.outer(h, p))
        D[ifreq, :] = L @ M[ifreq, :]

    # 利用共轭对称性填充负频率
    for ifreq in range(1, nf - 1):
        D[nfft - ifreq, :] = np.conj(D[ifreq, :])

    d = np.real(ifft(D, axis=0))[:nt, :]
    return d


# ==========================================================
# 3. SeisRadonFreqInv — 频率域线性 Radon 反演
#    最小二乘 + Tikhonov 正则化
#    m = (L^H L + μI)^{-1} L^H d
# ==========================================================
def SeisRadonFreqInv(
    d,
    dt=0.004, h=None, p=None,
    flow=2, fhigh=80,
    mu=1e-5,
):
    """
    频率域线性 Radon 反演（最小二乘 + L2 正则化）

    参数
    ----
    d     : np.ndarray, shape (nt, nx)  输入地震数据
    dt    : 时间采样间隔 (s)
    h     : offset 数组 (m), shape (nx,)
    p     : 慢度数组 (s/m), shape (np_,)
    flow  : 最低处理频率 (Hz)
    fhigh : 最高处理频率 (Hz)
    mu    : Tikhonov 正则化参数（越大越平滑）

    返回
    ----
    m : np.ndarray, shape (nt, np_)  Radon 域数据
    """
    if h is None: h = np.array([0.0])
    if p is None: p = np.array([0.0])

    nt, nx = d.shape
    np_    = len(p)
    nfft   = nt
    freq   = fftfreq(nfft, d=dt)
    nf     = nfft // 2 + 1

    D = fft(d, axis=0)            # (nfft, nx)
    M = np.zeros((nfft, np_), dtype=complex)

    for ifreq in range(nf):
        f = freq[ifreq]
        if abs(f) < flow or abs(f) > fhigh:
            continue
        # 相移矩阵 L : (nx, np_)
        L = np.exp(1j * 2 * np.pi * f * np.outer(h, p))
        # 正则化最小二乘：m = (L^H L + μI)^{-1} L^H d
        LH  = L.conj().T                          # (np_, nx)
        A   = LH @ L + mu * np.eye(np_)           # (np_, np_)
        rhs = LH @ D[ifreq, :]                    # (np_,)
        M[ifreq, :] = np.linalg.solve(A, rhs)

    # 共轭对称填充
    for ifreq in range(1, nf - 1):
        M[nfft - ifreq, :] = np.conj(M[ifreq, :])

    m = np.real(ifft(M, axis=0))
    return m