"""
stacking.py
===========
时间叠加模块，严格对应 Julia SeisNoise stack.jl。

包含：
  linear_stack     ← Julia mean stack
  pws              ← Julia pws（相位加权叠加，Schimmel & Paulssen 1997）
  robust_stack     ← Julia robuststack（Pavlis & Vernon 2010）
  robust_pws       ← Julia robustpws（联合鲁棒+相位加权）
  median_mute      ← Julia medianmute（剔除振幅异常窗口）
  remove_nan       ← Julia remove_nan
  shorten          ← Julia shorten（截短滞后范围）
  stack            ← Julia stack（按时间间隔或全叠加，统一入口）
  normalize_stack
"""

import numpy as np
from scipy.signal import hilbert
from scipy.linalg import norm


# ─────────────────────────────────────────────
# 1. 工具函数
# ─────────────────────────────────────────────

def remove_nan(corr_data: dict) -> dict:
    """
    剔除含 NaN 的窗口列。
    对应 Julia remove_nan!(C)
    """
    x   = corr_data['corr']
    t   = corr_data['t']
    ind = [i for i in range(x.shape[1]) if not np.any(np.isnan(x[:, i]))]

    if len(ind) == 0:
        raise ValueError("所有窗口均含 NaN，无法继续。")

    return {**corr_data,
            'corr': x[:, ind],
            't'   : t[ind]}


def median_mute(corr_data: dict,
                high: float = 10.0,
                low:  float = 0.0) -> dict:
    """
    剔除最大振幅超过中位数 high 倍或低于 low 倍的窗口。
    对应 Julia medianmute!(C, high, low)

    Parameters
    ----------
    high : 振幅上限倍数（默认 10）
    low  : 振幅下限倍数（默认 0，即不剔除过小窗口）
    """
    assert low < high, "low 必须小于 high"

    x      = corr_data['corr']
    t      = corr_data['t']

    # 对应 Julia: maxamp = vec(maximum(abs.(A), dims=1))
    maxamp = np.max(np.abs(x), axis=0).astype(float)
    maxamp[np.isnan(maxamp)] = np.inf

    # 对应 Julia: medianmax = median(maxamp)
    medianmax = np.median(maxamp)

    # 对应 Julia: ind = findall(x-> low*medianmax <= x <= high*medianmax, maxamp)
    ind = np.where((maxamp >= low  * medianmax) &
                   (maxamp <= high * medianmax))[0]

    n_removed = x.shape[1] - len(ind)
    if n_removed > 0:
        print(f"  median_mute：剔除 {n_removed} 个异常窗口 "
              f"（阈值 {low}~{high} × median={medianmax:.3e}）")

    return {**corr_data,
            'corr': x[:, ind],
            't'   : t[ind]}


def shorten(corr_data: dict, newlag: float) -> dict:
    """
    截短滞后范围到 ±newlag 秒。
    对应 Julia shorten!(C, newlag)

    Parameters
    ----------
    newlag : 新的最大滞后时间（秒），必须 < 原 maxlag
    """
    maxlag = corr_data['maxlag']
    assert 0 < newlag < maxlag, \
        f"newlag={newlag} 必须满足 0 < newlag < maxlag={maxlag}"

    lags = corr_data['lags']

    # 对应 Julia: ind = findall(x -> abs(x) <= newlag, lags)
    ind  = np.where(np.abs(lags) <= newlag)[0]

    return {**corr_data,
            'corr'  : corr_data['corr'][ind, :],
            'lags'  : lags[ind],
            'maxlag': float(lags[ind[-1]])}


def normalize_stack(stacked: dict) -> dict:
    """
    对叠加结果做 abs_max 归一化，使 max=1.0。

    robust_stack 和 robust_pws 输出的绝对振幅偏小，
    归一化后与 linear/pws 可比，便于后续 dv/v 计算。

    Parameters
    ----------
    stacked : stack() 的返回值，corr shape (n_lags, n_stacks)
    Returns
    -------
    归一化后的 stacked 字典，corr 每列除以其绝对值最大值
    """
    corr = stacked['corr'].copy()
    max_val = np.max(np.abs(corr), axis=0, keepdims=True)  # shape (1, n_stacks)
    max_val = np.where(max_val > 0, max_val, 1.0)  # 避免除零
    corr /= max_val

    return {**stacked, 'corr': corr}


# ─────────────────────────────────────────────
# 2. 叠加方法
# ─────────────────────────────────────────────

def linear_stack(A: np.ndarray) -> np.ndarray:
    """线性叠加，对应 Julia mean(C.corr, dims=2)"""
    return np.mean(A, axis=1, keepdims=True)


def pws(A: np.ndarray, pow: float = 2.0) -> np.ndarray:
    """
    相位加权叠加，对应 Julia pws(A; pow=2)
    Schimmel & Paulssen 1997
    """
    n_lags, N   = A.shape
    phase       = np.angle(hilbert(A, axis=0))          # (n_lags, N)
    phase_stack = (np.abs(
                       np.sum(np.exp(1j * phase), axis=1, keepdims=True) / N
                   ) ** pow)                             # (n_lags, 1)

    # 优化：先均值再加权，数学等价且省内存
    return np.mean(A, axis=1, keepdims=True) * phase_stack


def robust_stack(A: np.ndarray,
                 eps:     float = 1e-4,
                 maxiter: int   = 10) -> np.ndarray:
    """
    鲁棒叠加，对应 Julia robuststack(A; ϵ=1e-4, maxiter=10)
    Pavlis & Vernon 2010
    """
    n_lags, N = A.shape
    Bold = np.median(A, axis=1, keepdims=True)
    Bold /= (norm(Bold, 2) + 1e-30)
    d2   = norm(A, axis=0)                              # shape (N,)

    def _update_weights(B):
        # 对应 Julia: BdotD = sum(A .* Bold, dims=1)
        BdotD    = np.sum(A * B, axis=0)                # shape (N,)
        # 对应 Julia: A[:,ii] .- (BdotD[ii] .* Bold)
        # 关键：BdotD[np.newaxis,:] * B → (n_lags, N)，不是外积
        residual = A - BdotD[np.newaxis, :] * B         # (n_lags, N) ✅
        res_norm = norm(residual, axis=0)               # shape (N,)
        return np.abs(BdotD) / (d2 * res_norm + 1e-30)

    w     = _update_weights(Bold)
    Bnew  = np.average(A, axis=1, weights=w).reshape(-1, 1)
    Bnew /= (norm(Bnew, 2) + 1e-30)

    eps_n = norm(Bnew - Bold, 1) / (norm(Bnew, 2) * N + 1e-30)
    Bold  = Bnew
    itr   = 0

    while eps_n > eps and itr <= maxiter:
        w     = _update_weights(Bold)
        Bnew  = np.average(A, axis=1, weights=w).reshape(-1, 1)
        Bnew /= (norm(Bnew, 2) + 1e-30)
        eps_n = norm(Bnew - Bold, 1) / (norm(Bnew, 2) * N + 1e-30)
        Bold  = Bnew
        itr  += 1

    print(f"  robust_stack：收敛于第 {itr} 次迭代，ϵN={eps_n:.2e}")
    return Bnew


def robust_pws(A: np.ndarray,
               eps:     float = 1e-6,
               maxiter: int   = 10,
               pow:     float = 2.0) -> np.ndarray:
    """
    联合鲁棒+相位加权叠加，对应 Julia robustpws(A; ϵ=1e-6, maxiter=10, pow=2)
    严格遵循 Julia：W = A .* w'; return pws(W, pow=pow)
    """
    n_lags, N = A.shape
    Bold = np.median(A, axis=1, keepdims=True)
    # 注意：robustpws 初始值不归一化（与 robuststack 不同）
    d2   = norm(A, axis=0)                              # shape (N,)

    def _update_weights(B):
        BdotD    = np.sum(A * B, axis=0)                # shape (N,)
        residual = A - BdotD[np.newaxis, :] * B         # (n_lags, N) ✅
        res_norm = norm(residual, axis=0)
        w_loc    = np.abs(BdotD) / (d2 * res_norm + 1e-30)
        w_loc   /= (w_loc.sum() + 1e-30)               # 归一化，robustpws 特有
        return w_loc

    w     = _update_weights(Bold)
    Bnew  = np.average(A, axis=1, weights=w).reshape(-1, 1)
    eps_n = norm(Bnew - Bold, 2) / (norm(Bnew, 2) * N + 1e-30)
    Bold  = Bnew
    itr   = 0

    while eps_n > eps and itr <= maxiter:
        w     = _update_weights(Bold)
        Bnew  = np.average(A, axis=1, weights=w).reshape(-1, 1)
        eps_n = norm(Bnew - Bold, 2) / (norm(Bnew, 2) * N + 1e-30)
        Bold  = Bnew
        itr  += 1

    print(f"  robust_pws：收敛于第 {itr} 次迭代，ϵN={eps_n:.2e}")


    W = A * w[np.newaxis, :]                            # (n_lags, N)
    return pws(W, pow=pow)


# ─────────────────────────────────────────────
# 3. 统一叠加入口
# ─────────────────────────────────────────────

# 叠加方法名 → 函数映射
_STACK_FUNC = {
    'linear'     : linear_stack,
    'mean'       : linear_stack,   # 别名
    'pws'        : pws,
    'robust'     : robust_stack,
    'robuststack': robust_stack,   # 别名
    'robust_pws' : robust_pws,
    'robustpws'  : robust_pws,     # 别名
}


def stack(corr_data: dict,
          method:    str   = 'linear',
          allstack:  bool  = True,
          interval:  float = 86400.0,
          pws_pow:   float = 2.0,
          robust_eps: float = 1e-4,
          maxiter:   int   = 10,
          normalize: bool  = True) -> dict:   # ← 新增
    """
    统一叠加入口，对应 Julia stack!(C; interval, allstack, stacktype)。

    Parameters
    ----------
    corr_data  : process_acf 返回的 CorrData 字典
    method     : 叠加方式
                 'linear'/'mean'   → 线性叠加（Julia 默认）
                 'pws'             → 相位加权叠加
                 'robust'          → 鲁棒叠加
                 'robust_pws'      → 联合鲁棒+相位加权
    allstack   : True  → 全部窗口叠成一个（对应 Julia allstack=true）
                 False → 按 interval 分组叠加
    interval   : allstack=False 时的分组间隔（秒，默认 86400=1天）
    pws_pow    : PWS 相位权重锐度（默认 2）
    robust_eps : robust 收敛阈值（默认 1e-4）
    maxiter    : robust 最大迭代次数（默认 10）
    normalize  : 是否对叠加结果做 abs_max 归一化（默认 True）
                 确保所有方法输出振幅可比（max=1.0）
                 对 linear/pws 无影响，主要修正 robust/robust_pws 的振幅偏小问题

    Returns
    -------
    叠加后的 CorrData 字典，corr shape (n_lags, n_stacks)
    """
    if method not in _STACK_FUNC:
        raise ValueError(
            f"未知叠加方法: '{method}'，"
            f"可选 {list(_STACK_FUNC.keys())}"
        )

    stack_fn = _STACK_FUNC[method]

    def _call(A):
        if method in ('pws',):
            return stack_fn(A, pow=pws_pow)
        elif method in ('robust', 'robuststack'):
            return stack_fn(A, eps=robust_eps, maxiter=maxiter)
        elif method in ('robust_pws', 'robustpws'):
            return stack_fn(A, eps=robust_eps, maxiter=maxiter, pow=pws_pow)
        else:
            return stack_fn(A)

    corr = corr_data['corr']
    t    = corr_data['t']

    if allstack:
        stacked = _call(corr)
        t_out   = np.array([t[0]])
        print(f"  stack：{method}  全叠加  {corr.shape[1]} 窗口 → 1 叠加")

    else:
        t_floor  = (t // interval) * interval
        t_unique = np.unique(t_floor)
        stack_list = []
        for t_grp in t_unique:
            ind = np.where(t_floor == t_grp)[0]
            stack_list.append(_call(corr[:, ind]))
        stacked = np.hstack(stack_list)
        t_out   = t_unique
        print(f"  stack：{method}  interval={interval/3600:.1f}h  "
              f"{corr.shape[1]} 窗口 → {len(t_unique)} 叠加")

    result = {**corr_data,
              'corr'      : stacked,
              't'         : t_out,
              'stack_type': method,
              'allstack'  : allstack}

    # ── abs_max 归一化 ────────────────────────────────────────────────
    if normalize:
        result = normalize_stack(result)

    return result