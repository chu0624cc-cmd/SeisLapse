import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
from dataclasses import dataclass
from typing import Optional

"""
SeisPlot — 地震数据二维可视化函数
====================================

调用方式
--------
SeisPlot(
    d,                          # 必填：二维 numpy 数组 (nt, nx)

    # ── 绘图类型与风格 ──────────────────────────────
    plot_type = "TX",           # "TX" | "FK" | "Amplitude"
    style     = "color",        # "color" | "wiggles" | "overlay"  (仅TX)
    cmap      = "gray",         # 任意 matplotlib colormap

    # ── 振幅控制 ────────────────────────────────────
    pclip     = 98,             # 百分位裁剪 (0~100)
    vmin      = None,           # 手动色标下限（设置后pclip失效）
    vmax      = None,           # 手动色标上限

    # ── 坐标轴参数 ──────────────────────────────────
    ox        = 0,              # x轴起点
    dx        = 1,              # x轴采样间隔
    oy        = 0,              # y轴起点（时间起点）
    dy        = 1,              # y轴采样间隔（dt）

    # ── 标签 ────────────────────────────────────────
    title     = " ",            # 标题
    titlesize = 16,             # 标题字号
    xlabel    = " ",            # x轴标签
    xunits    = " ",            # x轴单位
    ylabel    = " ",            # y轴标签
    yunits    = " ",            # y轴单位
    labelsize = 14,             # 轴标签字号

    # ── 刻度 ────────────────────────────────────────
    xticks       = None,        # 自定义刻度位置列表
    yticks       = None,
    xticklabels  = None,        # 自定义刻度标签列表
    yticklabels  = None,
    ticksize     = 11,          # 刻度字号

    # ── 波形道参数（style="wiggles"或"overlay"时有效）──
    wiggle_fill_color      = "k",  # 正值填充颜色
    wiggle_line_color      = "k",  # 波形线颜色
    wiggle_trace_increment = 1,    # 每隔几道画一道
    xcur      = 1.2,               # 波形摆动幅度系数
    scal      = None,              # 手动缩放比例

    # ── 图像显示 ─────────────────────────────────────
    aspect        = "auto",     # 纵横比
    interpolation = "bilinear", # 插值方式
    fmax          = 100,        # FK/Amplitude 最大频率(Hz)
    line_width    = 1.0,        # 波形线宽

    # ── 图形尺寸（独立模式下有效）────────────────────
    fignum = None,              # 指定图窗编号
    wbox   = 6,                 # 图宽（英寸）
    hbox   = 6,                 # 图高（英寸）
    dpi    = 100,               # 分辨率

    # ── 子图模式 ─────────────────────────────────────
    ax     = None,              # 传入外部 axes 对象（推荐多子图时使用）

    # ── 保存 ─────────────────────────────────────────
    name   = None,              # 保存路径，如 "output.png"
)

参数说明
--------
d : np.ndarray
    二维地震数据，shape = (nt, nx)
extent : Extent, 可选
    坐标与标签信息对象，传入后自动覆盖 ox/dx/oy/dy 及标签参数
plot_type : str
    绘图类型：
    - "TX"        时间-空间域剖面（默认）
    - "FK"        频率-波数域
    - "Amplitude" 平均振幅频谱
style : str
    显示风格（仅 TX 模式有效）：
    - "color"   彩色填充图（默认）
    - "wiggles" 纯波形道
    - "overlay" 彩色图 + 波形叠加
ax : matplotlib.axes.Axes, 可选
    外部传入的子图对象。多子图布局时必须传入，
    否则 SeisPlot 会自动创建新 figure 导致布局混乱。

返回
----
im : matplotlib 图像对象

示例
----
# 独立显示
SeisPlot(d, style="color", cmap="gray", dy=0.004, dx=5)

# 多子图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
SeisPlot(d, ax=axes[0], style="overlay", cmap="gray", dy=dt, dx=dx)
SeisPlot(m, ax=axes[1], style="color",   cmap="seismic", dy=dt, dx=dp)
plt.tight_layout()
plt.show()

# 保存
SeisPlot(d, cmap="RdBu", dy=0.004, dx=5, name="section.png", dpi=300)
"""


@dataclass
class Extent:
    title: str = " "
    label1: str = " "
    label2: str = " "
    unit1: str = " "
    unit2: str = " "
    o1: float = 0.0
    d1: float = 1.0
    o2: float = 0.0
    d2: float = 1.0


def SeisPlot(
    d: np.ndarray,
    extent: Optional[Extent] = None,
    plot_type: str = "TX",
    style: str = "color",
    cmap: str = "PuOr",
    pclip: float = 98,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    aspect: str = "auto",
    interpolation: str = "bilinear",
    fmax: float = 100,
    line_width: float = 1.0,
    wiggle_fill_color: str = "k",
    wiggle_line_color: str = "k",
    wiggle_trace_increment: int = 1,
    xcur: float = 1.2,
    scal: Optional[float] = None,
    title: str = " ",
    titlesize: int = 16,
    xlabel: str = " ",
    xunits: str = " ",
    ylabel: str = " ",
    yunits: str = " ",
    labelsize: int = 14,
    ox: float = 0,
    dx: float = 1,
    oy: float = 0,
    dy: float = 1,
    xticks: Optional[list] = None,
    yticks: Optional[list] = None,
    xticklabels: Optional[list] = None,
    yticklabels: Optional[list] = None,
    ticksize: int = 11,
    fignum: Optional[int] = None,
    wbox: float = 6,
    hbox: float = 6,
    dpi: int = 100,
    name: Optional[str] = None,
    ax=None,          # ← 新增：支持外部传入 ax
):
    # ------------------------------------------------------------------ #
    #  若传入 Extent，覆盖坐标和标签
    # ------------------------------------------------------------------ #
    if extent is not None:
        title  = extent.title
        xlabel = extent.label2
        xunits = f"({extent.unit2})"
        ylabel = extent.label1
        yunits = f"({extent.unit1})"
        ox, dx = extent.o2, extent.d2
        oy, dy = extent.o1, extent.d1

    # ------------------------------------------------------------------ #
    #  振幅裁剪
    # ------------------------------------------------------------------ #
    if vmin is None or vmax is None:
        if pclip <= 100:
            a = -np.percentile(np.abs(d.ravel()), pclip)
        else:
            a = -np.max(np.abs(d.ravel())) * pclip / 100.0
        b = -a
    else:
        a = vmin
        b = vmax

    # ------------------------------------------------------------------ #
    #  创建图窗 —— 仅在没有外部 ax 时才创建新 figure
    # ------------------------------------------------------------------ #
    if ax is None:
        # 独立调用模式：自己创建 figure
        if fignum is None:
            fig = plt.figure(figsize=(wbox, hbox), dpi=dpi,
                             facecolor="w", edgecolor="k")
        else:
            fig = plt.figure(num=fignum, figsize=(wbox, hbox), dpi=dpi,
                             facecolor="w", edgecolor="k")
        ax = fig.add_subplot(111)
    else:
        # 外部 subplot 模式：直接使用传入的 ax，不创建 figure
        fig = ax.get_figure()

    im = None

    # ================================================================== #
    #  TX 模式
    # ================================================================== #
    if plot_type == "TX":
        nt, nx_size = d.shape

        if style != "wiggles":
            im = ax.imshow(
                d,
                cmap=cmap,
                vmin=a,
                vmax=b,
                extent=[
                    ox - dx / 2,
                    ox + (nx_size - 1) * dx + dx / 2,
                    oy + (nt - 1) * dy,
                    oy,
                ],
                aspect=aspect,
                interpolation=interpolation,
                origin="upper",
            )

        if style != "color":
            margin = dx if style == "wiggles" else dx / 2
            t_vec = oy + dy * np.arange(nt)
            x_vec = ox + dx * np.arange(nx_size)
            delta = wiggle_trace_increment * dx
            alpha = xcur * delta

            if scal is None:
                dmax = np.max(np.abs(d))
                if dmax > 0:
                    alpha = alpha / dmax
            else:
                alpha = alpha * scal

            for k in range(0, nx_size, wiggle_trace_increment):
                sc = x_vec[k]
                s  = d[:, k] * alpha + sc
                ax.plot(s, t_vec,
                        color=wiggle_line_color,
                        linewidth=line_width)
                if style != "overlay":
                    ax.fill_betweenx(
                        t_vec, sc, s,
                        where=(s >= sc),
                        facecolor=wiggle_fill_color,
                    )

            ax.set_xlim(ox - margin, ox + (nx_size - 1) * dx + margin)
            ax.set_ylim(oy + (nt - 1) * dy, oy)

    # ================================================================== #
    #  FK 模式
    # ================================================================== #
    elif plot_type == "FK":
        xlabel = "Wavenumber"
        xunits = "(1/m)"
        ylabel = "Frequency"
        yunits = "(Hz)"

        nt, nx_size = d.shape
        dk   = 1.0 / (dx * nx_size)
        kmin = -dk * nx_size / 2.0
        kmax =  dk * nx_size / 2.0
        df   = 1.0 / (dy * nt)
        FMAX = df * nt / 2.0
        if fmax > FMAX:
            fmax = FMAX

        nf = int(np.floor((nt / 2) * fmax / FMAX))
        D  = np.abs(fftshift(fft(fft(d, axis=0), axis=1)))
        half = D.shape[0] // 2
        D  = D[half: half + nf, :]

        if vmin is None or vmax is None:
            a = 0.0
            b = (np.percentile(np.abs(D.ravel()), pclip)
                 if pclip <= 100
                 else np.max(np.abs(D.ravel())) * pclip / 100.0)

        im = ax.imshow(
            D, cmap=cmap, vmin=a, vmax=b,
            extent=[kmin, kmax, fmax, 0],
            aspect=aspect, interpolation=interpolation, origin="upper",
        )

    # ================================================================== #
    #  Amplitude 模式
    # ================================================================== #
    elif plot_type == "Amplitude":
        xlabel = "Frequency"
        xunits = "(Hz)"
        ylabel = "Amplitude"
        yunits = ""

        nt, nx_size = d.shape
        df   = 1.0 / (dy * nt)
        FMAX = df * nt / 2.0
        if fmax > FMAX:
            fmax = FMAX

        nf    = int(np.floor((nt / 2) * fmax / FMAX))
        spec  = fftshift(np.sum(np.abs(fft(d, axis=0)), axis=1)) / nx_size
        half  = len(spec) // 2
        y_amp = spec[half: half + nf]

        norm = np.max(y_amp)
        if norm > 0:
            y_amp = y_amp / norm

        x_freq = np.linspace(0, fmax, len(y_amp))
        im = ax.plot(x_freq, y_amp, color="k", linewidth=line_width)
        ax.set_xlim(0, fmax)
        ax.set_ylim(0, 1.1)

    else:
        raise ValueError(f"plot_type='{plot_type}' 不支持，"
                         "请使用 'TX'、'FK' 或 'Amplitude'。")

    # ------------------------------------------------------------------ #
    #  公共标签与刻度
    # ------------------------------------------------------------------ #
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(f"{xlabel} {xunits}", fontsize=labelsize)
    ax.set_ylabel(f"{ylabel} {yunits}", fontsize=labelsize)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=ticksize)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontsize=ticksize)

    ax.tick_params(axis="both", labelsize=ticksize)

    # ------------------------------------------------------------------ #
    #  保存或显示（仅独立模式下触发）
    # ------------------------------------------------------------------ #
    if ax is not None and name is not None:
        fig.savefig(name, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return im