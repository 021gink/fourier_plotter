"""
visualize.py —— 傅里叶圆动画、原图/重建对比图、频谱图

配色取自“绘图仪 + 坐标纸”：冷灰纸面、藏青墨线、橙色标注笔。
中文字体：自动在系统中查找常见 CJK 字体（也可用环境变量 FOURIER_CJK_FONT
指定 .ttf/.otf 路径）；找不到时图中文字自动退回英文，避免出现方块字。
"""

from __future__ import annotations

import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation, font_manager  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from fourier import dominant_harmonics, epicycle_order, partial_sum, reconstruct  # noqa: E402

PAPER = "#F3F5F4"
INK = "#1B2A4A"
GRID = "#D5DBDF"
ACCENT = "#E4572E"
MUTED = "#6B7785"


def _setup_fonts() -> bool:
    path = os.environ.get("FOURIER_CJK_FONT")
    names = []
    if path and os.path.exists(path):
        files = ([os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith((".ttf", ".otf"))]
                 if os.path.isdir(path) else [path])
        for fp in files:
            font_manager.fontManager.addfont(fp)
            name = font_manager.FontProperties(fname=fp).get_name()
            if name not in names:
                names.append(name)
    candidates = ["Noto Sans SC", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
                  "Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Heiti SC"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    names += [n for n in candidates if n in installed]
    plt.rcParams.update({
        "font.sans-serif": names + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "font.size": 9,
    })
    return bool(names)


HAS_CJK = _setup_fonts()


def _t(zh: str, en: str) -> str:
    return zh if HAS_CJK else en


def _shape_axes(ax, title: str | None = None):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title)


def _draw_closed(ax, z, **kw):
    zc = np.append(z, z[0])
    ax.plot(zc.real, zc.imag, **kw)


def _limits(curves, pad=0.12):
    pts = np.concatenate(curves)
    lo = min(pts.real.min(), pts.imag.min())
    hi = max(pts.real.max(), pts.imag.max())
    cx, cy = (pts.real.max() + pts.real.min()) / 2, (pts.imag.max() + pts.imag.min()) / 2
    half = max(pts.real.max() - pts.real.min(), pts.imag.max() - pts.imag.min()) / 2
    half = half * (1 + pad) if half > 0 else max(abs(lo), abs(hi), 1.0)
    return (cx - half, cx + half), (cy - half, cy + half)


# ---------------------------------------------------------------------------
# 系数 → 图案图片（反向转换的输出）
# ---------------------------------------------------------------------------
def plot_pattern(path: str, coeff_sets, title: str | None = None, n_points: int = 1500):
    """按 z(t) = Σ c_k e^{i2πkt} 求和，以 (Re z, Im z) 画出每条轨迹。"""
    curves = [reconstruct(ks, c, n_points) for ks, c in coeff_sets]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    for z in curves:
        _draw_closed(ax, z, color=INK, lw=1.6)
    xl, yl = _limits(curves)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    _shape_axes(ax, title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 静态对比图：原图 | 提取的轮廓 | 各解码方案的重建
# ---------------------------------------------------------------------------
def plot_comparison(path: str, image, original, recons, title: str = ""):
    """recons: [(标签, 轮廓列表, 指标 dict 或 None), ...]"""
    n = 2 + len(recons)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.6), dpi=120)
    axes[0].imshow(image, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title(_t("原图", "Original"))

    xl, yl = _limits(original)
    for z in original:
        _draw_closed(axes[1], z, color=INK, lw=1.4)
    axes[1].set_xlim(xl)
    axes[1].set_ylim(yl)
    _shape_axes(axes[1], _t(f"提取轮廓 ×{len(original)}", f"Contours x{len(original)}"))

    for ax, (label, curves, metrics) in zip(axes[2:], recons):
        cxl, cyl = _limits(curves + original)
        for z in original:
            _draw_closed(ax, z, color=GRID, lw=3.0)
        lossy = metrics is not None and metrics["mean_hausdorff"] > 0.02
        for z in curves:
            _draw_closed(ax, z, color=ACCENT if lossy else INK, lw=1.3)
        ax.set_xlim(cxl)
        ax.set_ylim(cyl)
        _shape_axes(ax, label)
        if metrics is not None:
            ax.text(0.5, -0.04,
                    f"Hausdorff {metrics['hausdorff']:.3f} | "
                    f"{_t('面积差', 'ΔA')} {metrics['area_diff'] * 100:.1f}%",
                    transform=ax.transAxes, ha="center", va="top", color=MUTED, fontsize=8)
    if title:
        fig.suptitle(title, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 频谱图：系数幅值柱状图 + 音频实际频谱
# ---------------------------------------------------------------------------
def plot_spectrum(path: str, ks, c, audio, fs: int, f0: float, title: str = "", top: int = 6):
    dom = dominant_harmonics(ks, c, top)
    dom_k = {k for k, _ in dom}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.2), dpi=120)

    colors = [ACCENT if k in dom_k else (MUTED if k == 0 else INK) for k in ks]
    ax1.bar(ks, np.abs(c), width=0.8, color=colors)
    for k, a in dom:
        ax1.annotate(f"k={k:+d}\n{f0 * (abs(k) + 1):.0f} Hz", (k, a), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7, color=ACCENT)
    ax1.set_xlabel(_t("谐波阶数 k", "harmonic k"))
    ax1.set_ylabel("|c_k|")
    ax1.set_title(_t("轮廓谐波分解（橙色 = 主导谐波，灰色 = c0 位置项）",
                     "Contour harmonics (orange = dominant, gray = c0)"))
    ax1.set_ylim(0, np.abs(c[ks != 0]).max() * 1.35 if np.any(ks != 0) else 1)
    ax1.grid(axis="y", color=GRID, lw=0.6)
    ax1.set_axisbelow(True)

    # 音频频谱：对左声道做 Hann 窗 FFT，画 dB 幅度；谐波应恰好落在 f0 的整数倍上
    x = audio[:, 0]
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    db = 20 * np.log10(spec / (spec.max() + 1e-30) + 1e-9)
    K = int(np.abs(ks).max())
    fmax = f0 * (K + 2)
    m = freqs <= fmax
    ax2.plot(freqs[m], db[m], color=INK, lw=0.7)
    for k, _ in dom:
        ax2.axvline(f0 * (abs(k) + 1), color=ACCENT, lw=0.8, ls="--")
    ax2.set_ylim(-100, 5)
    ax2.set_xlim(0, fmax)
    ax2.set_xlabel(_t("频率 (Hz)，谐波 k 位于 f0·(|k|+1)", "frequency (Hz), harmonic k at f0(|k|+1)"))
    ax2.set_ylabel("dB")
    ax2.set_title(_t("合成音频频谱（左声道）", "Synthesized audio spectrum (left)"))
    ax2.grid(color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 傅里叶圆动画
# ---------------------------------------------------------------------------
def animate_epicycles(path: str, coeff_sets, frames: int = 160, fps: int = 25,
                      max_circles: int = 50, title: str = ""):
    """左：本轮（圆 + 连杆）实时绘制；右：保留前 N 个最大谐波的部分和，N 随时间增长。

    左图：链条第 j 节是向量 c_{k_j} e^{i2πk_j t}，以上一节末端为圆心、|c_{k_j}| 为半径旋转，
          末端走过 t: 0 → 1 即画出完整轮廓。
    右图：Parseval 定理保证部分和的 L2 误差单调下降，可以看到由椭圆逐步“长出”尖角与细节。
    """
    t_f = np.arange(frames) / frames
    unit = np.exp(2j * np.pi * np.linspace(0, 1, 48))
    chains, ghosts = [], []
    for ks, c in coeff_sets:
        sel = epicycle_order(ks, c)[: max_circles + 1]
        kk, cc = ks[sel], c[sel]
        terms = cc[None, :] * np.exp(2j * np.pi * np.outer(t_f, kk))
        chains.append((np.cumsum(terms, axis=1), np.abs(cc)))
        ghosts.append(reconstruct(ks, c, 800))

    pts = [p.ravel() for p, _ in chains] + ghosts
    xl, yl = _limits(pts, pad=0.05)

    total_terms = max(int(np.sum(np.abs(c) > 1e-9)) - 1 for _, c in coeff_sets)
    schedule = np.unique(np.round(np.geomspace(1, max(total_terms, 1), frames)).astype(int))
    schedule = schedule[np.linspace(0, len(schedule) - 1, frames).astype(int)]
    partial_cache = {n: [partial_sum(ks, c, n, 500) for ks, c in coeff_sets] for n in np.unique(schedule)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.2), dpi=80)
    for ax in (ax1, ax2):
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        for g in ghosts:
            _draw_closed(ax, g, color=GRID, lw=2.5)
    _shape_axes(ax1, _t("傅里叶本轮绘制", "Epicycles"))
    _shape_axes(ax2, _t("部分和收敛", "Partial-sum convergence"))

    circles = LineCollection([], colors=MUTED, linewidths=0.5, alpha=0.6)
    ax1.add_collection(circles)
    arms = [ax1.plot([], [], color=INK, lw=0.9)[0] for _ in chains]
    traces = [ax1.plot([], [], color=ACCENT, lw=1.8)[0] for _ in chains]
    partials = [ax2.plot([], [], color=INK, lw=1.4)[0] for _ in chains]
    label = ax2.text(0.02, 0.02, "", transform=ax2.transAxes, color=ACCENT, fontsize=11, fontweight="bold")
    if title:
        fig.suptitle(title, fontweight="bold")

    def update(f):
        segs = []
        for (P, radii), arm, trace in zip(chains, arms, traces):
            p = P[f]
            for j in range(1, len(p)):
                if radii[j] > 1e-4:
                    ring = p[j - 1] + radii[j] * unit
                    segs.append(np.c_[ring.real, ring.imag])
            arm.set_data(p.real, p.imag)
            tip = P[: f + 1, -1]
            trace.set_data(tip.real, tip.imag)
        circles.set_segments(segs)
        n = int(schedule[f])
        for line, z in zip(partials, partial_cache[n]):
            zc = np.append(z, z[0])
            line.set_data(zc.real, zc.imag)
        label.set_text(_t(f"前 {n} 个谐波", f"top {n} harmonics"))
        return [circles, label, *arms, *traces, *partials]

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    if path.endswith(".mp4") and shutil.which("ffmpeg"):
        anim.save(path, writer=animation.FFMpegWriter(fps=fps))
    else:
        if path.endswith(".mp4"):
            path = path[:-4] + ".gif"
        anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path
