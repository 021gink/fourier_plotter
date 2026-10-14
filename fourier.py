"""
fourier.py —— 闭合轮廓的复傅里叶级数分解与重建

    z(t) = Σ_{k=-N}^{N} c_k · e^{i2πkt},     t ∈ [0, 1)
    c_k  = ∫_0^1 z(t) · e^{-i2πkt} dt

几何意义（“本轮 + 均轮”）：
    每一项 c_k e^{i2πkt} 是一个半径 |c_k|、初相 arg(c_k)、每周期转 k 圈的旋转向量。
    k > 0 逆时针转，k < 0 顺时针转，c_0 是轮廓的质心（平移）。
    把所有向量首尾相接，链条末端画出的轨迹就是原轮廓。
"""

from __future__ import annotations

import numpy as np


def fourier_coefficients(z: np.ndarray, harmonics: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """计算 k = -K..K 的复系数。

    离散化：c_k ≈ (1/N) Σ_n z_n e^{-i2πkn/N}，这正是 DFT 除以 N。
    对周期、等距采样的光滑函数，这个矩形求和是谱精度的积分公式。
    DFT 的第 (k mod N) 个分量对应频率 k（负频率在数组末尾）。
    要求 2K+1 ≤ N，否则高阶系数会与低阶混叠（Nyquist 限制）。
    """
    n = len(z)
    if 2 * harmonics + 1 > n:
        raise ValueError(f"阶数 K={harmonics} 过高：需要 2K+1 ≤ 采样点数 {n}")
    spectrum = np.fft.fft(z) / n
    ks = np.arange(-harmonics, harmonics + 1)
    return ks, spectrum[ks % n]


def reconstruct(ks: np.ndarray, c: np.ndarray, n_points: int = 1024) -> np.ndarray:
    """按级数求和：z(t_j) = Σ_k c_k e^{i2πk t_j}，t_j = j / n_points。
    用矩阵乘法一次性计算（外积 t ⊗ k 构成所有相位）。"""
    t = np.arange(n_points) / n_points
    return np.exp(2j * np.pi * np.outer(t, ks)) @ c


def epicycle_order(ks: np.ndarray, c: np.ndarray) -> np.ndarray:
    """本轮绘制顺序：c_0（固定中心）放最前，其余按半径 |c_k| 从大到小。
    大圆先画 → 链条前几节就勾勒出粗略形状，后面的小圆逐步补充细节。"""
    rest = [i for i in np.argsort(-np.abs(c)) if ks[i] != 0]
    dc = [int(np.where(ks == 0)[0][0])] if 0 in ks else []
    return np.array(dc + rest, dtype=int)


def partial_sum(ks: np.ndarray, c: np.ndarray, n_terms: int, n_points: int = 1024) -> np.ndarray:
    """只保留 c_0 和最大的 n_terms 个谐波求和——展示“从粗到精”的收敛过程。
    按幅值截断是 L2 意义下的最优 n 项逼近（Parseval：误差能量 = 被丢弃项 |c_k|² 之和）。"""
    order = epicycle_order(ks, c)
    keep = order[: n_terms + (1 if 0 in ks else 0)]
    return reconstruct(ks[keep], c[keep], n_points)


def dominant_harmonics(ks: np.ndarray, c: np.ndarray, top: int = 6, rel: float = 0.01) -> list[tuple[int, float]]:
    """返回幅值最大的若干非零阶谐波 [(k, |c_k|), ...]。
    c_0 只代表位置，不计入；幅值低于最大值 rel 倍的视为数值/像素噪声，不计入。"""
    mask = ks != 0
    idx = np.where(mask)[0][np.argsort(-np.abs(c[mask]))][:top]
    peak = np.abs(c[idx[0]]) if len(idx) else 0.0
    return [(int(ks[i]), float(np.abs(c[i]))) for i in idx if np.abs(c[i]) >= rel * peak]


def rotational_symmetry(ks: np.ndarray, c: np.ndarray, max_order: int = 12, threshold: float = 0.95):
    """从系数检测 p 重旋转对称。

    若逆时针轮廓绕质心有 p 重对称：z(t + 1/p) = e^{i2π/p} z(t)，
    代入级数得  c_k e^{i2πk/p} = e^{i2π/p} c_k  ⇒  只有 k ≡ 1 (mod p) 的系数非零。
    因此统计 k ≡ 1 (mod p) 上的能量占比，取占比超过阈值的最大 p。
    例：五角星 p=5 → 非零阶为 1, -4, 6, -9, 11, ...（相邻峰间隔 5）。
    若几乎全部能量在 k=1，就是圆（任意阶对称），返回 'inf'。
    """
    mask = ks != 0
    e = np.abs(c[mask]) ** 2
    kk = ks[mask]
    total = e.sum()
    if total == 0:
        return None, 0.0
    if e[kk == 1].sum() / total > 0.995:
        return "inf", float(e[kk == 1].sum() / total)
    best = (1, 1.0)
    for p in range(2, max_order + 1):
        frac = e[(kk - 1) % p == 0].sum() / total
        if frac >= threshold:
            best = (p, float(frac))
    return best


def high_frequency_ratio(ks: np.ndarray, c: np.ndarray, cutoff_frac: float = 0.25) -> float:
    """高频能量占比：|k| > cutoff·K 的能量 / 非零阶总能量。
    尖角、细线、毛刺越多，系数衰减越慢（光滑曲线 |c_k| 衰减快于任意多项式，
    有尖角的折线约按 1/k² 衰减，有跳变则约 1/k），这个比例越高 → 听感越“亮/刺”。"""
    K = int(np.abs(ks).max())
    mask = ks != 0
    e = np.abs(c) ** 2
    hi = e[np.abs(ks) > cutoff_frac * K].sum()
    return float(hi / max(e[mask].sum(), 1e-30))


def coefficient_error(orig: list[np.ndarray], recon: list[np.ndarray]) -> tuple[float, float]:
    """系数相对误差 ||ĉ - g·c|| / ||g·c||，g 为最小二乘拟合的全局实数增益。

    音频做了峰值归一化，解码出的系数整体差一个正实数倍 g；
    先用 g = Re<c, ĉ> / ||c||² 消掉这个已知的不可恢复量，再看剩余误差。
    """
    a = np.concatenate(orig)
    b = np.concatenate(recon)
    g = float(np.real(np.vdot(a, b)) / max(np.vdot(a, a).real, 1e-30))
    if abs(g) < 1e-30:
        return float("inf"), g
    return float(np.linalg.norm(b - g * a) / np.linalg.norm(g * a)), g
