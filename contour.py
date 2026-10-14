"""
contour.py —— 轮廓提取、预处理与几何误差度量

数学约定
--------
平面上的点 (x, y) 统一用复数 z = x + i·y 表示，一条闭合轮廓就是一个
周期复函数 z(t), t ∈ [0, 1)。离散化后为 N 个等弧长采样点 z_n = z(n/N)。

图像坐标 y 轴向下，这里翻转为数学坐标（y 轴向上），所以
    z = (x - W/2)/S  -  i·(y - H/2)/S,   S = max(W, H)/2
使整张图落在 [-1, 1] 的单位方框内，且多轮廓之间的相对位置与尺寸被保留。
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# 1. 图像 → 二值掩膜
# ---------------------------------------------------------------------------
def load_binary(path: str, blur_sigma: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """读取图片并二值化，返回 (灰度图, 前景掩膜)。

    - 高斯模糊：低通滤波，抑制像素级噪声（照片中的麦秆纹理等）。
    - Otsu 阈值：最大化前景/背景类间方差，自动选择阈值。
    - 极性判定：图像边框上占多数的像素视为背景，保证前景（图案）为白色 255，
      这样无论“黑底白图”还是“白底黑图”都能正确提取。
    - 形态学开闭运算：去掉孤立噪点、填补细小缝隙。
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    g = cv2.GaussianBlur(img, (0, 0), blur_sigma) if blur_sigma > 0 else img
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = np.concatenate([bw[0], bw[-1], bw[:, 0], bw[:, -1]])
    if border.mean() > 127:
        bw = 255 - bw

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    return img, bw


# ---------------------------------------------------------------------------
# 2. 几何工具
# ---------------------------------------------------------------------------
def signed_area(z: np.ndarray) -> float:
    """鞋带公式（Green 定理离散形式）求有向面积：
        A = 1/2 · Σ (x_n y_{n+1} - x_{n+1} y_n) = 1/2 · Im Σ conj(z_n)·z_{n+1}
    A > 0 表示逆时针（数学正向）。"""
    return 0.5 * float(np.imag(np.sum(np.conj(z) * np.roll(z, -1))))


def resample_arclength(z: np.ndarray, n_points: int) -> np.ndarray:
    """把闭合折线重采样为 n_points 个等弧长点。

    傅里叶系数 c_k = ∫ z(t) e^{-i2πkt} dt 中的参数 t 取“归一化弧长”，
    等弧长采样后 t_n = n/N 均匀分布，矩形求和（对周期函数即梯形法则）
    具有谱精度，数值积分稳定；否则像素链码的疏密不均会混入虚假高频。
    """
    zc = np.append(z, z[0])                        # 闭合：首尾相连
    seg = np.abs(np.diff(zc))                      # 每段弦长
    s = np.concatenate([[0.0], np.cumsum(seg)])    # 累积弧长 s(n)
    total = s[-1]
    su = np.arange(n_points) * total / n_points    # 目标弧长位置（不含终点，保持周期）
    re = np.interp(su, s, zc.real)
    im = np.interp(su, s, zc.imag)
    return re + 1j * im


def _dedupe(z: np.ndarray) -> np.ndarray:
    """去掉相邻重复点（零长度段会让弧长参数化退化）。"""
    keep = np.abs(np.diff(np.append(z, z[0]))) > 1e-12
    return z[keep]


def roll_to_topmost(z: np.ndarray) -> np.ndarray:
    """把轮廓起点平移到最上方的点（Im z 最大）。

    起点平移 t → t + τ 会让 c_k 乘上 e^{i2πkτ}：只改变相位、不改变形状。
    统一起点后，原始与重建轮廓的系数相位才可以直接比较。
    若有多个并列最高点（如像素化的圆顶），取其中最靠左的一个。
    """
    top = z.imag.max()
    cand = np.where(z.imag >= top - 1e-9)[0]
    idx = cand[np.argmin(z.real[cand])]
    return np.roll(z, -idx)


def preprocess_contour(z: np.ndarray, n_points: int = 512, smooth_sigma: float = 1.0) -> np.ndarray:
    """单条轮廓的完整预处理流水线：
    去重 → 统一为逆时针 → 起点归到最上方 → 等弧长重采样 → 周期高斯平滑 → 再次对齐起点。

    统一逆时针的意义：方向反转等价于 t → -t，会把 c_k 与 c_{-k} 互换。
    固定方向后，“正 k 阶 = 与轮廓同向旋转的圆”这一解释才有一致性。
    """
    z = _dedupe(np.asarray(z, dtype=complex))
    if signed_area(z) < 0:
        z = z[::-1]
    z = roll_to_topmost(z)
    z = resample_arclength(z, n_points)
    if smooth_sigma > 0:
        # mode="wrap"：轮廓是周期信号，首尾需循环卷积，否则接缝处会被拉扯
        z = gaussian_filter1d(z.real, smooth_sigma, mode="wrap") + 1j * gaussian_filter1d(
            z.imag, smooth_sigma, mode="wrap"
        )
    return roll_to_topmost(z)


# ---------------------------------------------------------------------------
# 3. 图像 → 多条预处理后的轮廓
# ---------------------------------------------------------------------------
def extract_contours(
    path: str,
    n_points: int = 512,
    smooth_sigma: float = 1.0,
    min_area_frac: float = 5e-4,
    max_contours: int = 8,
) -> dict:
    """用 OpenCV findContours 提取所有闭合轮廓，每条独立预处理。

    RETR_LIST：返回全部轮廓（外边界 + 孔洞边界），同心圆环会得到内外两条，
    麦田怪圈这类“圆环 + 直线 + 卫星圆”组合图案的每个部件都会被保留。
    按面积从大到小排序，最多保留 max_contours 条（音频中依次编码为时间段）。
    """
    img, bw = load_binary(path)
    h, w = bw.shape
    found, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    min_area = min_area_frac * h * w
    raw = [c[:, 0, :].astype(float) for c in found if len(c) >= 8 and cv2.contourArea(c) >= min_area]
    raw.sort(key=lambda c: -cv2.contourArea(c.astype(np.float32)))
    raw = raw[:max_contours]
    if not raw:
        raise ValueError("未找到足够大的闭合轮廓，请检查图片对比度或调低 min_area_frac")

    scale = max(h, w) / 2.0
    contours = []
    for c in raw:
        z = (c[:, 0] - w / 2.0) / scale - 1j * (c[:, 1] - h / 2.0) / scale
        contours.append(preprocess_contour(z, n_points, smooth_sigma))
    return {"image": img, "mask": bw, "contours": contours, "shape": (h, w)}


# ---------------------------------------------------------------------------
# 4. 闭环误差度量
# ---------------------------------------------------------------------------
def _joint_normalize(zs: list[np.ndarray]) -> list[np.ndarray]:
    """对整组轮廓做联合相似归一化：减去整体质心、除以整体 RMS 半径。

    音频编码时做了峰值归一化（防削波），绝对尺度本身就无法从音频中恢复，
    所以比较形状前先消除平移与全局缩放；多轮廓共用同一组参数，
    因此轮廓之间的相对位置与相对大小仍然参与误差计算。
    """
    pts = np.concatenate(zs)
    mu = pts.mean()
    r = np.sqrt(np.mean(np.abs(pts - mu) ** 2)) or 1.0
    return [(z - mu) / r for z in zs]


def compare_contour_sets(orig: list[np.ndarray], recon: list[np.ndarray], dense: int = 4096) -> dict:
    """比较两组轮廓，返回（单位：图案 RMS 半径）
    - hausdorff: 对称 Hausdorff 距离  max(sup_a d(a,B), sup_b d(b,A))，最坏偏差
    - mean_hausdorff: 平均 Hausdorff（双向平均最近距离），整体偏差
    - area_diff: 归一化面积差  |A_orig - A_rec| / A_orig（面积取有向面积绝对值）
    多轮廓取逐条平均。
    """
    a_list = _joint_normalize([resample_arclength(z, dense) for z in orig])
    b_list = _joint_normalize([resample_arclength(z, dense) for z in recon])
    hd, mhd, ad = [], [], []
    for a, b in zip(a_list, b_list):
        pa = np.c_[a.real, a.imag]
        pb = np.c_[b.real, b.imag]
        d_ab = cKDTree(pb).query(pa)[0]
        d_ba = cKDTree(pa).query(pb)[0]
        hd.append(max(d_ab.max(), d_ba.max()))
        mhd.append(0.5 * (d_ab.mean() + d_ba.mean()))
        area_a = abs(signed_area(a))
        ad.append(abs(area_a - abs(signed_area(b))) / max(area_a, 1e-12))
    return {
        "hausdorff": float(np.mean(hd)),
        "mean_hausdorff": float(np.mean(mhd)),
        "area_diff": float(np.mean(ad)),
    }
