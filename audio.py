"""
audio.py —— 傅里叶系数 ⇄ 立体声音频

映射规则（每个谐波 c_k 对应一个正弦振荡器）
    频率 f_k = f0 · (|k| + 1)     （c_0 → f0，c_{±1} → 2f0，……）
    幅值 = |c_k|，相位 φ_k = arg(c_k)

两种立体声编码
--------------
mode="spec"（题目原始规则）
    L += |c_k| cos(ω_k t + φ_k)
    R += |c_k| cos(ω_k t - φ_k)
  用相量（phasor）表示：信号 Re(P·e^{iωt}) 的相量为 P。
  则同一频率 f0(m+1) 上，L 的相量 = c_m + c_{-m}，R 的相量 = conj(c_m + c_{-m})。
  R 恰好是 L 的复共轭——不携带任何新信息！只能恢复和 s_m = c_m + c_{-m}，
  c_m 与 c_{-m} 无法分离（详见报告）。

mode="quadrature"（正交立体声，可逆）
    L += |c_k| cos(ω_k t + φ_k)
    R += sgn(k) · |c_k| sin(ω_k t + φ_k)        （sgn(0) 取 +1）
  每个振荡器的频率、幅值、相位仍完全按题目规则；只是右声道改用正交(sin)分量，
  并让 ±k 两个旋转方向的符号相反：
    L 相量 = c_m + c_{-m}
    R 相量 = -i·c_m + i·c_{-m}
  ⇒ c_m = (P_L + i·P_R)/2,  c_{-m} = (P_L - i·P_R)/2     —— 两个未知数两个方程，严格可逆。
  （这与“示波器音乐”同理：L/R 接示波器 X/Y 时，正 k 画逆时针、负 k 画顺时针。）

多轮廓：每条轮廓占一个时间段，段间插入静音间隔，解码时按静音自动切分。
所有振荡器共用文件级全局时间轴 t（从文件开头算起），因此切分边界即使有几个
采样点的误差，也不会引入相位误差。
"""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile

FS = 44100           # 采样率
GAP_S = 0.04         # 段间静音（秒）
FADE_S = 0.01        # 每段首尾淡入淡出（秒），避免爆音
PEAK = 0.9           # 归一化峰值（满幅的 90%，防止削波）
MODES = ("spec", "quadrature")


def harmonic_frequency(k: np.ndarray | int, f0: float) -> np.ndarray:
    """f_k = f0 · (|k| + 1)。"""
    return f0 * (np.abs(k) + 1)


def check_nyquist(harmonics: int, f0: float, fs: int = FS) -> None:
    """最高频率 f0·(K+1) 必须低于 Nyquist 频率 fs/2，否则会混叠成错误的低频。"""
    fmax = f0 * (harmonics + 1)
    if fmax >= 0.98 * fs / 2:
        raise ValueError(
            f"最高谐波频率 {fmax:.0f} Hz 超过 Nyquist ({fs / 2:.0f} Hz)，"
            f"请降低 --harmonics 或 --base-freq（需 f0·(K+1) < fs/2）"
        )


def _oscillator_bank(ks, c, f0, t, mode):
    """把一组系数合成为 (L, R) 两路浮点信号。"""
    left = np.zeros_like(t)
    right = np.zeros_like(t)
    for k, ck in zip(ks, c):
        amp, phi = np.abs(ck), np.angle(ck)
        if amp < 1e-12:
            continue
        theta = 2 * np.pi * harmonic_frequency(k, f0) * t
        left += amp * np.cos(theta + phi)
        if mode == "spec":
            right += amp * np.cos(theta - phi)
        else:
            right += (1.0 if k >= 0 else -1.0) * amp * np.sin(theta + phi)
    return left, right


def encode(coeff_sets, f0: float = 220.0, duration: float = 2.5, mode: str = "quadrature", fs: int = FS):
    """系数组列表 [(ks, c), ...] → int16 立体声 PCM，返回 (pcm, gain)。

    gain 为峰值归一化系数：pcm ≈ gain · Σ 振荡器。它不写入音频，
    因此从音频只能恢复“差一个全局倍数”的系数（形状不变）。
    """
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES}")
    K = int(max(np.abs(ks).max() for ks, _ in coeff_sets))
    check_nyquist(K, f0, fs)

    n = len(coeff_sets)
    gap = int(round(GAP_S * fs))
    seg = (int(round(duration * fs)) - gap * (n - 1)) // n
    if seg < int(0.15 * fs):
        raise ValueError("轮廓过多导致每段过短，请增大 --duration 或减少轮廓数")
    total = n * seg + gap * (n - 1)
    t = np.arange(total) / fs                    # 全局时间轴

    fade = int(FADE_S * fs)
    env = np.ones(seg)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)   # 升余弦淡入
    env[:fade], env[-fade:] = ramp, ramp[::-1]

    out = np.zeros((total, 2))
    for i, (ks, c) in enumerate(coeff_sets):
        s0 = i * (seg + gap)
        left, right = _oscillator_bank(ks, c, f0, t[s0 : s0 + seg], mode)
        out[s0 : s0 + seg, 0] = left * env
        out[s0 : s0 + seg, 1] = right * env

    peak = np.abs(out).max()
    gain = PEAK / peak if peak > 0 else 1.0
    pcm = np.round(out * gain * 32767).astype(np.int16)   # 16-bit 量化
    return pcm, gain


def write_wav(path: str, pcm: np.ndarray, fs: int = FS) -> None:
    wavfile.write(path, fs, pcm)


def read_wav(path: str) -> tuple[int, np.ndarray]:
    """读取 WAV，统一转为 [-1, 1] 浮点、形状 (n, 2)。单声道复制为双声道。"""
    fs, data = wavfile.read(path)
    return fs, to_float_stereo(data)


def to_float_stereo(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / float(np.iinfo(data.dtype).max + 1)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = np.c_[data, data]
    return data[:, :2]


def split_segments(x: np.ndarray, fs: int, min_gap_s: float = GAP_S / 2, min_len_s: float = 0.05):
    """按静音切分时间段，返回 [(start, end), ...]。
    静音判定：两声道绝对值都 < 2 个 16-bit 量化级，持续 ≥ min_gap_s。
    没有静音（如外部音频）时整段视为一条轮廓。"""
    active = np.max(np.abs(x), axis=1) > 1.5 / 32768
    idx = np.where(active)[0]
    if len(idx) == 0:
        return []
    min_gap = int(min_gap_s * fs)
    breaks = np.where(np.diff(idx) > min_gap)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks] + 1, idx[-1] + 1]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= int(min_len_s * fs)]


def _extract_phasors(x, t, K, f0, max_window_s, fs):
    """窄带提取：在已知频率 f_m = f0(m+1), m=0..K 上做最小二乘拟合。

    模型  x(t) = Σ_m  a_m cos(ω_m t) - b_m sin(ω_m t)  =  Σ_m Re(P_m e^{iω_m t}),
    P_m = a_m + i b_m 即该频率的复相量。所有 ω_m 都是 2πf0 的整数倍，
    分析窗口覆盖数十个基频周期时设计矩阵接近正交，等价于逐频 DFT 投影，
    但最小二乘对任意基频/窗口长度都没有频谱泄漏误差。
    """
    n = len(t)
    trim = int(0.15 * n)                       # 去掉首尾淡入淡出部分，只用稳态段
    lo, hi = trim, n - trim
    max_n = int(max_window_s * fs)
    if hi - lo > max_n:
        mid = (lo + hi) // 2
        lo, hi = mid - max_n // 2, mid + max_n // 2
    tt, xx = t[lo:hi], x[lo:hi]
    w = 2 * np.pi * f0 * (np.arange(K + 1) + 1)
    ph = np.outer(tt, w)
    A = np.hstack([np.cos(ph), -np.sin(ph)])
    sol, *_ = np.linalg.lstsq(A, xx, rcond=None)
    P = sol[: K + 1] + 1j * sol[K + 1 :]       # 形状 (K+1, 2)：两声道的相量
    return P[:, 0], P[:, 1]


def decode(
    audio,
    harmonics: int = 64,
    f0: float = 220.0,
    mode: str = "quadrature",
    fs: int = FS,
    spec_strategy: str = "symmetric",
    max_window_s: float = 1.0,
):
    """音频（WAV 路径或 numpy 数组）→ 系数组列表 [(ks, c), ...]。

    spec 模式下只能得到 s_m = c_m + c_{-m}，需要一个假设来“分配”：
      - symmetric：c_m = c_{-m} = s_m/2（最小范数解，不偏向任何旋转方向）
      - positive ：c_m = s_m, c_{-m} = 0（假设全部能量都在逆时针分量）
    """
    if isinstance(audio, str):
        fs, x = read_wav(audio)
    else:
        x = to_float_stereo(np.asarray(audio))
    check_nyquist(harmonics, f0, fs)
    t_global = np.arange(len(x)) / fs
    segments = split_segments(x, fs) or [(0, len(x))]

    ks = np.arange(-harmonics, harmonics + 1)
    K = harmonics
    results = []
    for s0, s1 in segments:
        PL, PR = _extract_phasors(x[s0:s1], t_global[s0:s1], K, f0, max_window_s, fs)
        c = np.zeros(2 * K + 1, dtype=complex)
        if mode == "quadrature":
            c[K] = (PL[0] + 1j * PR[0]) / 2                 # m = 0
            c[K + 1 :] = (PL[1:] + 1j * PR[1:]) / 2         # k = +m
            c[:K] = ((PL[1:] - 1j * PR[1:]) / 2)[::-1]      # k = -m（倒序放入 -K..-1）
        else:
            s = (PL + np.conj(PR)) / 2                       # 两声道的冗余估计取平均
            c[K] = s[0]
            if spec_strategy == "positive":
                c[K + 1 :] = s[1:]
            else:
                c[K + 1 :] = s[1:] / 2
                c[:K] = (s[1:] / 2)[::-1]
        results.append((ks, c))
    return results
