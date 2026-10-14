"""
main.py —— 图案 ⇄ 声波 双向转换器 CLI

用法
----
  图案 → 声波（并自动做闭环验证）：
      python main.py --image star.png [--harmonics 64] [--base-freq 220] [--out outputs]
  声波 → 图案：
      python main.py --audio star_quadrature.wav [--mode quadrature] [--harmonics 64] [--base-freq 220]
  一键跑三个测试图案并生成误差报告：
      python main.py --demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audio as au  # noqa: E402
import fourier as fr  # noqa: E402
import visualize as vz  # noqa: E402
from contour import compare_contour_sets, extract_contours  # noqa: E402


# ---------------------------------------------------------------------------
# 测试图案
# ---------------------------------------------------------------------------
def make_test_images(folder: str) -> dict[str, str]:
    """生成三个测试图案：五角星、圆、麦田怪圈风格的三重对称几何图案。"""
    os.makedirs(folder, exist_ok=True)
    paths = {}

    # 五角星：外顶点半径 R，内顶点半径 R·sin18°/sin126°（正五角星比例）
    img = np.full((512, 512), 255, np.uint8)
    R, r = 210, 210 * np.sin(np.radians(18)) / np.sin(np.radians(126))
    ang = np.pi / 2 + np.arange(10) * np.pi / 5
    rad = np.where(np.arange(10) % 2 == 0, R, r)
    poly = np.c_[256 + rad * np.cos(ang), 256 - rad * np.sin(ang)].astype(np.int32)
    cv2.fillPoly(img, [poly], 0, lineType=cv2.LINE_AA)
    paths["star"] = os.path.join(folder, "star.png")
    cv2.imwrite(paths["star"], img)

    img = np.full((512, 512), 255, np.uint8)
    cv2.circle(img, (256, 256), 190, 0, -1, lineType=cv2.LINE_AA)
    paths["circle"] = os.path.join(folder, "circle.png")
    cv2.imwrite(paths["circle"], img)

    # 麦田怪圈：深色麦田 + 浅色压倒区；中心圆 + 圆环 + 3 条直线小径 + 卫星圆 + 散落小圆
    rng = np.random.default_rng(7)
    h = w = 640
    field = rng.normal(70, 14, (h, w))
    field += 10 * np.sin(np.arange(w) / 3.0)[None, :]          # 模拟麦垄条纹
    mask = np.zeros((h, w), np.uint8)
    c = (w // 2, h // 2)
    cv2.circle(mask, c, 40, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, c, 112, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, c, 92, 0, -1, cv2.LINE_AA)
    cv2.circle(mask, c, 40, 255, -1, cv2.LINE_AA)
    for deg in (90, 210, 330):
        a = np.radians(deg)
        p1 = (int(c[0] + 105 * np.cos(a)), int(c[1] - 105 * np.sin(a)))
        p2 = (int(c[0] + 228 * np.cos(a)), int(c[1] - 228 * np.sin(a)))
        cv2.line(mask, p1, p2, 255, 10, cv2.LINE_AA)
        cv2.circle(mask, p2, 30, 255, -1, cv2.LINE_AA)
    for deg in (30, 150, 270):
        a = np.radians(deg)
        cv2.circle(mask, (int(c[0] + 190 * np.cos(a)), int(c[1] - 190 * np.sin(a))), 16, 255, -1, cv2.LINE_AA)
    photo = np.where(mask > 127, rng.normal(205, 12, (h, w)), field)
    paths["crop_circle"] = os.path.join(folder, "crop_circle.png")
    cv2.imwrite(paths["crop_circle"], np.clip(photo, 0, 255).astype(np.uint8))
    return paths


# ---------------------------------------------------------------------------
# 图案 → 声波 → 图案 完整闭环
# ---------------------------------------------------------------------------
def spectral_centroid(coeff_sets, f0: float) -> float:
    """频谱质心 Σ f_k |c_k|² / Σ |c_k|²：心理声学里“明亮度”的主要指标。"""
    num = den = 0.0
    for ks, c in coeff_sets:
        e = np.abs(c) ** 2
        num += float(np.sum(au.harmonic_frequency(ks, f0) * e))
        den += float(e.sum())
    return num / max(den, 1e-30)


def process_image(path, out_dir, name=None, harmonics=64, f0=220.0, n_points=512,
                  duration=2.5, max_contours=8, smooth=1.0, anim=True, anim_format="gif"):
    name = name or os.path.splitext(os.path.basename(path))[0]
    od = os.path.join(out_dir, name)
    os.makedirs(od, exist_ok=True)
    print(f"[{name}] 提取轮廓 …")

    # A1/A2 轮廓提取 + 等弧长重采样 + 平滑 + 起点对齐
    ext = extract_contours(path, n_points=n_points, smooth_sigma=smooth, max_contours=max_contours)
    contours = ext["contours"]
    cv2.imwrite(os.path.join(od, "original.png"), ext["image"])

    # A3 傅里叶分解
    coeffs = [fr.fourier_coefficients(z, harmonics) for z in contours]
    np.savez(os.path.join(od, "coefficients.npz"),
             ks=coeffs[0][0], c=np.array([c for _, c in coeffs]))

    # 基线：只做 K 阶截断（不经过音频），用于把“截断损失”和“音频损失”分开
    trunc = [fr.reconstruct(ks, c, n_points) for ks, c in coeffs]
    results = {"truncation": {**compare_contour_sets(contours, trunc), "coef_error": 0.0}}

    # A4 音频合成 → 写 WAV → 读回 WAV → B 反解系数 → 重建
    recon_curves = {"truncation": trunc}
    audio_files = {}
    main_audio = None
    for mode in au.MODES:
        pcm, gain = au.encode(coeffs, f0=f0, duration=duration, mode=mode)
        wav = os.path.join(od, f"{name}_{mode}.wav")
        au.write_wav(wav, pcm)
        audio_files[mode] = os.path.basename(wav)
        fs, x = au.read_wav(wav)
        if mode == "quadrature":
            main_audio = x
        strategies = ["symmetric", "positive"] if mode == "spec" else [None]
        for strat in strategies:
            key = mode if strat is None else f"{mode}_{strat}"
            dec = au.decode(wav, harmonics=harmonics, f0=f0, mode=mode, spec_strategy=strat or "symmetric")
            if len(dec) != len(coeffs):
                raise RuntimeError(f"解码段数 {len(dec)} ≠ 轮廓数 {len(coeffs)}")
            curves = [fr.reconstruct(ks, c, n_points) for ks, c in dec]
            coef_err, g = fr.coefficient_error([c for _, c in coeffs], [c for _, c in dec])
            results[key] = {**compare_contour_sets(contours, curves), "coef_error": coef_err,
                            "gain_fit": g, "encode_gain": gain}
            # 音频峰值归一化让系数整体放大了 g 倍；对比图中除以 g，与原轮廓叠放在同一尺度
            recon_curves[key] = [z / g for z in curves] if abs(g) > 1e-9 else curves
            vz.plot_pattern(os.path.join(od, f"recon_{key}.png"), dec,
                            vz._t(f"由音频还原（{key}）", f"Decoded from audio ({key})"))

    # C 可视化
    ks0, c0 = coeffs[0]
    vz.plot_spectrum(os.path.join(od, "spectrum.png"), ks0, c0, main_audio, au.FS, f0,
                     vz._t(f"{name}：最大轮廓的频谱", f"{name}: spectrum of largest contour"))
    labels = {
        "truncation": vz._t(f"仅截断 K={harmonics}", f"Truncated K={harmonics}"),
        "quadrature": vz._t("音频闭环 · 正交编码", "Audio loop · quadrature"),
        "spec_symmetric": vz._t("音频闭环 · 原规则(对称解)", "Audio loop · spec (symmetric)"),
        "spec_positive": vz._t("音频闭环 · 原规则(正频解)", "Audio loop · spec (positive)"),
    }
    vz.plot_comparison(os.path.join(od, "comparison.png"), ext["image"], contours,
                       [(labels[k], recon_curves[k], results[k]) for k in labels])
    anim_file = None
    if anim:
        print(f"[{name}] 渲染傅里叶圆动画 …")
        anim_file = vz.animate_epicycles(os.path.join(od, f"epicycles.{anim_format}"), coeffs,
                                         title=vz._t(f"{name} 傅里叶圆", f"{name} epicycles"))

    sym = fr.rotational_symmetry(ks0, c0)
    summary = {
        "name": name,
        "n_contours": len(contours),
        "harmonics": harmonics,
        "base_freq": f0,
        "dominant": [{"k": k, "amp": a, "freq": f0 * (abs(k) + 1)} for k, a in fr.dominant_harmonics(ks0, c0, 6)],
        "symmetry": {"order": sym[0], "energy_fraction": sym[1]},
        "high_freq_ratio": fr.high_frequency_ratio(ks0, c0),
        "spectral_centroid_hz": spectral_centroid(coeffs, f0),
        "dc_ratio": float(sum(abs(c[ks == 0][0]) ** 2 for ks, c in coeffs)
                          / sum(float(np.sum(np.abs(c) ** 2)) for _, c in coeffs)),
        "metrics": results,
        "files": {
            "original": "original.png", "comparison": "comparison.png", "spectrum": "spectrum.png",
            "animation": os.path.basename(anim_file) if anim_file else None, "audio": audio_files,
            "recon": {k: f"recon_{k}.png" for k in results if k != "truncation"},
        },
    }
    with open(os.path.join(od, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _print_summary(summary)
    return summary


def _print_summary(s):
    print(f"[{s['name']}] 轮廓数 {s['n_contours']}，主导谐波："
          + ", ".join(f"k={d['k']:+d}({d['freq']:.0f}Hz)" for d in s["dominant"][:4]))
    for k, m in s["metrics"].items():
        print(f"    {k:16s} Hausdorff={m['hausdorff']:.4f}  平均HD={m['mean_hausdorff']:.4f}  "
              f"面积差={m['area_diff'] * 100:6.2f}%  系数误差={m['coef_error']:.2e}")


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
TITLES = {"star": "五角星", "circle": "圆形", "crop_circle": "麦田怪圈（三重对称几何图案）"}
ROW_NAMES = {
    "truncation": "仅 K 阶截断（不经音频）",
    "quadrature": "音频闭环 · 正交立体声",
    "spec_symmetric": "音频闭环 · 原规则 · 对称解",
    "spec_positive": "音频闭环 · 原规则 · 正频解",
}


def write_report(summaries, path):
    L = ["# 闭环误差分析报告", "",
         "流程：图案 → 轮廓 → 系数 c_k → WAV(44.1 kHz/16-bit) → 读回 → 反解 c_k → 重建轮廓。",
         "距离以“图案 RMS 半径”为单位（比较前做联合平移/缩放归一化，因为峰值归一化本身就丢弃了绝对尺度）。", ""]
    for s in summaries:
        sym = s["symmetry"]["order"]
        sym_txt = "圆（全部能量在 k=+1）" if sym == "inf" else f"{sym} 重旋转对称（k ≡ 1 mod {sym} 能量占比 {s['symmetry']['energy_fraction'] * 100:.1f}%）"
        L += [f"## {TITLES.get(s['name'], s['name'])}", "",
              f"- 轮廓数：{s['n_contours']}；检测到：{sym_txt}",
              "- 主导谐波：" + "，".join(f"k={d['k']:+d} → {d['freq']:.0f} Hz" for d in s["dominant"]),
              f"- 高频能量占比（|k|>K/4）：{s['high_freq_ratio'] * 100:.3f}%；频谱质心：{s['spectral_centroid_hz']:.0f} Hz",
              "", "| 环节 | Hausdorff | 平均 Hausdorff | 面积差 | 系数相对误差 |", "|---|---|---|---|---|"]
        for k, m in s["metrics"].items():
            L.append(f"| {ROW_NAMES[k]} | {m['hausdorff']:.4f} | {m['mean_hausdorff']:.4f} | "
                     f"{m['area_diff'] * 100:.2f}% | {m['coef_error']:.2e} |")
        L.append("")
    L += [
        "## 信息在哪里丢失、为什么", "",
        "1. **原规则立体声不可逆（最大损失）。** 频率 f₀(|k|+1) 让 c_k 与 c_{-k} 落在同一频率。"
        "用相量表示，左声道 Σ|c|cos(ωt+φ) 的相量是 c_m + c_{-m}，右声道 Σ|c|cos(ωt−φ) 的相量恰为其复共轭——"
        "右声道是左声道的冗余拷贝，从音频只能得到和 s_m = c_m + c_{-m}，两个旋转方向无法分离。",
        "   - 对称解 c_{±m}=s_m/2 重建出 z'(t) = [z(t)+z(−t)]/2，满足 z'(t)=z'(−t)：曲线沿同一路径去而复返，面积为 0。圆退化为线段，五角星塌缩成一条折线。",
        "   - 正频解 c_m=s_m, c_{-m}=0 假设全部能量逆时针旋转：圆恰好无损（它只有 c_{+1}），但五角星的 c_{-4}, c_{-9}… 被翻到 +4, +9，五重对称被破坏。",
        "2. **修复：正交立体声。** 振荡器的频率/幅值/相位仍按原规则，右声道改为 sgn(k)·|c_k|·sin(ωt+φ)，"
        "则 P_R = −i·c_m + i·c_{-m}，联立得 c_{±m} = (P_L ± i·P_R)/2，严格可逆（与示波器音乐 X/Y 同构）。",
        "3. **全局增益。** 为防止削波做了峰值归一化，增益未写入音频，因此绝对尺寸只能恢复到一个正实数倍；形状、相对位置、相对大小都保留。",
        "4. **16-bit 量化。** 量化噪声约为满幅的 2⁻¹⁶，主导系数几乎无损，但 |c_k| 很小的高阶谐波相对误差较大，表现为正交模式下 ~10⁻⁴ 量级的系数误差。",
        "5. **K 阶截断（发生在音频之前）。** |k|>K 的细节无法进入音频：尖角被圆滑、细直线出现 Gibbs 振铃。麦田怪圈的细小径受影响最明显。",
        "6. **预处理有意丢弃的信息。** 像素锯齿被重采样+平滑去掉；所有轮廓统一为逆时针，所以“哪条是孔洞边界”的拓扑信息不进入音频；多轮廓按时间分段，段的顺序即轮廓面积排序。",
        "",
        "## 图案特征 ↔ 听觉特征", "",
        "| 图案特征 | 系数上的表现 | 听到的是 |",
        "|---|---|---|",
        "| 整体大小 | 所有 |c_k| 等比缩放 | 被峰值归一化抹平（听不出） |",
        "| 偏离中心的位置 | c₀ 增大 | 基频 f₀ 处的持续低音变响 |",
        "| 椭圆/整体轮廓 | c_{+1}（与 c_{−1}）主导 | 2f₀ 处的强音，接近纯音 |",
        "| p 重旋转对称 | 只有 k ≡ 1 (mod p) 非零：峰在 1, 1−p, 1+p, 1−2p… | 泛音列稀疏、等间隔：五角星峰在 2f₀、5f₀、7f₀、10f₀、12f₀…，像带“空洞”的和弦 |",
        "| 尖角、细线、毛刺（粗糙度） | |c_k| 随 k 衰减变慢（角 ≈ 1/k²） | 高频能量与频谱质心升高 → 更亮、更刺耳 |",
        "| 光滑曲线 | 系数指数级衰减 | 柔和、接近纯音 |",
        "| 描绘方向 / 起点 | 方向：c_k ↔ c_{−k}；起点：c_k 乘 e^{i2πkτ} | 单声道听感不变，只体现为左右声道的相位关系 |",
        "| 多个部件 | 多组系数 | 依次播放的多个音块 |",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="图案 ⇄ 声波 双向转换器（傅里叶级数绘图仪）")
    p.add_argument("--image", help="输入图片（PNG/JPG）：图案 → 声波，并做闭环验证")
    p.add_argument("--audio", help="输入 WAV：声波 → 图案")
    p.add_argument("--demo", action="store_true", help="生成三个测试图案并输出全部结果与报告")
    p.add_argument("--harmonics", type=int, default=64, help="傅里叶阶数 K（k = -K..K），默认 64")
    p.add_argument("--base-freq", type=float, default=220.0, help="基频 f0 (Hz)，默认 220")
    p.add_argument("--mode", choices=au.MODES, default="quadrature",
                   help="--audio 解码时使用的立体声编码：quadrature(可逆) 或 spec(题目原规则)")
    p.add_argument("--points", type=int, default=512, help="每条轮廓的等弧长采样点数")
    p.add_argument("--duration", type=float, default=2.5, help="音频时长（秒）")
    p.add_argument("--max-contours", type=int, default=8)
    p.add_argument("--smooth", type=float, default=1.0, help="周期高斯平滑 σ（采样点），0 关闭")
    p.add_argument("--no-anim", action="store_true", help="跳过傅里叶圆动画")
    p.add_argument("--anim-format", choices=["gif", "mp4"], default="gif")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    common = dict(harmonics=a.harmonics, f0=a.base_freq, n_points=a.points, max_contours=a.max_contours,
                  smooth=a.smooth, anim=not a.no_anim, anim_format=a.anim_format)

    if a.demo:
        imgs = make_test_images(os.path.join(a.out, "test_images"))
        summaries = []
        for name, path in imgs.items():
            dur = 3.0 if name == "crop_circle" else a.duration
            summaries.append(process_image(path, a.out, name, duration=dur, **common))
        write_report(summaries, os.path.join(a.out, "report.md"))
        with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"完成，报告：{os.path.join(a.out, 'report.md')}")
    elif a.image:
        s = process_image(a.image, a.out, duration=a.duration, **common)
        write_report([s], os.path.join(a.out, s["name"], "report.md"))
    elif a.audio:
        dec = au.decode(a.audio, harmonics=a.harmonics, f0=a.base_freq, mode=a.mode)
        stem = os.path.splitext(os.path.basename(a.audio))[0]
        out_png = os.path.join(a.out, f"{stem}_pattern.png")
        vz.plot_pattern(out_png, dec, vz._t(f"由音频还原：{stem}", f"Decoded: {stem}"))
        np.savez(os.path.join(a.out, f"{stem}_coefficients.npz"), ks=dec[0][0], c=np.array([c for _, c in dec]))
        if not a.no_anim:
            vz.animate_epicycles(os.path.join(a.out, f"{stem}_epicycles.{a.anim_format}"), dec)
        print(f"解出 {len(dec)} 条轮廓 → {out_png}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
