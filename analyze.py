"""
彙整 results.json（多個方法 x 多個 seed，n_tasks=80）：
- 計算每個方法跨 seed 的 diagonal accuracy 曲線（mean +- std）
- 計算 BWT、final_avg_acc 的跨 seed 統計
- 計算可塑性診斷曲線（dead_frac_h2、eff_rank_h1）跨 seed 平均
- 畫圖：diagonal accuracy 曲線、BWT/final_avg 長條圖、dead unit / effective rank 曲線
- 輸出彙整數字到 summary_stats.json，供報告引用
"""
import json
import os

import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    Image = ImageDraw = ImageFont = None

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREFERRED_METHODS = [
    "Naive",
    "EWC",
    "Replay",
    "ReplayEWC",
    "DarkReplayEWC",
    "AdaptiveDarkReplayEWC",
    "PressureDarkReplayEWC",
    "HorizonDarkReplayEWC",
    "BenefitDarkReplayEWC",
    "SlowBenefitDarkReplayEWC",
    "SurpriseReplayEWC",
    "MarginSurpriseReplayEWC",
    "HippocampalReplayEWC",
    "TaskBalancedReplay",
    "ContinualBP",
    "ReplayContinualBP",
]
COLORS = {
    "Naive": "#888888",
    "EWC": "#1f77b4",
    "Replay": "#2ca02c",
    "ReplayEWC": "#17becf",
    "DarkReplayEWC": "#bcbd22",
    "AdaptiveDarkReplayEWC": "#7f7f00",
    "PressureDarkReplayEWC": "#4c78a8",
    "HorizonDarkReplayEWC": "#2f4b7c",
    "BenefitDarkReplayEWC": "#f58518",
    "SlowBenefitDarkReplayEWC": "#54a24b",
    "SurpriseReplayEWC": "#e377c2",
    "MarginSurpriseReplayEWC": "#8c564b",
    "HippocampalReplayEWC": "#006d77",
    "TaskBalancedReplay": "#9467bd",
    "ContinualBP": "#d62728",
    "ReplayContinualBP": "#ff7f0e",
}


def _rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _font(size=18, bold=False):
    if ImageFont is None:
        return None
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _scale(v, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (v - src_min) * (dst_max - dst_min) / (src_max - src_min)


def _draw_axes(draw, box, x_max, y_min, y_max, y_label, x_label="Task index"):
    left, top, right, bottom = box
    axis_color = (40, 40, 40)
    grid_color = (225, 225, 225)
    label_font = _font(16)
    small_font = _font(13)
    draw.line((left, bottom, right, bottom), fill=axis_color, width=2)
    draw.line((left, top, left, bottom), fill=axis_color, width=2)
    for frac in np.linspace(0, 1, 6):
        y_val = y_min + frac * (y_max - y_min)
        y = _scale(y_val, y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=grid_color, width=1)
        draw.text((left - 58, y - 8), f"{y_val:.2f}", fill=axis_color, font=small_font)
    for x_val in np.linspace(0, x_max, 5):
        x = _scale(x_val, 0, x_max, left, right)
        draw.line((x, bottom, x, bottom + 5), fill=axis_color, width=1)
        draw.text((x - 10, bottom + 12), f"{int(x_val)}", fill=axis_color, font=small_font)
    draw.text(((left + right) / 2 - 38, bottom + 45), x_label, fill=axis_color, font=label_font)
    draw.text((16, (top + bottom) / 2 - 10), y_label, fill=axis_color, font=label_font)


def _draw_legend(draw, methods, x, y, max_rows=None):
    font = _font(15)
    for i, method in enumerate(methods):
        if max_rows is None:
            col = 0
            row = i
        else:
            col = i // max_rows
            row = i % max_rows
        xx = x + col * 220
        yy = y + row * 24
        color = _rgb(COLORS.get(method, "#555555"))
        draw.rectangle((xx, yy + 4, xx + 16, yy + 18), fill=color)
        draw.text((xx + 24, yy), method, fill=(30, 30, 30), font=font)


def _save_pil_line_chart(path, title, y_label, tasks, series, methods, y_min, y_max, hline=None):
    width, height = 1350, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = _font(24, bold=True)
    draw.text((70, 26), title, fill=(25, 25, 25), font=title_font)
    box = (90, 85, 1080, 655)
    _draw_axes(draw, box, int(tasks[-1]), y_min, y_max, y_label)
    left, top, right, bottom = box
    if hline is not None:
        y = _scale(hline, y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=(40, 40, 40), width=1)
    xs = [_scale(float(t), 0, float(tasks[-1]), left, right) for t in tasks]
    for method in methods:
        ys = [_scale(float(v), y_min, y_max, bottom, top) for v in series[method]]
        pts = list(zip(xs, ys))
        color = _rgb(COLORS.get(method, "#555555"))
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=3)
    _draw_legend(draw, methods, 1110, 100)
    img.save(path)


def _draw_bar_panel(draw, box, methods, means, stds, title, y_min, y_max):
    left, top, right, bottom = box
    font = _font(14)
    title_font = _font(18, bold=True)
    axis_color = (40, 40, 40)
    grid_color = (225, 225, 225)
    draw.text((left, top - 35), title, fill=(25, 25, 25), font=title_font)
    draw.line((left, bottom, right, bottom), fill=axis_color, width=2)
    draw.line((left, top, left, bottom), fill=axis_color, width=2)
    for frac in np.linspace(0, 1, 6):
        y_val = y_min + frac * (y_max - y_min)
        y = _scale(y_val, y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=grid_color, width=1)
        draw.text((left - 56, y - 8), f"{y_val:.2f}", fill=axis_color, font=_font(12))
    if y_min < 0 < y_max:
        y0 = _scale(0, y_min, y_max, bottom, top)
        draw.line((left, y0, right, y0), fill=(0, 0, 0), width=1)
    n = len(methods)
    slot = (right - left) / n
    bar_w = max(22, slot * 0.55)
    for i, method in enumerate(methods):
        x = left + slot * i + slot / 2
        mean = means[i]
        std = stds[i]
        y = _scale(mean, y_min, y_max, bottom, top)
        y0 = _scale(0, y_min, y_max, bottom, top) if y_min < 0 < y_max else bottom
        color = _rgb(COLORS.get(method, "#555555"))
        draw.rectangle((x - bar_w / 2, min(y, y0), x + bar_w / 2, max(y, y0)), fill=color)
        ye1 = _scale(mean - std, y_min, y_max, bottom, top)
        ye2 = _scale(mean + std, y_min, y_max, bottom, top)
        draw.line((x, ye1, x, ye2), fill=(20, 20, 20), width=2)
        draw.line((x - 6, ye1, x + 6, ye1), fill=(20, 20, 20), width=2)
        draw.line((x - 6, ye2, x + 6, ye2), fill=(20, 20, 20), width=2)
        draw.text((x - 5, bottom + 12), str(i + 1), fill=axis_color, font=font)


def _save_pil_bar_chart(path, mode, summary, methods):
    width, height = 1350, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = _font(24, bold=True)
    draw.text((70, 26), f"BWT and final average accuracy ({mode})", fill=(25, 25, 25), font=title_font)
    bwt_means = [summary[m]["bwt_mean"] for m in methods]
    bwt_stds = [summary[m]["bwt_std"] for m in methods]
    final_means = [summary[m]["final_avg_mean"] for m in methods]
    final_stds = [summary[m]["final_avg_std"] for m in methods]
    _draw_bar_panel(draw, (90, 120, 595, 620), methods, bwt_means, bwt_stds,
                    "BWT: higher / closer to 0 is better", min(-0.45, min(bwt_means) - 0.05), 0.05)
    _draw_bar_panel(draw, (720, 120, 1225, 620), methods, final_means, final_stds,
                    "Final avg accuracy: higher is better", 0.0, max(0.2, max(final_means) + 0.1))
    _draw_legend(draw, methods, 90, 660, max_rows=4)
    img.save(path)


def _draw_line_panel(draw, box, title, y_label, tasks, series, methods, y_min, y_max, hline=None):
    left, top, right, bottom = box
    draw.text((left, top - 35), title, fill=(25, 25, 25), font=_font(18, bold=True))
    _draw_axes(draw, box, int(tasks[-1]), y_min, y_max, y_label)
    if hline is not None:
        y = _scale(hline, y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=(40, 40, 40), width=1)
    xs = [_scale(float(t), 0, float(tasks[-1]), left, right) for t in tasks]
    for method in methods:
        ys = [_scale(float(v), y_min, y_max, bottom, top) for v in series[method]]
        pts = list(zip(xs, ys))
        color = _rgb(COLORS.get(method, "#555555"))
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=3)


def _save_pil_dual_line_chart(path, mode, summary, methods, tasks):
    width, height = 1500, 800
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 26), f"Plasticity diagnostics ({mode})", fill=(25, 25, 25), font=_font(24, bold=True))
    dead_series = {m: summary[m]["dead_h2_mean"] for m in methods}
    eff_series = {m: summary[m]["eff_h1_mean"] for m in methods}
    max_eff = max(max(vals) for vals in eff_series.values())
    _draw_line_panel(draw, (90, 125, 680, 640), "Dead-unit fraction h2: lower is better", "Dead frac",
                     tasks, dead_series, methods, 0.0, 1.0)
    _draw_line_panel(draw, (815, 125, 1405, 640), "Effective rank h1: higher is better", "Eff rank",
                     tasks, eff_series, methods, 0.0, max(64.0, max_eff + 5.0))
    _draw_legend(draw, methods, 90, 680, max_rows=4)
    img.save(path)


def _save_pil_figures(summary, methods, tasks, mode):
    diag_series = {m: summary[m]["diag_mean"] for m in methods}
    _save_pil_line_chart(
        os.path.join(OUT_DIR, f"fig1_diagonal_accuracy_{mode}.png"),
        f"Diagonal accuracy A[t,t] ({mode}) - higher is better",
        "Accuracy (higher better)",
        tasks,
        diag_series,
        methods,
        0.0,
        1.0,
        hline=0.1,
    )
    _save_pil_bar_chart(os.path.join(OUT_DIR, f"fig2_bwt_finalacc_{mode}.png"), mode, summary, methods)
    _save_pil_dual_line_chart(
        os.path.join(OUT_DIR, f"fig3_plasticity_diagnostics_{mode}.png"),
        mode,
        summary,
        methods,
        tasks,
    )


def load_results(mode):
    with open(os.path.join(OUT_DIR, f"results_{mode}.json")) as f:
        return json.load(f)


def diag_of(rec):
    A = np.array(rec["acc_matrix"])
    n = A.shape[0]
    return np.array([A[i, i] for i in range(n)])


def summarize_matrix(rec):
    A = np.array(rec["acc_matrix"])
    n = A.shape[0]
    diag = diag_of(rec)
    last_row = A[-1, :]
    bwt_terms = [last_row[j] - A[j, j] for j in range(n - 1)]
    forgetting_terms = []
    for j in range(n - 1):
        history = A[j:, j]
        forgetting_terms.append(np.nanmax(history) - last_row[j])
    learned_avg_acc = float(np.nanmean(diag))
    final_avg_acc = float(np.nanmean(last_row))
    return dict(
        bwt=float(np.mean(bwt_terms)) if bwt_terms else float("nan"),
        final_avg_acc=final_avg_acc,
        learned_avg_acc=learned_avg_acc,
        mean_forgetting=float(np.mean(forgetting_terms)) if forgetting_terms else float("nan"),
        retention_ratio=float(final_avg_acc / max(learned_avg_acc, 1e-12)),
        plasticity_slope=float(np.polyfit(np.arange(n), diag, 1)[0]) if n >= 2 else float("nan"),
    )


def print_method_summary(summary, methods):
    for method in methods:
        s = summary[method]
        print(f"{method:18s} early_diag={s['early_diag_mean']:.3f} late_diag={s['late_diag_mean']:.3f} "
              f"bwt={s['bwt_mean']:.3f}±{s['bwt_std']:.3f} "
              f"forget={s['mean_forgetting_mean']:.3f} retention={s['retention_ratio_mean']:.3f} "
              f"dead_h2(early->late)={s['early_dead_h2']:.2f}->{s['late_dead_h2']:.2f} "
              f"eff_h1(early->late)={s['early_eff_h1']:.1f}->{s['late_eff_h1']:.1f}")


def main():
    import sys
    mode = "label_permuted"
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    if mode not in ["label_permuted", "input_permuted", "class_il", "conflicting"]:
        raise ValueError("Mode must be 'label_permuted', 'input_permuted', 'class_il', or 'conflicting'")

    results = load_results(mode)
    methods = [m for m in PREFERRED_METHODS if m in results and len(results[m]) > 0]
    methods.extend(sorted(m for m in results if m not in methods and len(results[m]) > 0))
    if not methods:
        raise ValueError(f"No method records found in results_{mode}.json")

    n_tasks = results[methods[0]][0]["n_tasks"]
    tasks = np.arange(n_tasks)

    summary = {}
    for method in methods:
        recs = results[method]
        diags = np.stack([diag_of(r) for r in recs])  # (n_seed, n_tasks)
        rec_summaries = [summarize_matrix(r) for r in recs]
        bwts = np.array([s["bwt"] for s in rec_summaries])
        finals = np.array([s["final_avg_acc"] for s in rec_summaries])
        forgets = np.array([s["mean_forgetting"] for s in rec_summaries])
        retentions = np.array([s["retention_ratio"] for s in rec_summaries])
        slopes = np.array([s["plasticity_slope"] for s in rec_summaries])
        dead_h2 = np.stack([[d["dead_frac_h2"] for d in r["diagnostics"]] for r in recs])
        eff_h1 = np.stack([[d["eff_rank_h1"] for d in r["diagnostics"]] for r in recs])
        wn = np.stack([[d["weight_norm"] for d in r["diagnostics"]] for r in recs])
        plast_first = np.stack([r["plasticity_first_batch_acc"] for r in recs])

        summary[method] = dict(
            diag_mean=diags.mean(axis=0).tolist(),
            diag_std=diags.std(axis=0).tolist(),
            early_diag_mean=float(diags[:, :10].mean()),
            late_diag_mean=float(diags[:, -10:].mean()),
            bwt_mean=float(bwts.mean()), bwt_std=float(bwts.std()),
            final_avg_mean=float(finals.mean()), final_avg_std=float(finals.std()),
            mean_forgetting_mean=float(forgets.mean()), mean_forgetting_std=float(forgets.std()),
            retention_ratio_mean=float(retentions.mean()), retention_ratio_std=float(retentions.std()),
            plasticity_slope_mean=float(slopes.mean()), plasticity_slope_std=float(slopes.std()),
            dead_h2_mean=dead_h2.mean(axis=0).tolist(),
            eff_h1_mean=eff_h1.mean(axis=0).tolist(),
            weight_norm_mean=wn.mean(axis=0).tolist(),
            plasticity_first_batch_mean=plast_first.mean(axis=0).tolist(),
            early_dead_h2=float(dead_h2[:, :10].mean()), late_dead_h2=float(dead_h2[:, -10:].mean()),
            early_eff_h1=float(eff_h1[:, :10].mean()), late_eff_h1=float(eff_h1[:, -10:].mean()),
        )

    with open(os.path.join(OUT_DIR, f"summary_stats_{mode}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if plt is None:
        print(f"saved summary_stats_{mode}.json")
        if Image is not None:
            _save_pil_figures(summary, methods, tasks, mode)
            print(f"matplotlib is not installed; saved Pillow fallback PNG figures for {mode}.")
        else:
            print("matplotlib and Pillow are not installed in this Python environment; skipped figure generation.")
        print_method_summary(summary, methods)
        return

    # --- 圖 1：diagonal accuracy（可塑性）曲線，含 seed 間 std 帶 ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in methods:
        s = summary[method]
        mean = np.array(s["diag_mean"])
        std = np.array(s["diag_std"])
        color = COLORS.get(method, None)
        ax.plot(tasks, mean, label=method, color=color, linewidth=2)
        ax.fill_between(tasks, mean - std, mean + std, color=color, alpha=0.15)
    ax.axhline(0.1, color="black", linestyle="--", linewidth=1, label="chance (10%)")
    ax.set_xlabel("Task index")
    ax.set_ylabel("Accuracy (higher is better)")
    ax.set_title(f"Plasticity over a long task stream: diagonal accuracy A[t,t]\n(higher is better; {mode}, n_tasks=80, mean ± std over 3 seeds)")
    ax.legend()
    ax.set_ylim(0, 0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig1_diagonal_accuracy_{mode}.png"), dpi=150)
    plt.close(fig)

    # --- 圖 2：BWT 與 final_avg_acc 長條圖 ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(methods))
    bwt_means = [summary[m]["bwt_mean"] for m in methods]
    bwt_stds = [summary[m]["bwt_std"] for m in methods]
    axes[0].bar(x, bwt_means, yerr=bwt_stds, color=[COLORS.get(m, None) for m in methods], capsize=4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods, rotation=20)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Backward Transfer (BWT)\n(higher / closer to 0 is better)")
    axes[0].set_ylabel("BWT")

    final_means = [summary[m]["final_avg_mean"] for m in methods]
    final_stds = [summary[m]["final_avg_std"] for m in methods]
    axes[1].bar(x, final_means, yerr=final_stds, color=[COLORS.get(m, None) for m in methods], capsize=4)
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods, rotation=20)
    axes[1].axhline(0.1, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"Final average accuracy over all 80 tasks ({mode})\n(higher is better)")
    axes[1].set_ylabel("accuracy (higher is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig2_bwt_finalacc_{mode}.png"), dpi=150)
    plt.close(fig)

    # --- 圖 3：可塑性流失診斷（dead unit fraction、effective rank）---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in methods:
        s = summary[method]
        axes[0].plot(tasks, s["dead_h2_mean"], label=method, color=COLORS.get(method, None), linewidth=2)
        axes[1].plot(tasks, s["eff_h1_mean"], label=method, color=COLORS.get(method, None), linewidth=2)
    axes[0].set_title("Dead-unit fraction, hidden layer 2\n(lower is better)")
    axes[0].set_xlabel("Task index"); axes[0].set_ylabel("dead fraction (lower is better)")
    axes[0].legend()
    axes[1].set_title("Effective rank, hidden layer 1\n(higher is better; less representation collapse)")
    axes[1].set_xlabel("Task index"); axes[1].set_ylabel("effective rank (higher is better)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig3_plasticity_diagnostics_{mode}.png"), dpi=150)
    plt.close(fig)

    print(f"saved fig1_diagonal_accuracy_{mode}.png, fig2_bwt_finalacc_{mode}.png, fig3_plasticity_diagnostics_{mode}.png, summary_stats_{mode}.json")
    print_method_summary(summary, methods)


if __name__ == "__main__":
    main()
