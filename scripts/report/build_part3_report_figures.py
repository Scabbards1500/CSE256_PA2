"""Redraw compact paper figures from saved Part 3 results, without model inference."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter, NullLocator
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "figures"
DATA = json.loads((ROOT / "outputs" / "part3" / "results.json").read_text(encoding="utf-8"))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "savefig.facecolor": "white",
})


def save(fig: plt.Figure, name: str) -> None:
    """Keep the explicit canvas dimensions so LaTeX scaling is predictable."""
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    assert bounds.x0 >= 0 and bounds.y0 >= 0, (name, "clipped left/bottom", bounds)
    assert bounds.x1 <= fig.get_figwidth() and bounds.y1 <= fig.get_figheight(), (name, "clipped right/top", bounds)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    print(f"{name}: {fig.get_figwidth():.2f} x {fig.get_figheight():.2f} inches")
    plt.close(fig)


def visible_token(tokenizer, token_id: int) -> str:
    return (tokenizer.decode([int(token_id)])
            .replace(" ", "_").replace("\n", r"\n").replace("\t", r"\t"))


def wrap_token(token: str, renderer, font: FontProperties, max_width: float) -> str:
    """Wrap complete token strings at measured glyph widths; never truncate them."""
    lines, current = [], ""
    for char in token:
        candidate = current + char
        if current and renderer.get_text_width_height_descent(candidate, font, False)[0] > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    lines.append(current)
    wrapped = "\n".join(lines)
    assert wrapped.replace("\n", "") == token
    return wrapped


def draw_grid(tokenizer, key: str) -> None:
    grid = DATA["grids"][key]
    probabilities = np.asarray(grid["true_next_probability"], dtype=float)
    top_ids = np.asarray(grid["top_ids"], dtype=int)
    assert probabilities.shape == top_ids.shape == (13, 10)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    fig = plt.figure(figsize=(7.10, 3.60))
    ax = fig.add_axes([0.101, 0.190, 0.765, 0.795])
    cax = fig.add_axes([0.894, 0.190, 0.015, 0.795])
    image = ax.pcolormesh(np.arange(-0.5, 10.5), np.arange(-0.5, 13.5),
                         probabilities, vmin=0, vmax=1, cmap="Blues", shading="flat")
    ax.set(xlim=(-0.5, 9.5), ylim=(12.5, -0.5))
    ax.set_yticks(range(13), ["0 (emb)"] + [str(i) for i in range(1, 13)])
    ax.set_ylabel("Lens depth", labelpad=4)
    ax.tick_params(axis="both", length=0, pad=4)
    labels = [f"{visible_token(tokenizer, x)}\n$\\rightarrow$ {visible_token(tokenizer, y)}"
              for x, y in zip(grid["input_ids"], grid["target_ids"])]
    ax.set_xticks(range(10), labels, fontsize=7.4, linespacing=1.2)
    ax.set_xlabel("Input token → true next token (underscores mark spaces)", labelpad=7)
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 13, 1), minor=True)
    ax.grid(which="minor", color="#c6d1dc", linewidth=0.35)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#8493a1")
    bar = fig.colorbar(image, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    bar.solids.set_rasterized(False)
    bar.set_label("True next-token probability", labelpad=4)
    bar.ax.tick_params(labelsize=7.4, length=2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # ACL's 16 cm text width scales this 7.1 inch canvas by approximately 0.887.
    font = FontProperties(family="DejaVu Sans", size=7.4)
    cell_width = ax.get_window_extent().width / 10
    cell_height = ax.get_window_extent().height / 13
    cells = []
    for depth in range(13):
        for position in range(10):
            label = wrap_token(visible_token(tokenizer, top_ids[depth, position]),
                               renderer, font, cell_width - 4)
            cells.append(ax.text(position, depth, label, fontproperties=font,
                                 ha="center", va="center", linespacing=1.0,
                                 color="white" if probabilities[depth, position] > 0.55 else "#152333"))
    fig.canvas.draw()
    for cell in cells:
        box = cell.get_window_extent(fig.canvas.get_renderer())
        assert box.width <= cell_width - 2, (key, cell.get_text(), box.width, cell_width)
        assert box.height <= cell_height - 1, (key, cell.get_text(), box.height, cell_height)
    save(fig, f"part3_grid_{key}")


def draw_lens() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.10, 2.60))
    fig.subplots_adjust(left=0.075, right=0.987, bottom=0.19, top=0.88, wspace=0.30)
    depths = np.arange(13)
    final = np.asarray(DATA["lens"]["step143000"]["accuracy"]) * 100
    axes[0].plot(depths, final, color="#21699c", marker="o", markersize=3.4, linewidth=1.5)
    axes[0].set_title("(a) Final checkpoint", loc="left", pad=6, fontsize=9.5)
    for depth, offset, align in [(0, (5, 8), "left"), (6, (0, 10), "center"), (12, (-7, 3), "right")]:
        axes[0].annotate(f"{final[depth]:.4f}%", (depth, final[depth]),
                         xytext=offset, textcoords="offset points", ha=align, va="bottom", fontsize=9.1)
    styles = [("step512", "#8a609a", "s"), ("step1000", "#d78522", "^"),
              ("step8000", "#31947d", "D"), ("step143000", "#21699c", "o")]
    for checkpoint, color, marker in styles:
        values = np.asarray(DATA["lens"][checkpoint]["accuracy"]) * 100
        axes[1].plot(depths, values, color=color, marker=marker, markersize=2.8,
                     linewidth=1.2, label=checkpoint)
    axes[1].set_title("(b) Across training", loc="left", pad=6, fontsize=9.5)
    axes[1].legend(loc="upper left", frameon=False, borderaxespad=0.25,
                   handlelength=1.5, handletextpad=0.4, labelspacing=0.22, fontsize=9.1)
    for ax in axes:
        ax.set(xlim=(-0.3, 12.3), ylim=(-0.7, 40), xlabel="Lens depth",
               ylabel="Next-token accuracy (%)", xticks=range(0, 13, 2), yticks=range(0, 41, 10))
        ax.grid(color="#dce2e7", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.6, labelsize=9.1)
        ax.xaxis.label.set_fontsize(9.1)
        ax.yaxis.label.set_fontsize(9.1)
    save(fig, "part3_lens_accuracy")


def draw_transplants() -> None:
    deltas = np.asarray(DATA["deltas"], dtype=float)
    assert deltas.shape == (14, 4)
    assert np.all(deltas > 0), "Logarithmic plotting assumes the saved increases are positive."
    assert np.allclose(deltas, np.asarray(DATA["transplant_losses"]) - DATA["baseline_loss"], atol=1e-12)
    fig = plt.figure(figsize=(3.35, 4.05))
    ax = fig.add_axes([0.235, 0.235, 0.735, 0.735])
    cax = fig.add_axes([0.235, 0.090, 0.735, 0.030])
    norm = LogNorm(vmin=0.1, vmax=100)
    image = ax.pcolormesh(np.arange(-0.5, 4.5), np.arange(-0.5, 14.5),
                         deltas, cmap="YlOrRd", norm=norm, shading="flat")
    ax.set(xlim=(-0.5, 3.5), ylim=(13.5, -0.5))
    ax.set_yticks(range(14), ["Embedding"] + [f"Block {i}" for i in range(12)] + ["Unembedding"])
    ax.set_xticks(range(4), ["0", "1,000", "8,000", "32,000"])
    ax.set_xlabel("Donor training step", labelpad=5)
    ax.tick_params(axis="both", length=0, pad=3, labelsize=7)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 14, 1), minor=True)
    ax.grid(which="minor", color="#ead5ba", linewidth=0.35)
    ax.tick_params(which="minor", length=0)
    for row in range(14):
        for column in range(4):
            value = deltas[row, column]
            ax.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=7.3,
                    color="white" if norm(value) > 0.70 else "#2c2925")
    bar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[0.1, 1, 10, 100])
    bar.solids.set_rasterized(False)
    bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    bar.ax.xaxis.set_minor_locator(NullLocator())
    bar.ax.tick_params(labelsize=7, length=2, pad=2)
    bar.set_label("Loss increase (nats/token; log scale)", fontsize=7.5, labelpad=3)
    save(fig, "part3_transplants")


def main() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
        tokenizer = AutoTokenizer.from_pretrained(DATA["model"], revision="step143000",
            cache_dir=str(ROOT / ".cache" / "huggingface"), local_files_only=True)
    draw_grid(tokenizer, "obama")
    draw_grid(tokenizer, "custom")
    draw_lens()
    draw_transplants()


if __name__ == "__main__":
    main()
