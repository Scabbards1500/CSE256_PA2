"""Draw compact, vector model diagrams matching the implemented architectures."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'report/figures'
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
INK = '#273747'
LINE = '#627383'
CENTER = 1.72


def draw(kind):
    encoder = kind == 'encoder'
    accent = '#3975A3' if encoder else '#267E73'
    fill = '#EDF4FA' if encoder else '#EAF5F1'
    fig, ax = plt.subplots(figsize=(3.35, 2.50))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 3.35), ylim=(0, 2.50), aspect='equal')
    ax.axis('off')

    def label(x, y, text, size=8, **kwargs):
        return ax.text(x, y, text, ha='center', va='center', fontsize=size,
                       color=INK, linespacing=1.20, **kwargs)

    def box(y, height, text, width=2.48, colored=False, size=8):
        patch = FancyBboxPatch((CENTER-width/2, y-height/2), width, height,
                              boxstyle='round,pad=0.008,rounding_size=0.035',
                              linewidth=0.8, edgecolor=accent if colored else LINE,
                              facecolor=fill if colored else 'white', zorder=3)
        ax.add_patch(patch)
        label(CENTER, y, text, size=size, zorder=4)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                    mutation_scale=7, linewidth=0.8,
                                    color=LINE, shrinkA=1.2, shrinkB=1.2, zorder=2))

    def residual(split_y, add_y):
        bypass_x = 0.34
        ax.plot([CENTER, bypass_x, bypass_x], [split_y, split_y, add_y],
                color=LINE, linewidth=0.8, zorder=2)
        ax.plot(CENTER, split_y, '.', color=LINE, markersize=2.6, zorder=3)
        arrow((bypass_x, add_y), (CENTER-0.055, add_y))
        ax.add_patch(Circle((CENTER, add_y), 0.055, facecolor='white',
                            edgecolor=accent, linewidth=0.8, zorder=3))
        label(CENTER, add_y+0.002, '+', size=8, zorder=4)

    box(2.30, 0.28, 'Token + learned position embeddings', size=7.8)

    # Each repeated block contains two pre-normalized residual sublayers.
    enclosure = FancyBboxPatch((0.16, 1.02), 2.97, 1.07,
                              boxstyle='round,pad=0.01,rounding_size=0.055',
                              edgecolor=accent, facecolor='#FAFCFD',
                              linewidth=0.85, linestyle=(0, (3, 2)), zorder=0)
    ax.add_patch(enclosure)
    label(0.79, 2.08, 'Block × 4', size=7.5,
          bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.4}, zorder=5)
    attention = 'Bidirectional self-attention' if encoder else 'Causal self-attention'
    box(1.82, 0.30, 'LayerNorm → ' + attention + '\n2 heads',
        width=2.48, colored=True, size=7.3)
    arrow((CENTER, 2.152), (CENTER, 1.978))
    residual(2.02, 1.595)
    ax.plot([CENTER, CENTER], [1.67, 1.65], color=LINE, linewidth=0.8)

    hidden = 256 if encoder else 100
    box(1.36, 0.30, f'LayerNorm → FFN (ReLU)\n64 → {hidden} → 64', colored=True)
    ax.plot([CENTER, CENTER], [1.54, 1.51], color=LINE, linewidth=0.8)
    residual(1.53, 1.135)
    ax.plot([CENTER, CENTER], [1.21, 1.19], color=LINE, linewidth=0.8)

    arrow((CENTER, 1.08), (CENTER, 0.938))
    if encoder:
        box(0.78, 0.30, 'Final LayerNorm\nMean pooling (non-padding tokens)', size=7.8)
        box(0.27, 0.36, 'ReLU classifier: 64 → 100 → 3\nSpeaker logits', colored=True)
    else:
        box(0.78, 0.30, 'Final LayerNorm')
        box(0.27, 0.36, 'Untied linear: 64 → 5,755\nNext-token logits at each position', colored=True)
    arrow((CENTER, 0.622), (CENTER, 0.458))

    # Save a vector version for the report and a raster preview for inspection.
    fig.savefig(FIGURES / f'{kind}_architecture.pdf')
    fig.savefig(FIGURES / f'{kind}_architecture.png', dpi=240)
    plt.close(fig)


if __name__ == '__main__':
    for model_kind in ('encoder', 'decoder'):
        draw(model_kind)
    print('Saved encoder and decoder architecture diagrams.')
