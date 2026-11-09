"""Regenerate Part 1's LaTeX table/macros and figures from saved run outputs."""

import json
import shutil
import runpy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / 'outputs' / 'part1'
report = ROOT / 'report'
metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
history = metrics['history']
generated = report / 'generated'
figures = report / 'figures'
generated.mkdir(parents=True, exist_ok=True)
figures.mkdir(parents=True, exist_ok=True)

macros = [
    rf"\newcommand{{\PartOneFinalAccuracy}}{{{history[-1]['test_accuracy']:.2f}}}",
    rf"\newcommand{{\PartOneTrainAccuracy}}{{{history[-1]['train_accuracy']:.2f}}}",
    rf"\newcommand{{\PartOneEncoderParams}}{{{metrics['encoder_parameters']:,}}}",
    rf"\newcommand{{\PartOneClassifierParams}}{{{metrics['classifier_parameters']:,}}}",
]
(generated / 'part1_numbers.tex').write_text('\n'.join(macros) + '\n', encoding='utf-8')
rows = [r'\begin{tabular}{rrr}', r'\toprule',
        r'Epoch & Train accuracy (\%) & Test accuracy (\%) \\', r'\midrule']
for row in history:
    rows.append(f"{row['epoch']} & {row['train_accuracy']:.2f} & {row['test_accuracy']:.2f} " + r'\\')
rows += [r'\bottomrule', r'\end{tabular}']
(generated / 'part1_accuracy_table.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')

# Sized for a two-column figure, with every value replacing the separate table.
fig, ax = plt.subplots(figsize=(7.1, 2.4), constrained_layout=True)
epochs = [r['epoch'] for r in history]
for key, label, color, marker, offset, alignment in (
    ('train_accuracy', 'Train', '#2463a0', 'o', 7, 'bottom'),
    ('test_accuracy', 'Test', '#b34c12', 's', -9, 'top'),
):
    values = [r[key] for r in history]
    ax.plot(epochs, values, color=color, marker=marker, label=label,
            markersize=3.5, linewidth=1.5)
    for epoch, value in zip(epochs, values):
        ax.annotate(f'{value:.2f}', (epoch, value), xytext=(0, offset),
                    textcoords='offset points', ha='center', va=alignment,
                    fontsize=7.5, color=color)
ax.set(xlabel='Epoch', ylabel='Accuracy (%)', xticks=epochs,
       xlim=(0.4, 15.6), ylim=(35, 107), yticks=range(40, 101, 10))
ax.tick_params(labelsize=8)
ax.xaxis.label.set_size(9)
ax.yaxis.label.set_size(9)
ax.grid(axis='y', alpha=0.2)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper left', frameon=False, fontsize=9)
fig.savefig(figures / 'part1_accuracy.png', dpi=300)
plt.close(fig)
for name in ('sentence1_layer1_head1.png', 'sentence2_layer1_head1.png'):
    shutil.copy2(run / 'figures' / name, figures / ('part1_' + name))
runpy.run_path(str(ROOT / 'scripts/build_encoder_attention_figure.py'), run_name='__main__')
print('Regenerated report/generated/part1_*.tex and report/figures/part1_*.png')
