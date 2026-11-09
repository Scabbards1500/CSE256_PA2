"""Regenerate Part 2's report numbers, table and figures from saved metrics."""

import json
import runpy
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / 'outputs' / 'part2'
report = ROOT / 'report'
metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
history = metrics['history']
generated = report / 'generated'
figures = report / 'figures'
generated.mkdir(parents=True, exist_ok=True)
figures.mkdir(parents=True, exist_ok=True)

last = history[-1]
macros = [
    rf"\newcommand{{\PartTwoTrainPPL}}{{{last['train_perplexity']:.2f}}}",
    rf"\newcommand{{\PartTwoObamaPPL}}{{{last['obama_perplexity']:.2f}}}",
    rf"\newcommand{{\PartTwoWBushPPL}}{{{last['wbush_perplexity']:.2f}}}",
    rf"\newcommand{{\PartTwoHBushPPL}}{{{last['hbush_perplexity']:.2f}}}",
    rf"\newcommand{{\PartTwoDecoderParams}}{{{metrics['decoder_parameters']:,}}}",
    rf"\newcommand{{\PartTwoTotalParams}}{{{metrics['total_parameters']:,}}}",
]
(generated / 'part2_numbers.tex').write_text('\n'.join(macros) + '\n', encoding='utf-8')
rows = [r'\begin{tabular}{rrrrr}', r'\toprule',
        r'Step & Train & Obama & G.W. Bush & G.H.W. Bush \\', r'\midrule']
for row in history:
    rows.append(f"{row['step']} & {row['train_perplexity']:.2f} & {row['obama_perplexity']:.2f} & "
                f"{row['wbush_perplexity']:.2f} & {row['hbush_perplexity']:.2f} " + r'\\')
rows += [r'\bottomrule', r'\end{tabular}']
(generated / 'part2_perplexity_table.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')

# Match the full-width classification plot: annotate every checkpoint so the
# figure retains all values previously shown in the separate table.
fig, ax = plt.subplots(figsize=(7.1, 3.0), constrained_layout=True)
steps = [row['step'] for row in history]
for name, label, color, marker, offset, alignment in (
    ('train', 'Train (fixed subset)', '#2463a0', 'o', -7, 'top'),
    ('obama', 'Obama', '#b34c12', 's', -6, 'top'),
    ('wbush', 'George W. Bush', '#8552a1', '^', 6, 'bottom'),
    ('hbush', 'George H. W. Bush', '#2a806c', 'D', 4, 'bottom'),
):
    values = [row[f'{name}_perplexity'] for row in history]
    ax.plot(steps, values, color=color, marker=marker, label=label,
            markersize=3.5, linewidth=1.5)
    for step, value in zip(steps, values):
        ax.annotate(f'{value:.2f}', (step, value), xytext=(0, offset),
                    textcoords='offset points', ha='center', va=alignment,
                    fontsize=7.5, color=color)
ax.set(xlabel='Training step', ylabel='Perplexity (lower is better)',
       xticks=steps, xlim=(75, 530), ylim=(50, 900), yticks=range(100, 901, 100))
ax.tick_params(labelsize=8)
ax.xaxis.label.set_size(9)
ax.yaxis.label.set_size(9)
ax.grid(axis='y', alpha=0.2)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper right', frameon=False, fontsize=8, ncol=2)
fig.savefig(figures / 'part2_perplexity.png', dpi=300)
plt.close(fig)
for name in ('sentence1_layer1_head1.png', 'sentence2_layer1_head1.png'):
    shutil.copy2(run / 'figures' / name, figures / ('part2_' + name))

analysis_path = run / 'corpus_analysis.json'
if analysis_path.exists():
    corpora = json.loads(analysis_path.read_text(encoding='utf-8'))['corpora']
    rows = [r'\begin{tabular}{lrr}', r'\toprule',
            r'Test set & OOV (\%) & Unseen in LM train (\%) \\', r'\midrule']
    for name, label in [('obama', 'Obama'), ('wbush', 'G.W. Bush'), ('hbush', 'G.H.W. Bush')]:
        c = corpora[f'test_LM_{name}.txt']
        rows.append(f"{label} & {100*c['oov_rate']:.2f} & {100*c['absent_from_lm_train_rate']:.2f} " + r'\\')
    rows += [r'\bottomrule', r'\end{tabular}']
    (generated / 'part2_corpus_table.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')
runpy.run_path(str(ROOT / 'scripts/build_decoder_attention_figure.py'), run_name='__main__')
print('Regenerated report/generated/part2_*.tex and report/figures/part2_*.png')
