"""Fill Markdown interpretations from computed outputs and improve figure legibility."""
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'part3_pythia.ipynb'
nb = nbformat.read(path, 4)
r = json.loads((ROOT / 'outputs/part3/results.json').read_text())
a = r['lens']['step143000']['accuracy']
curves = r['lens']
d = r['deltas']
obama = r['grids']['obama']['true_next_probability']
custom = r['grids']['custom']['true_next_probability']

analysis321 = f'''**What the predictions show.** Over all {r['target_count']:,} next-token targets, the final
checkpoint scores **{100*a[0]:.8f}% at depth 0**, **{100*a[6]:.8f}% at depth 6**, and
**{100*a[12]:.8f}% at depth 12**; the exact counts and all 13 values are printed above.
Overall readability increases toward the output, although the curve is not strictly monotonic
(e.g., depth 2 is below depth 1).

The grids also show that an individual prediction need not improve monotonically. After `Blue`,
the Obama sentence's true next token `_America` is already top-ranked at depth 8; its probability
is {obama[8][-1]:.4f}, rises to {obama[10][-1]:.4f} at depth 10, and ends at {obama[12][-1]:.4f}.
In the custom sentence, `_we` after `if` is top-ranked at depth 3 with probability
{custom[3][6]:.4f}, is no longer top-ranked by depth 6 (probability {custom[6][6]:.4f}), and returns
at later depths, ending at {custom[12][6]:.4f}. Thus intermediate readouts are provisional,
not a sequence of increasingly confident final answers. Some final predictions also remain
wrong, as the grids show. The last sentence token is used only as a target, so each 11-token
sentence gives 10 scored columns; no unobserved continuation is invented. Underscores show
spaces, and long token labels wrap within a cell only for readability.'''

def pct(revision, depth):
    return 100*curves[revision]['accuracy'][depth]
analysis322 = f'''**Middle-layer readability is nonmonotonic and becomes worse late in training.**
At depth 6, accuracy changes from {pct('step512',6):.4f}% (step512) to
{pct('step1000',6):.4f}% (step1000), {pct('step8000',6):.4f}% (step8000), then only
{pct('step143000',6):.4f}% (step143000). Depth 8 first improves from
{pct('step512',8):.4f}% to {pct('step1000',8):.4f}% and {pct('step8000',8):.4f}%,
then falls to {pct('step143000',8):.4f}%. Depth 4 decreases throughout these four checkpoints:
{pct('step512',4):.4f}%, {pct('step1000',4):.4f}%, {pct('step8000',4):.4f}%,
and {pct('step143000',4):.4f}%.

In contrast, final-depth accuracy rises from {pct('step512',12):.4f}% to
{pct('step1000',12):.4f}%, {pct('step8000',12):.4f}%, and {pct('step143000',12):.4f}%.
The trained network therefore improves its final predictions while making the middle states
less directly readable by its own final output head. This is consistent with the middle states
and final readout becoming less aligned: they need not speak the same representational
“language.” It does **not** show that the middle layers contain less useful information.
Both the hidden states and the checkpoint's output head/final LayerNorm change across training,
so these curves do not isolate which change causes the mismatch. A learned layer-specific
translation, as in the [tuned lens](https://arxiv.org/abs/2303.08112), addresses that limitation;
no tuned lens is trained in this assignment.'''
idx = {name:i for i,name in enumerate(r['components'])}
analysis33 = f'''**Random replacements are most tolerated in the middle, but none is harmless.**
The intact model's loss is **{r['baseline_loss']:.10f} nats/token**. Among step0 replacements,
`block7` has the smallest increase (**+{d[idx['block7']][0]:.6f}**), followed by `block8`
(+{d[idx['block8']][0]:.6f}) and `block6` (+{d[idx['block6']][0]:.6f}). Even the smallest
change raises loss to {r['transplant_losses'][idx['block7']][0]:.6f}, so “little damage” is
relative to the other replacements, not no damage. This experiment also replaces one block
at a time; it does not show that these blocks could all be removed together.

The **unembedding is most sensitive**: replacing it with step0 weights raises loss by
**+{d[idx['unembedding']][0]:.6f}**. The input embedding (+{d[idx['embedding']][0]:.6f}),
first block (+{d[idx['block0']][0]:.6f}), and last block (+{d[idx['block11']][0]:.6f})
are also substantially more sensitive than the middle blocks in this test. These boundary
components are especially difficult to replace with random weights while preserving the
final model's behavior.

**A later donor does not always hurt less.** For the unembedding, the loss increases are
+{d[idx['unembedding']][0]:.6f} (step0), +{d[idx['unembedding']][1]:.6f} (step1000),
+{d[idx['unembedding']][2]:.6f} (step8000), and +{d[idx['unembedding']][3]:.6f} (step32000):
every later donor in this set hurts more. This is not limited to the output head.
For `block10`, step8000 causes +{d[idx['block10']][2]:.6f}, whereas step32000 causes
+{d[idx['block10']][3]:.6f}. For `block4`, step8000 causes +{d[idx['block4']][2]:.6f},
far above step1000's +{d[idx['block4']][1]:.6f}, before falling to
+{d[idx['block4']][3]:.6f} at step32000. Later training of a component therefore does not
ensure greater compatibility with the *unchanged final-model surroundings*. The results
are consistent with co-adaptation of components, but do not determine whether representation
alignment, weight scale, or another mechanism causes each failure. All 56 restoration checks
pass and the restored baseline matches, ruling out accumulated changes between transplants.
Heatmap colors are logarithmic; numeric annotations are the original signed loss differences,
not the color-clipped values.'''
analyses = {'analysis_321':analysis321, 'analysis_322':analysis322, 'analysis_33':analysis33}
for cell in nb.cells:
    for tag in cell.metadata.get('tags', []):
        if tag in analyses:
            cell.source = analyses[tag]
    if cell.cell_type == 'code' and 'def plot_lens_grid' in cell.source:
        cell.source = cell.source.replace('def visible_token(index):', 'import textwrap\n\ndef visible_token(index):')
        cell.source = cell.source.replace('ax.text(pos, depth, visible_token(predictions[depth, pos]),',
            'label = "\\n".join(textwrap.wrap(visible_token(predictions[depth, pos]), width=11))\n            ax.text(pos, depth, label,')
    if cell.cell_type == 'code' and 'COLOR_FLOOR = ' in cell.source:
        cell.source = cell.source.replace('COLOR_FLOOR = 1e-4', 'COLOR_FLOOR = 0.1')
        cell.source = cell.source.replace('log color scale; floor 0.0001', 'log color scale; floor 0.1')
        cell.source = cell.source.replace('                print(case)\n', '')
        cell.source += '''\nprint("Number of later-donor counterexamples:", len(nonmonotonic))
for case in sorted(nonmonotonic, key=lambda c: c["extra_damage"], reverse=True)[:5]:
    print(f"{case['component']} {case['earlier']} -> {case['later']}: "
          f"delta {case['earlier_delta']:.8f} -> {case['later_delta']:.8f}")'''
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
nbformat.write(nb, path)
print('Filled analyses from actual run and adjusted label wrapping; rerun to produce final outputs.')
