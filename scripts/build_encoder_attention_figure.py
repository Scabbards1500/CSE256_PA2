"""Build the report's three-panel encoder attention figure from the final checkpoint."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / '.local_packages').is_dir():
    sys.path.insert(0, str(ROOT / '.local_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from main import load_texts
from tokenizer import SimpleTokenizer
from transformer import SpeakerClassifier, TransformerEncoder

run = ROOT / 'outputs/part1'
saved = torch.load(run / 'model.pt', map_location='cpu')
tokenizer = SimpleTokenizer(' '.join(load_texts(ROOT / 'speechesdataset')))
assert tokenizer.stoi == saved['tokenizer_stoi']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpeakerClassifier(TransformerEncoder(tokenizer.vocab_size)).to(device).eval()
model.load_state_dict(saved['model_state_dict'])
sentences = [
    'They have not served a Red America or a Blue America',
    'We must never give up on the American people.',
    'Our nation will build a better future for all.',
]
records, arrays = [], []
with torch.no_grad():
    for i, sentence in enumerate(sentences, start=1):
        ids = tokenizer.encode(sentence)
        assert 0 < len(ids) <= 32 and 1 not in ids
        x = torch.tensor([ids + [0]*(32-len(ids))], device=device)
        _, maps = model.encoder(x)
        max_error = 0.0
        for attention in maps:
            sums = attention.sum(-1)
            assert torch.isfinite(attention).all()
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0)
            assert torch.count_nonzero(attention[...,len(ids):]) == 0
            max_error = max(max_error, float((sums-1).abs().max()))
        array = maps[0][0,0,:len(ids),:len(ids)].cpu().numpy()
        if i <= 2:
            # The first two panels must retain the previously reported data.
            original = np.load(run / f'figures/sentence{i}_layer1_head1.npy')
            np.testing.assert_allclose(array, original, atol=1e-6, rtol=1e-5)
            array = original
        else:
            np.save(run / 'figures/sentence3_layer1_head1.npy', array)
        tokens = [tokenizer.itos[index] for index in ids]
        records.append({'sentence':sentence, 'tokens':tokens,
                        'max_row_sum_error':max_error, 'padded_key_attention':0.0,
                        'layers_checked':len(maps), 'heads_per_layer':maps[0].shape[1]})
        arrays.append(array)

# Fixed physical positions keep both heatmap gaps identical regardless of
# different word-label lengths. Automatic constrained layout changes them.
figure_width, figure_height = 7.1, 2.55
panel_size, panel_gap = 1.63, 0.56
left, bottom = 0.50, 0.22
fig = plt.figure(figsize=(figure_width, figure_height))
axes = []
for i, (record, array) in enumerate(zip(records, arrays)):
    ax = fig.add_axes([(left + i*(panel_size+panel_gap))/figure_width,
                       bottom/figure_height, panel_size/figure_width,
                       panel_size/figure_height])
    axes.append(ax)
    plot = ax.imshow(array, cmap='viridis', vmin=0, vmax=1, aspect='equal')
    tokens = record['tokens']
    ax.set_xticks(range(len(tokens)), tokens, rotation=55, ha='left', fontsize=6.5)
    ax.set_yticks(range(len(tokens)), tokens, fontsize=6.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=2, pad=1)
    ax.set_xlabel(f'({chr(97+i)})', fontsize=8, labelpad=6)
    # Layer/head information appears once in the figure caption, not in titles.
colorbar_axes = fig.add_axes([6.64/figure_width, bottom/figure_height,
                              .06/figure_width, panel_size/figure_height])
colorbar = fig.colorbar(plot, cax=colorbar_axes)
colorbar.ax.tick_params(labelsize=6.5, length=2)
colorbar.set_label('Attention probability', fontsize=7, labelpad=4)
fig.canvas.draw()
panel_bounds = [ax.get_window_extent() for ax in axes]
gaps = [panel_bounds[i+1].x0-panel_bounds[i].x1 for i in range(2)]
assert np.isclose(gaps[0], gaps[1], atol=1e-8)
for previous, current in zip(panel_bounds, axes[1:]):
    label_left = min(label.get_window_extent().x0 for label in current.get_yticklabels())
    assert label_left > previous.x1 + 2, 'Word labels must clear the preceding panel.'
print(f'Equal heatmap gaps: {gaps[0]/fig.dpi:.2f} and {gaps[1]/fig.dpi:.2f} inches')
fig.savefig(ROOT / 'report/figures/part1_attention_examples.png', dpi=350,
            bbox_inches='tight', pad_inches=.02)
plt.close(fig)
(run / 'attention_examples.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
print('Three-panel figure saved; maximum row-sum error:', max(r['max_row_sum_error'] for r in records))
array = arrays[2]
tokens = records[2]['tokens']
for row,col in sorted(np.ndindex(array.shape), key=lambda ij: array[ij], reverse=True)[:6]:
    print(f'{tokens[row]} -> {tokens[col]}: {array[row,col]:.6f}')
