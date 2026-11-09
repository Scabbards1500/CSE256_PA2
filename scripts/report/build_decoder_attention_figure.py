"""Build the report's three-panel decoder attention figure from the final checkpoint."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if (ROOT / '.local_packages').is_dir():
    sys.path.insert(0, str(ROOT / '.local_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from cse256_pa2.train import load_texts
from cse256_pa2.tokenizer import SimpleTokenizer
from cse256_pa2.transformer import DecoderLanguageModel

run = ROOT / 'outputs/part2'
saved = torch.load(run / 'model.pt', map_location='cpu')
config = saved['config']
tokenizer = SimpleTokenizer(' '.join(load_texts(ROOT / 'speechesdataset')))
assert tokenizer.stoi == saved['tokenizer_stoi']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DecoderLanguageModel(
    tokenizer.vocab_size, block_size=config['block_size'],
    n_embd=config['n_embd'], n_head=config['n_head'], n_layer=config['n_layer'],
    n_hidden=config['decoder_ffn_hidden'],
).to(device).eval()
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
        assert 0 < len(ids) <= config['block_size'] and 1 not in ids
        x = torch.tensor([ids + [0]*(config['block_size']-len(ids))], device=device)
        _, maps = model.decoder(x)
        max_error = 0.0
        for attention in maps:
            sums = attention.sum(-1)
            assert torch.isfinite(attention).all()
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0)
            assert torch.count_nonzero(attention[..., len(ids):]) == 0
            assert torch.count_nonzero(attention.triu(diagonal=1)) == 0
            max_error = max(max_error, float((sums-1).abs().max()))
        array = maps[0][0, 0, :len(ids), :len(ids)].cpu().numpy()
        if i <= 2:
            # Preserve the previously reported attention values in the first two panels.
            original = np.load(run / f'figures/sentence{i}_layer1_head1.npy')
            np.testing.assert_allclose(array, original, atol=1e-6, rtol=1e-5)
            array = original
        else:
            np.save(run / 'figures/sentence3_layer1_head1.npy', array)
        tokens = [tokenizer.itos[index] for index in ids]
        records.append({'sentence': sentence, 'tokens': tokens,
                        'max_row_sum_error': max_error, 'padded_key_attention': 0.0,
                        'future_key_attention': 0.0, 'layers_checked': len(maps),
                        'heads_per_layer': maps[0].shape[1]})
        arrays.append(array)

# Match the encoder figure's fixed panel positions and shared color scale.
# Fixed positions keep the gaps equal despite different word-label lengths.
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
    # Layer/head information appears once in the caption rather than above panels.
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
fig.savefig(ROOT / 'report/figures/part2_attention_examples.png', dpi=350,
            bbox_inches='tight', pad_inches=.02)
plt.close(fig)
(run / 'attention_examples.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
print('Three-panel figure saved; maximum row-sum error:', max(r['max_row_sum_error'] for r in records))
array, tokens = arrays[2], records[2]['tokens']
for row, col in sorted(np.ndindex(array.shape), key=lambda ij: array[ij], reverse=True)[:12]:
    print(f'{tokens[row]} -> {tokens[col]}: {array[row, col]:.6f}')
