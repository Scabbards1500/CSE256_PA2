"""Validate saved Part 3 outputs and export provenance without recomputing models."""
import hashlib
import json
from pathlib import Path
import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
nb_path = ROOT / 'part3_pythia.ipynb'
nb = nbformat.read(nb_path, 4)
r = json.loads((ROOT/'outputs/part3/results.json').read_text())
code = [c for c in nb.cells if c.cell_type == 'code']
assert [c.execution_count for c in code] == list(range(1, len(code)+1))
assert not any(o.output_type == 'error' for c in code for o in c.outputs)
images = [o for c in code for o in c.outputs if 'image/png' in o.get('data', {})]
assert len(images) == 5, f'Expected five embedded figures, got {len(images)}'
assert sum('image/png' in o.get('data', {}) for o in nb.cells[7].outputs) == 2
assert not any('Interpretation is filled' in c.source for c in nb.cells)
for heading in ['3.1 Reading Pythia', '3.2.1 Watching', '3.2.2 Comparing', '3.3 Transplanting']:
    assert any(heading in c.source for c in nb.cells)
text = '\n'.join(o.get('text', '') for c in code for o in c.outputs)
assert 'name' not in r
assert nb.cells[0].cell_type == 'code'
assert 'print("Name: Hanling Wang")' in nb.cells[0].source
first_output = ''.join(o.get('text', '') for o in nb.cells[0].outputs)
assert 'Name: Hanling Wang' in first_output.splitlines()
assert f"torch: {r['torch_version']}" in text
assert f"transformers: {r['transformers_version']}" in text
for depth in [0,6,12]: assert f'Layer {depth}:' in text
assert r['target_count'] == 6016
assert set(r['checkpoint_commits']) == {'step0','step512','step1000','step8000','step32000','step143000'}
for v in r['lens'].values():
    assert len(v['correct']) == len(v['accuracy']) == 13
    assert v['targets'] == r['target_count']
    np.testing.assert_allclose(np.array(v['correct'])/v['targets'], v['accuracy'], rtol=0, atol=1e-15)
assert np.array(r['deltas']).shape == (14,4)
np.testing.assert_allclose(np.array(r['transplant_losses'])-r['baseline_loss'], r['deltas'], atol=1e-12)
assert r['restoration_checks'] == 56
assert r['baseline_loss'] == r['restored_loss']
assert r['normalization_check']['direct_max_error'] == 0
assert r['normalization_check']['double_norm_max_error'] > 1
source_hash = hashlib.sha256(nb_path.read_bytes()).hexdigest()
html = (ROOT/'outputs/part3/part3_notebook.html').read_text(encoding='utf-8')
assert source_hash in html, 'HTML export is stale relative to notebook'
pdf = ROOT/'report/part3_notebook.pdf'
assert pdf.exists() and pdf.stat().st_size > 0
manifest = {'notebook_sha256':source_hash,
            'notebook_pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
            'results_sha256':hashlib.sha256((ROOT/'outputs/part3/results.json').read_bytes()).hexdigest(),
            'executed_code_cells':len(code),'embedded_pngs':len(images),
            'restoration_checks':r['restoration_checks'],'status':'passed'}
(ROOT/'outputs/part3/validation.json').write_text(json.dumps(manifest, indent=2),encoding='utf-8')
print(json.dumps(manifest, indent=2))
