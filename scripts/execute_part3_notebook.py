"""Execute the notebook with this interpreter, saving outputs after each cell."""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'part3_pythia.ipynb'
nb = nbformat.read(path, as_version=4)
print('Preparing notebook execution with', sys.executable, flush=True)
# Read public model weights from the already populated project cache.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('IPYTHONDIR', str(ROOT / '.cache/ipython'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR', str(ROOT / '.cache/jupyter'))
for key in ['MPLCONFIGDIR', 'IPYTHONDIR', 'JUPYTER_RUNTIME_DIR']:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
print('Kernel command configured.', flush=True)

def saved_cell(cell, cell_index, **kwargs):
    nbformat.write(nb, path)
    print(f'Completed cell {cell_index + 1}/{len(nb.cells)}', flush=True)
    for output in cell.get('outputs', []):
        if output.output_type == 'stream':
            # Notebook retains full logs, while terminal progress remains concise.
            value = output.text
            if len(value) > 3000:
                value = value[-3000:]
            print(value, end='' if value.endswith('\n') else '\n', flush=True)

client = NotebookClient(nb, km=km, timeout=1800, allow_errors=False,
                        resources={'metadata': {'path': str(ROOT)}},
                        on_cell_executed=saved_cell)
try:
    print('Starting notebook kernel...', flush=True)
    client.execute()
finally:
    nbformat.write(nb, path)
print('Executed notebook saved:', path, flush=True)
