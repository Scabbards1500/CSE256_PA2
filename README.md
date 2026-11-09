# CSE256 PA2 — Transformers and Pythia

Part 1 implements a Transformer speaker classifier, Part 2 a causal language
model, and Part 3 studies Pythia with logit lenses and component transplants.

## Project layout

| Path | Contents |
| --- | --- |
| `main.py` | Command-line entry point for Parts 1 and 2 |
| `cse256_pa2/` | Datasets, tokenization, Transformer models, training and attention utilities |
| `part3_pythia.ipynb` | Executed Part 3 notebook, including answers and figures |
| `scripts/analysis/` | Error analysis, corpus statistics and causality checks |
| `scripts/report/` | Report tables and figures generated from saved results |
| `scripts/notebook/` | Part 3 model download, notebook execution, export and validation |
| `scripts/run_part*.ps1` | Windows launchers for this machine |
| `speechesdataset/` | Assignment training and test data |
| `outputs/part1/`, `part2/`, `part3/` | Saved results, checkpoints and figures |
| `tests/` | Encoder, decoder and evaluation checks |
| `docs/` | Assignment handout and detailed reproduction notes |
| `requirements.txt`, `requirements/part3.txt` | Base and Part 3 dependencies |

## Run

Use Python 3.11 with a suitable PyTorch build. Run commands from the repository root.

```sh
python -m pip install -r requirements.txt
python -m nltk.downloader punkt
python main.py part1
python main.py part2
python -m unittest discover -s tests -v
```

For this Windows machine, the launchers select the existing Python environment
and local dependencies:

```powershell
.\scripts\run_part1.ps1
.\scripts\run_part2.ps1
.\scripts\run_part3.ps1
```

See [reproduction and experiment notes](docs/reproduction.md) for Part 3 setup,
evaluation details, saved outputs and report generation. The
[assignment handout](docs/CSE256_PA2_FA26.pdf) is included for reference.

## Report and HTML exports

`report/` stays local and is ignored by Git, as are `output/`, `tmp/`, environment
directories and downloaded model caches. The executed notebook and experiment
results remain in the repository.

`outputs/part3/part3_notebook.html` is a generated copy of the notebook used to
print its PDF. It embeds styles, scripts and images, which made one HTML file
account for a large amount of source text. HTML exports are now ignored and can
be regenerated with `scripts/notebook/export_part3_notebook.py`; no notebook
execution is needed to export the saved results. Local HTML files are preserved.
