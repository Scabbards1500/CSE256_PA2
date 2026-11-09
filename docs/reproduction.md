# Reproduction and experiment notes

Run commands from the repository root. See [the project overview](../README.md) for the directory layout.

CSE256 PA2 - Parts 1, 2 and 3 completed.

## NORMAL SETUP
Use Python 3.11 and an appropriate PyTorch build for your CPU/GPU.

```text
python -m pip install -r requirements.txt
python -m nltk.downloader punkt
```


TRAIN PART 1 (assignment defaults, 15 epochs)

```text
python main.py part1
```

Equivalent: python main.py
Optional: --device cpu, --seed 42, --epochs 15, --output-dir PATH.
The run uses train_CLS.tsv and train_LM.txt to construct the supplied word
vocabulary, trains only on train_CLS.tsv, and evaluates test_CLS.tsv.
The .tsv test file is named .txt in the handout; both names are supported.

ANALYZE PART 1 ERRORS (saved final checkpoint, no retraining)

```text
python scripts/analysis/analyze_part1_errors.py
```

Runs the saved classifier on all 750 test segments, verifies the reported
642 correct predictions, and generates per-class recall and the confusion
matrix used in the report. It preserves the checkpoint and original metrics.

TRAIN PART 2 (assignment defaults, 500 updates)

```text
python main.py part2
```

Optional: --device cpu, --seed 42, --max-iters 500, --output-dir PATH.
Uses the same tokenizer vocabulary; trains a fresh causal decoder only on
train_LM.txt. Test perplexity is evaluated after updates 100/200/300/400/500
over ALL stride-one windows from each politician's test_LM_*.txt file.
Target occurrences are token-weighted before exponentiating mean NLL;
interior corpus words occur in up to 32 overlapping target windows.
The training perplexity diagnostic uses 3,200 windows selected once with
seed 43 (200 batches), independently of the training sampler's RNG.

ANALYZE PART 2 ERRORS (saved final checkpoint, no retraining)

```text
python scripts/analysis/analyze_part2_errors.py
```

Computes top-1 next-token accuracy and mean NLL for three disjoint target
groups: LM-seen, in-vocabulary but LM-unseen, and OOV encoded as <unk>.
Uses the same overlapping-window evaluation as the report and verifies
the final perplexities. The report pools groups by target occurrence count.
The checkpoint, original metrics, and vocabulary are preserved.

## THIS WINDOWS MACHINE
Environment: D:\DevTools\Anaconda3-5.2.0\envs\pytorch311\python.exe
Python 3.11.8 / torch 2.2.2+cu121 / NVIDIA RTX 3070 Ti Laptop GPU.
The environment lacks NLTK, so .local_packages/nltk is a local copy of NLTK
3.8.1 from Anaconda base. It uses the existing user punkt data. No shared
Conda environment was modified. This workspace copy is not submission code.
Run with the wrapper, which temporarily adds that directory to PYTHONPATH:

```text
.\scripts\run_part1.ps1
.\scripts\run_part2.ps1
```

For tests or other commands in PowerShell:

```text
$env:PYTHONPATH = "$PWD\.local_packages"
& 'D:\DevTools\Anaconda3-5.2.0\envs\pytorch311\python.exe' main.py part1
```


## OUTPUTS
outputs/part1/history.csv : every epoch's training loss and train/test accuracy
outputs/part1/metrics.json : configuration, parameter counts, metrics, checks
outputs/part1/model.pt : FINAL epoch state_dict, config, and tokenizer mapping
outputs/part1/vocabulary.json : stable word-to-ID mapping
outputs/part1/error_analysis.json : confusion counts, recalls and source hashes
outputs/part1/predictions.csv : all 750 final-checkpoint test predictions
report/figures/part1_confusion.png : counts and row-normalized confusion matrix
outputs/part1/figures/ : attention heatmaps and exact plotted .npy matrices
outputs/part2/history.csv : train/test perplexities at each evaluation step
outputs/part2/metrics.json : configuration, parameter counts and causal checks
outputs/part2/model.pt : FINAL step-500 model, config and tokenizer mapping
outputs/part2/train_eval_indices.json : fixed training diagnostic windows
outputs/part2/figures/ : decoder attention heatmaps and their .npy matrices
outputs/part2/corpus_analysis.json : reproducible token coverage statistics
outputs/part2/error_analysis.json : next-token errors and NLL by target group
A rerun with the same output directory replaces its previous results.
Final checkpoints are epoch 15 and step 500, not chosen by test performance.
Seed 42 and sorted vocabulary IDs make runs repeatable on the same software
and hardware; bitwise equivalence across platforms is not guaranteed.

## MODEL
All attention and Transformer blocks are implemented from basic PyTorch
operations; no library Transformer layers are used. Four pre-LayerNorm
blocks, two heads, 64-dimensional embeddings, 256-wide ReLU encoder FFNs,
learned absolute position embeddings, residual connections, final LayerNorm.
Masked mean pooling excludes padding; classifier is 64 -> 100 -> 3 with ReLU.
Adam lr=0.001, no weight decay or dropout, batch=16, length=32, epochs=15.
Part 2 uses the same width/layer/head counts, a causal attention mask and
64 -> 100 -> 64 ReLU FFNs. Untied vocabulary output head at each position.
The decoder backbone has 489,168 parameters; with the head it has 863,243.
500 updates process 256,000 target-token occurrences in overlapping windows.

## CHECKS

```text
python -m unittest discover -s tests -v
```

Tests cover normalization, padded-key masking, padding invariance,
bidirectional context, gradients, and agreement with a reference attention
operation. The reference operation is used only by a test, never the model.
The run also checks attention at all four layers and both heads for two
fixed sentences and saves layer 1/head 1 plots for the report.
Decoder checks additionally cover exact zero future attention, unchanged
prefix logits after future-token perturbations, next-word target alignment,
and perplexity weighting/mode restoration with unequal-sized batches.

## REPORT

```text
python scripts/report/build_part1_report.py
python scripts/analysis/analyze_lm_corpora.py
python scripts/report/build_part2_report.py
```

Regenerates numeric tables/macros and figures from the respective outputs.
The prose in report/texs/A2_Experiment.tex and A2_Decoder.tex describes the
saved runs; review the analysis after changing the configuration or outputs.
Main LaTeX file: report/acl_latex.tex (compile from the report directory).
If a TeX distribution is installed:

```text
cd report
pdflatex acl_latex.tex
pdflatex acl_latex.tex
```

No terminal TeX installation was detected. The Codex editor's built-in
LaTeX compiler successfully compiled the combined Part A source.

## PART 3: EXECUTED NOTEBOOK
part3_pythia.ipynb is the self-contained, executed notebook, with all analysis
code, Markdown answers, exact numerical outputs and inline figures retained.
It uses EleutherAI/pythia-160m-deduped at step0/512/1000/8000/32000/143000.
All inference uses float32. The full Obama test file yields 6,017 subword
tokens and 6,016 next-token targets, each scored once in chunks with at most
256 inputs. Context restarts at each chunk; the boundary token is reused
as input, so no target is omitted. All checkpoints and transplants share
exactly this evaluation protocol. Do not compare its subword perplexity
directly with Part 2's word-level, overlapping-window perplexity.

The final hidden state already contains final LayerNorm; the notebook
verifies its readout equals the model logits. Earlier lens depths use the
checkpoint's own final LayerNorm and output head. Transplants replace only
one component, restore it in finally, check exact tensor equality after all
56 swaps, and verify full-model fingerprints and restored baseline loss.
The final LayerNorm is never transplanted. Block indices are zero-based;
lens depth 0 means embeddings and depths 1-12 mean block outputs.

PART 3 ENVIRONMENT (this machine)
.part3_env inherits PyTorch/GPU from pytorch311 but pins compatible notebook
and Hugging Face packages locally. The original Conda environment was not
modified. Python3.11.8 / torch2.2.2+cu121 / transformers4.39.3 / hub0.30.2.
To recreate from the project root in PowerShell:

```text
& 'D:\DevTools\Anaconda3-5.2.0\envs\pytorch311\python.exe' -m venv --system-site-packages .part3_env
& '.\.part3_env\Scripts\python.exe' -m pip install -r requirements/part3.txt
& '.\.part3_env\Scripts\python.exe' scripts/notebook/download_part3_models.py
```

Downloads are public model weights (~4 GB) cached in .cache/huggingface.
For a different machine, use your GPU-enabled Python environment instead.

## REEXECUTE PART 3

```text
.\scripts\run_part3.ps1
```

Or use the matching environment as the Jupyter kernel and Run All from the
project directory. The wrapper's execution script uses the current Python
interpreter explicitly, loads the prepared cache offline, and saves the
notebook after each code cell. Do not clear notebook outputs before submission.
If the model cache is missing, run download_part3_models.py first.
If rerunning with changed inputs/settings, update the Markdown explanations
to match the new numbers; scripts/notebook/finalize_part3_notebook.py is the initial
authoring helper and is not needed for an ordinary reproducibility rerun.

## PART 3 EXPORT AND AUDIT

```text
& '.\.part3_env\Scripts\python.exe' scripts/notebook/export_part3_notebook.py
& '.\.part3_env\Scripts\python.exe' scripts/notebook/validate_part3.py
```

Exports the saved notebook WITHOUT rerunning or hiding its code/outputs:

```text
report/part3_notebook.pdf : Part B, included after Part A by acl_latex.tex
outputs/part3/part3_notebook.html : standalone HTML from the same notebook
outputs/part3/results.json : computed results and checkpoint commit hashes
outputs/part3/validation.json : output checks and matching artifact hashes
outputs/part3/*.png : figure files also embedded in the notebook
```

The exporter prints the source notebook SHA256 for provenance and checks all
code cells executed without errors. It uses Playwright and an existing local
Chromium browser; on another machine install one with python -m playwright
install chromium, or pass --browser-executable PATH. Rerun export whenever
the saved notebook changes, then recompile report/acl_latex.tex.

## SCOPE AND SUBMISSION
Submit the executed part3_pythia.ipynb with the source code. The LaTeX main
document includes Part A followed by the PDF export of the SAME notebook.
Do not include .local_packages, .part3_env, .cache, __pycache__, or downloaded
model weights in submitted source code. Preserve report/generated,
report/figures and report/part3_notebook.pdf so the combined report compiles.

## Local report exports

The `report/`, `output/`, and `tmp/` directories and generated HTML under
`outputs/` are ignored by Git. The HTML is an intermediate notebook-to-PDF
export with embedded styles and figures, not a separate implementation.
Report compilation requires the local LaTeX sources; these are intentionally
not included in a fresh clone. The executed notebook and saved numerical
results remain versioned.
