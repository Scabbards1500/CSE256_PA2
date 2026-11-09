"""Create the self-contained Part 3 notebook; never overwrite an executed run."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'part3_pythia.ipynb'
if path.exists():
    raise SystemExit('Notebook already exists; preserve its executed outputs.')
cells = []
def md(text, tag=None):
    cell = nbf.v4.new_markdown_cell(text.strip())
    if tag:
        cell.metadata['tags'] = [tag]
    cells.append(cell)
def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

code('''import torch
import transformers
print("Name: Hanling Wang")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)''')
md('''# Part B — Dissecting a Pretrained Pythia Language Model

All numerical results and figures below are computed by this notebook. Model:
[`EleutherAI/pythia-160m-deduped`](https://huggingface.co/EleutherAI/pythia-160m-deduped).
The model uses its own subword tokenizer, independently of Parts 1–2.
Lens depth 0 means embeddings; depths 1–12 mean outputs of successive blocks.
Transplant labels `block0`–`block11` use zero-based module indices.

## 3.1 Reading Pythia''')
md('''**1. Why checkpoints?** The term-frequency study depends most on the checkpoint design.
Section 3.3 tracks when term frequency becomes associated with task accuracy, using frequencies
counted only in the data seen by each checkpoint. Intermediate models reveal development over
training, while consistent data order controls comparisons across model sizes. The memorization
study in Section 3.2 could largely use final models plus the known training order: measure which
sequences they reproduce, then relate those sequences to their positions in the training corpus.
Its time axis describes corpus order, not necessarily repeated checkpoint evaluation.
[Biderman et al., §§3.2–3.3, pp. 7–9](https://proceedings.mlr.press/v202/biderman23a/biderman23a.pdf#page=7).''')
md('''**2. Data.** Deduplication did not produce a clear language-modeling performance benefit
in these experiments. Section 2.6 reports no consistent advantage for the deduplicated suite
in their evaluations. This is a finding about this experimental setup, not evidence that
deduplication never helps.
[Biderman et al., §2.6, p. 5](https://proceedings.mlr.press/v202/biderman23a/biderman23a.pdf#page=5).''')
md('''## Shared experimental setup
We use float32, evaluation mode, no gradients, and disable TF32. Every evaluation uses the same
Obama file and tokenizer. The file is tokenized as one string without adding special tokens.
We partition its next-token pairs into consecutive blocks of at most **256 predictions**.
Each block starts with its own context (no state is carried across blocks), and the boundary
input token is reused so that **every token after the first is scored exactly once**.
This defines a fixed, bounded-context evaluation over the whole test file, including its final
partial block. It is not a claim of unlimited-context likelihood. Loss is token-weighted mean
negative log probability, in natural-log units (nats). No test data is used for training.

The code loads each checkpoint's own output head and final LayerNorm for its lens. For transplants,
the recipient always remains the final model, and only the named component is changed.''')
code('''import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from IPython.display import display

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path.cwd()
if not (ROOT / "speechesdataset").exists():
    raise RuntimeError("Run the notebook from the assignment project directory.")
OUT = ROOT / "outputs" / "part3"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / ".cache" / "huggingface"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"
FINAL = "step143000"
LENS_REVISIONS = ["step512", "step1000", "step8000", FINAL]
DONORS = ["step0", "step1000", "step8000", "step32000"]
CONTEXT = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
plt.rcParams.update({"font.size": 10, "figure.dpi": 120})

# A populated cache works offline; uncached checkpoints are downloaded normally.
OFFLINE = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=FINAL,
    cache_dir=str(CACHE), local_files_only=OFFLINE)
corpus_path = ROOT / "speechesdataset" / "test_LM_obama.txt"
text = corpus_path.read_text(encoding="utf-8")
ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
N_TARGETS = ids.shape[1] - 1
assert N_TARGETS > 0
chunks = []
for start in range(0, N_TARGETS, CONTEXT):
    end = min(start + CONTEXT, N_TARGETS)
    chunks.append((ids[:, start:end].to(DEVICE), ids[:, start+1:end+1].to(DEVICE)))
assert sum(y.numel() for _, y in chunks) == N_TARGETS
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name())
print("Corpus SHA256:", hashlib.sha256(corpus_path.read_bytes()).hexdigest())
print("Corpus tokens:", ids.shape[1], "| Evaluated next-token targets:", N_TARGETS)
print("Context limit:", CONTEXT, "| Chunks:", len(chunks),
      "| Final chunk targets:", chunks[-1][1].numel())

checkpoint_commits = {}
def load_model(revision, device=DEVICE):
    model = GPTNeoXForCausalLM.from_pretrained(
        MODEL_NAME, revision=revision, cache_dir=str(CACHE),
        torch_dtype=torch.float32, attn_implementation="eager",
        local_files_only=OFFLINE,
    ).to(device).eval()
    model.config.use_cache = False
    model.requires_grad_(False)
    assert (model.config.num_hidden_layers, model.config.num_attention_heads,
            model.config.hidden_size) == (12, 12, 768)
    assert next(model.parameters()).dtype == torch.float32
    checkpoint_commits[revision] = model.config._commit_hash
    return model

def clean_memory():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

@torch.inference_mode()
def evaluate_loss(model):
    total_nll = 0.0
    count = 0
    for x, y in chunks:
        logits = model(x, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
        total_nll += float(loss)
        count += y.numel()
    assert count == N_TARGETS
    return total_nll / count

final_model = load_model(FINAL)
print("Architecture: 12 blocks, 12 heads, hidden size 768, float32")
print("Parameters:", sum(p.numel() for p in final_model.parameters()))
print("Final checkpoint commit:", checkpoint_commits[FINAL])''')
md('''## 3.2.1 Watching a Prediction Form
The logit lens applies the model's final output head to each hidden depth. At depths 0–11,
we first apply the final LayerNorm. We explicitly test whether the last returned hidden state
already includes that normalization by comparing reconstructed logits with the model output.
This avoids accidentally applying final LayerNorm twice.''')
code('''@torch.inference_mode()
def check_final_normalization(model):
    x = tok("They have not served a Red America or a Blue America",
            add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEVICE)
    output = model(x, output_hidden_states=True, use_cache=False)
    assert len(output.hidden_states) == 13
    head = model.get_output_embeddings()
    last = output.hidden_states[-1]
    direct = head(last)
    twice = head(model.gpt_neox.final_layer_norm(last))
    direct_error = float((direct - output.logits).abs().max())
    twice_error = float((twice - output.logits).abs().max())
    torch.testing.assert_close(direct, output.logits, atol=1e-5, rtol=1e-5)
    print(f"Head(last hidden) vs actual logits: max error = {direct_error:.9g}")
    print(f"Head(LN(last hidden)) vs actual logits: max error = {twice_error:.9g}")
    print("Confirmed: last hidden already includes final LayerNorm.")
    return {"direct_max_error": direct_error, "double_norm_max_error": twice_error}

normalization_check = check_final_normalization(final_model)

def lens_logits(model, hidden, depth):
    if depth < model.config.num_hidden_layers:
        hidden = model.gpt_neox.final_layer_norm(hidden)
    return model.get_output_embeddings()(hidden)

def visible_token(index):
    return tok.decode([int(index)]).replace(" ", "_").replace("\\n", "\\\\n").replace("\\t", "\\\\t")

@torch.inference_mode()
def plot_lens_grid(model, sentence, name):
    full = tok(sentence, add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEVICE)
    x, y = full[:, :-1], full[:, 1:]
    output = model(x, output_hidden_states=True, use_cache=False)
    predictions, probabilities = [], []
    for depth, hidden in enumerate(output.hidden_states):
        logits = lens_logits(model, hidden, depth)
        predictions.append(logits.argmax(-1).squeeze(0).cpu().numpy())
        log_prob = logits.log_softmax(-1).gather(-1, y.unsqueeze(-1))
        probabilities.append(log_prob.exp().squeeze(0).squeeze(-1).cpu().numpy())
    predictions, probabilities = np.stack(predictions), np.stack(probabilities)
    fig, ax = plt.subplots(figsize=(10.5, 6.4), constrained_layout=True)
    image = ax.imshow(probabilities, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for depth in range(13):
        for pos in range(x.shape[1]):
            ax.text(pos, depth, visible_token(predictions[depth, pos]), ha="center", va="center",
                    fontsize=8, color="white" if probabilities[depth, pos] > .55 else "black")
    ax.set_xticks(range(x.shape[1]), [visible_token(i) for i in x[0]], rotation=40, ha="left")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Input token (underscores mark spaces)")
    ax.set_yticks(range(13), ["emb (0)"] + [str(i) for i in range(1, 13)])
    ax.set_ylabel("Lens depth")
    ax.set_title("Final checkpoint: " + name, pad=14)
    fig.colorbar(image, ax=ax, label="Probability of the true next token", shrink=.8)
    fig.savefig(OUT / f"lens_grid_{name}.png", dpi=180)
    plt.show()
    print("Sentence:", sentence)
    print("Token count:", full.shape[1], "| Scored positions:", x.shape[1])
    print("True next tokens:", [visible_token(i) for i in y[0]])
    return {"sentence": sentence, "input_ids": x[0].tolist(), "target_ids": y[0].tolist(),
            "top_ids": predictions.tolist(), "true_next_probability": probabilities.tolist()}

sentences = ["They have not served a Red America or a Blue America",
             "We can build a better future if we work together."]
grid_results = {}
for name, sentence in zip(["obama", "custom"], sentences):
    grid_results[name] = plot_lens_grid(final_model, sentence, name)''')
code('''@torch.inference_mode()
def lens_accuracy(model):
    correct = np.zeros(13, dtype=np.int64)
    model_correct = 0
    count = 0
    for x, y in chunks:
        output = model(x, output_hidden_states=True, use_cache=False)
        for depth, hidden in enumerate(output.hidden_states):
            logits = lens_logits(model, hidden, depth)
            correct[depth] += int((logits.argmax(-1) == y).sum())
        model_correct += int((output.logits.argmax(-1) == y).sum())
        count += y.numel()
    assert count == N_TARGETS
    assert correct[-1] == model_correct
    return {"correct": correct.tolist(), "targets": count,
            "accuracy": (correct / count).tolist()}

lens_results = {FINAL: lens_accuracy(final_model)}
final_accuracy = np.array(lens_results[FINAL]["accuracy"])
for depth in [0, 6, 12]:
    correct = lens_results[FINAL]["correct"][depth]
    print(f"Layer {depth}: {correct}/{N_TARGETS} = {100*final_accuracy[depth]:.10f}%")
print("All 13 accuracy values (%):", (100 * final_accuracy).tolist())
fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
ax.plot(range(13), 100 * final_accuracy, "o-")
ax.set(xlabel="Lens depth (0 = embeddings)", ylabel="Next-token top-1 accuracy (%)",
       xticks=range(13), title="Final checkpoint on the whole Obama test file")
ax.grid(alpha=.25)
fig.savefig(OUT / "lens_final_accuracy.png", dpi=180)
plt.show()''')
md('''Interpretation is filled after inspecting the executed results.''', 'analysis_321')
md('''## 3.2.2 Comparing Checkpoints
We apply the same evaluation to step512, step1000, step8000 and step143000.
Each curve uses that checkpoint's own final LayerNorm and output head.''')
code('''for revision in LENS_REVISIONS[:-1]:
    model = load_model(revision)
    check_final_normalization(model)
    lens_results[revision] = lens_accuracy(model)
    print(revision, "accuracy by depth (%):",
          [round(100 * value, 6) for value in lens_results[revision]["accuracy"]])
    del model
    clean_memory()

fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
for revision in LENS_REVISIONS:
    ax.plot(range(13), 100*np.array(lens_results[revision]["accuracy"]), "o-", label=revision)
ax.set(xlabel="Lens depth (0 = embeddings)", ylabel="Next-token top-1 accuracy (%)",
       xticks=range(13), title="Logit-lens accuracy across pretraining")
ax.grid(alpha=.25)
ax.legend()
fig.savefig(OUT / "lens_checkpoints.png", dpi=180)
plt.show()
print("Checkpoint | depth 4 | depth 6 | depth 8 | depth 12 (accuracy %)")
for revision in LENS_REVISIONS:
    a = np.array(lens_results[revision]["accuracy"]) * 100
    print(revision, "|", " | ".join(f"{a[d]:.8f}" for d in [4, 6, 8, 12]))''')
md('''Interpretation is filled after inspecting the executed curves.''', 'analysis_322')
md('''## 3.3 Transplanting Layers Across Time
The recipient is step143000. For each donor, we replace one component at a time:
the token embedding, one complete Transformer block (indices 0–11), or the output
unembedding. A block includes both attention and MLP weights and its internal LayerNorms.
The recipient's final LayerNorm is always unchanged. We restore the replaced component
in a `finally` clause and check exact equality with the original weights after every pair.
A full-model fingerprint verifies that the recipient is intact before/after each donor sweep.
The reported change is **transplant test loss minus intact-final test loss**, in nats/token.''')
code('''COMPONENTS = ["embedding"] + [f"block{i}" for i in range(12)] + ["unembedding"]

def component(model, name):
    if name == "embedding":
        return model.get_input_embeddings()
    if name == "unembedding":
        return model.get_output_embeddings()
    return model.gpt_neox.layers[int(name.removeprefix("block"))]

def fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

baseline_loss = evaluate_loss(final_model)
print(f"Intact final-model baseline loss: {baseline_loss:.10f} nats/token")
print(f"Baseline perplexity: {np.exp(baseline_loss):.10f}")
print("Evaluated targets:", N_TARGETS)
original_fingerprint = fingerprint(final_model)
transplant_losses = np.empty((len(COMPONENTS), len(DONORS)))
restore_checks = 0
start_time = time.perf_counter()
for donor_index, revision in enumerate(DONORS):
    assert fingerprint(final_model) == original_fingerprint
    donor = load_model(revision, device=torch.device("cpu"))
    for row, name in enumerate(COMPONENTS):
        recipient_module = component(final_model, name)
        saved = {k: v.detach().cpu().clone() for k, v in recipient_module.state_dict().items()}
        try:
            recipient_module.load_state_dict(component(donor, name).state_dict(), strict=True)
            transplant_losses[row, donor_index] = evaluate_loss(final_model)
        finally:
            recipient_module.load_state_dict(saved, strict=True)
        assert all(torch.equal(value.detach().cpu(), saved[key])
                   for key, value in recipient_module.state_dict().items())
        restore_checks += 1
        delta = transplant_losses[row, donor_index] - baseline_loss
        print(f"{revision:>10s} {name:>11s}: loss={transplant_losses[row, donor_index]:.8f}, delta={delta:+.8f}")
    assert fingerprint(final_model) == original_fingerprint
    del donor
    clean_memory()

restored_loss = evaluate_loss(final_model)
assert abs(restored_loss - baseline_loss) < 1e-7
assert restore_checks == 56
assert np.isfinite(transplant_losses).all()
deltas = transplant_losses - baseline_loss
print(f"Restoration checks passed: {restore_checks}/56")
print(f"Restored final loss: {restored_loss:.10f}")
print(f"Transplant experiment elapsed: {time.perf_counter()-start_time:.1f} seconds")''')
code('''COLOR_FLOOR = 1e-4
fig, ax = plt.subplots(figsize=(8.0, 8.5), constrained_layout=True)
color_values = np.clip(deltas, COLOR_FLOOR, None)
image = ax.imshow(color_values, aspect="auto", cmap="YlOrRd",
                  norm=LogNorm(vmin=COLOR_FLOOR, vmax=max(float(color_values.max()), COLOR_FLOOR*10)))
ax.set_xticks(range(len(DONORS)), DONORS)
ax.set_yticks(range(len(COMPONENTS)), COMPONENTS)
ax.set_xlabel("Donor checkpoint")
ax.set_ylabel("Component replaced in final model")
ax.set_title("Transplant change in Obama test loss (nats/token)")
for row in range(len(COMPONENTS)):
    for col in range(len(DONORS)):
        color = "white" if image.norm(color_values[row, col]) > .72 else "black"
        ax.text(col, row, f"{deltas[row, col]:+.6f}", ha="center", va="center", fontsize=9, color=color)
fig.colorbar(image, ax=ax, label="Loss increase (log color scale; floor 0.0001)", shrink=.8)
fig.savefig(OUT / "transplant_heatmap.png", dpi=200)
plt.show()
print("Exact computed deltas as Python floats (rows=components, columns=donors):")
for name, row in zip(COMPONENTS, deltas):
    print(name, row.tolist())
print("Smallest step0 increase:", COMPONENTS[int(deltas[:, 0].argmin())], float(deltas[:, 0].min()))
print("Largest step0 increase:", COMPONENTS[int(deltas[:, 0].argmax())], float(deltas[:, 0].max()))
print("Cases where a later donor hurts more than an earlier donor:")
nonmonotonic = []
for row, name in enumerate(COMPONENTS):
    for earlier in range(len(DONORS)):
        for later in range(earlier+1, len(DONORS)):
            increase = float(deltas[row, later] - deltas[row, earlier])
            if increase > 1e-6:
                case = {"component": name, "earlier": DONORS[earlier], "later": DONORS[later],
                        "earlier_delta": float(deltas[row, earlier]),
                        "later_delta": float(deltas[row, later]), "extra_damage": increase}
                nonmonotonic.append(case)
                print(case)
if not nonmonotonic:
    print("None at tolerance 1e-6 nats/token.")''')
md('''Interpretation is filled after inspecting the executed heatmap.''', 'analysis_33')
md('''## Reproducibility record
The following cell saves the results computed above, including full-precision counts, losses,
checkpoint commit hashes, normalization checks and evaluation settings. The notebook outputs
and figures remain the authoritative submission; the JSON is an auxiliary audit record.''')
code('''results = {
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__, "model": MODEL_NAME,
    "checkpoint_commits": checkpoint_commits, "device": str(DEVICE),
    "dtype": "float32", "context_limit": CONTEXT, "target_count": N_TARGETS,
    "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    "normalization_check": normalization_check, "grids": grid_results,
    "lens": lens_results, "baseline_loss": baseline_loss,
    "restored_loss": restored_loss, "restoration_checks": restore_checks,
    "components": COMPONENTS, "donors": DONORS,
    "transplant_losses": transplant_losses.tolist(), "deltas": deltas.tolist(),
    "nonmonotonic_cases": nonmonotonic, "heatmap_color_floor": COLOR_FLOOR,
}
(OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("Checkpoint commits:", json.dumps(checkpoint_commits, indent=2))
''')
md('''## References
- Biderman et al. (2023). [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://proceedings.mlr.press/v202/biderman23a/biderman23a.pdf).
- nostalgebraist (2020). [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).
- Belrose et al. (2023). [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112).''')
nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3 (Part 3)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11.8'},
})
nbf.write(nb, path)
print('Created', path, 'with', len(cells), 'cells')
