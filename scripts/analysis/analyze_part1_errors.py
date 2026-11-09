"""Audit the final Part 1 checkpoint's full-test classification errors."""

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if (ROOT / ".local_packages").is_dir():
    sys.path.insert(0, str(ROOT / ".local_packages"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from cse256_pa2.dataset import SpeechesClassificationDataset
from cse256_pa2.train import collate_batch, load_texts
from cse256_pa2.tokenizer import SimpleTokenizer
from cse256_pa2.transformer import SpeakerClassifier, TransformerEncoder


# Verified against CSE256_PA2_FA26.pdf, page 2, Preliminaries / Data.
LABEL_NAMES = {0: "Barack Obama", 1: "George W. Bush", 2: "George H. W. Bush"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_confusion_figure(confusion, row_percent):
    # Initials keep two-line labels readable at the report's single-column width.
    labels = ["Barack\nObama", "G. W.\nBush", "G. H. W.\nBush"]
    fig, ax = plt.subplots(figsize=(3.35, 2.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    image = ax.imshow(row_percent, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(3), labels, fontsize=8)
    ax.set_yticks(range(3), labels, fontsize=8)
    ax.tick_params(length=0, pad=3)
    ax.set_xlabel("Predicted speaker", fontsize=9, labelpad=4)
    ax.set_ylabel("Actual speaker", fontsize=9, labelpad=4)
    for actual in range(3):
        for predicted in range(3):
            ax.text(predicted, actual,
                    f"{confusion[actual, predicted]}\n({row_percent[actual, predicted]:.1f}%)",
                    ha="center", va="center", fontsize=8,
                    color="white" if row_percent[actual, predicted] >= 55 else "#172b4d")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04, ticks=[0, 50, 100])
    colorbar.ax.tick_params(labelsize=8, length=2)
    colorbar.ax.set_yticklabels(["0%", "50%", "100%"])
    fig.canvas.draw()
    tick_bounds = [label.get_window_extent() for label in ax.get_xticklabels()]
    assert all(left.x1 < right.x0 for left, right in zip(tick_bounds, tick_bounds[1:])), \
        "Predicted-speaker labels must not overlap."
    figure_path = ROOT / "report" / "figures" / "part1_confusion.png"
    fig.savefig(figure_path, dpi=300, facecolor="white")
    plt.close(fig)
    return figure_path


def main():
    run = ROOT / "outputs" / "part1"
    checkpoint_path = run / "model.pt"
    test_path = ROOT / "speechesdataset" / "test_CLS.tsv"
    unchanged_paths = [checkpoint_path, run / "metrics.json", run / "vocabulary.json"]
    original_hashes = {str(path): sha256(path) for path in unchanged_paths}
    saved = torch.load(checkpoint_path, map_location="cpu")
    config, stoi = saved["config"], saved["tokenizer_stoi"]
    assert config["epochs"] == 15 and config["block_size"] == 32
    tokenizer = SimpleTokenizer(" ".join(load_texts(ROOT / "speechesdataset")))
    assert tokenizer.stoi == stoi, "Training-derived tokenizer must match the checkpoint."
    assert tokenizer.vocab_size == len(stoi)
    assert stoi["<pad>"] == 0 and stoi["<unk>"] == 1

    data = SpeechesClassificationDataset(tokenizer, test_path)
    assert len(data) == 750
    loader = DataLoader(data, batch_size=config["batch_size"], shuffle=False,
                        collate_fn=collate_batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = TransformerEncoder(tokenizer.vocab_size, config["block_size"],
                                 config["n_embd"], config["n_head"], config["n_layer"])
    model = SpeakerClassifier(encoder, config["n_embd"], config["classifier_hidden"], 3)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)

    records = []
    confusion = np.zeros((3, 3), dtype=np.int64)
    offset = 0
    with torch.inference_mode():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            assert torch.isfinite(logits).all()
            predictions = logits.argmax(dim=-1).cpu()
            probabilities = logits.softmax(dim=-1).cpu()
            for index, (actual, predicted) in enumerate(zip(labels.tolist(), predictions.tolist())):
                sample_label, text = data.samples[offset + index]
                assert sample_label == actual
                confusion[actual, predicted] += 1
                original_ids = tokenizer.encode(text)
                records.append({
                    "sample_index_1based": offset + index + 1,
                    "actual_label": actual, "actual_speaker": LABEL_NAMES[actual],
                    "predicted_label": predicted, "predicted_speaker": LABEL_NAMES[predicted],
                    "correct": actual == predicted,
                    "token_count_before_truncation": len(original_ids),
                    "tokens_retained": min(len(original_ids), config["block_size"]),
                    "p_obama": float(probabilities[index, 0]),
                    "p_george_w_bush": float(probabilities[index, 1]),
                    "p_george_h_w_bush": float(probabilities[index, 2]),
                    "text": text,
                })
            offset += labels.numel()

    correct = int(np.trace(confusion))
    total = int(confusion.sum())
    assert offset == total == len(records) == 750
    assert correct == 642
    accuracy = 100.0 * correct / total
    assert accuracy == 85.6
    original_metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert accuracy == original_metrics["final_test_accuracy"]
    actual_counts = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    assert (actual_counts > 0).all()
    row_percent = 100.0 * confusion / actual_counts[:, None]
    np.testing.assert_allclose(row_percent.sum(axis=1), np.full(3, 100.0))
    per_class = [{
        "label": label, "speaker": LABEL_NAMES[label],
        "actual_count": int(actual_counts[label]),
        "actual_percent": float(100.0 * actual_counts[label] / total),
        "predicted_count": int(predicted_counts[label]),
        "predicted_percent": float(100.0 * predicted_counts[label] / total),
        "correct_count": int(confusion[label, label]),
        "error_count": int(actual_counts[label] - confusion[label, label]),
        "recall_percent": float(row_percent[label, label]),
    } for label in range(3)]
    error_directions = [{
        "actual_label": actual, "actual_speaker": LABEL_NAMES[actual],
        "predicted_label": predicted, "predicted_speaker": LABEL_NAMES[predicted],
        "count": int(confusion[actual, predicted]),
        "percent_of_actual_class": float(row_percent[actual, predicted]),
        "percent_of_all_errors": float(100.0 * confusion[actual, predicted] / (total - correct)),
    } for actual in range(3) for predicted in range(3) if actual != predicted]
    error_directions.sort(key=lambda row: (-row["count"], row["actual_label"], row["predicted_label"]))

    # The source checkpoint, existing metrics, and vocabulary must remain untouched.
    assert all(sha256(Path(path)) == value for path, value in original_hashes.items())
    result = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "test_file": str(test_path), "test_sha256": sha256(test_path),
        "label_mapping_source": "CSE256_PA2_FA26.pdf, page 2, Preliminaries / Data",
        "label_mapping": LABEL_NAMES,
        "tokenizer_matches_saved_vocabulary": True,
        "vocabulary_sha256": original_hashes[str(run / "vocabulary.json")],
        "config": config, "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "dtype": str(next(model.parameters()).dtype),
        "eval_mode": not model.training, "gradient_tracking": False, "tf32_enabled": False,
        "total_examples": total, "correct_predictions": correct,
        "total_errors": total - correct, "accuracy_percent": accuracy,
        "confusion_orientation": "Rows = actual labels; columns = predicted labels; label order [0,1,2].",
        "confusion_counts": confusion.tolist(), "confusion_row_percent": row_percent.tolist(),
        "per_class": per_class, "error_directions": error_directions,
        "source_checkpoint_metrics_and_vocabulary_unchanged": True,
    }
    (run / "error_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions_path = run / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    figure_path = save_confusion_figure(confusion, row_percent)

    print("Confusion matrix (actual rows, predicted columns; labels 0,1,2):")
    print(confusion)
    print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%; errors: {total-correct}")
    for row in per_class:
        print(f"{row['speaker']}: recall {row['recall_percent']:.8f}% "
              f"({row['correct_count']}/{row['actual_count']}), predicted {row['predicted_count']}")
    print("Error directions:", json.dumps(error_directions))
    print("Saved error_analysis.json, predictions.csv, and", figure_path)


if __name__ == "__main__":
    main()
