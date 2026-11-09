"""Run Part 1 or Part 2 with the assignment's default hyperparameters."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset

from .dataset import SpeechesClassificationDataset, LanguageModelingDataset
from .tokenizer import SimpleTokenizer
from .transformer import SpeakerClassifier, TransformerEncoder
from .utilities import Utilities

ROOT = Path(__file__).resolve().parents[1]
seed = 42
batch_size = 16
block_size = 32
learning_rate = 1e-3
n_embd = 64
n_head = 2
n_layer = 4
n_input = 64
n_hidden = 100
n_output = 3
epochs_CLS = 15
# Decoder language-model training/evaluation settings.
eval_interval = 100
max_iters = 500
eval_iters = 200


def load_texts(directory):
    """Use training files only, as in the starter; never build on test text."""
    return [path.read_text(encoding="utf-8") for path in sorted(Path(directory).iterdir())
            if path.is_file() and path.name.startswith("train_")]


def collate_batch(batch):
    data, labels = zip(*batch)
    padded = pad_sequence([x[:block_size] for x in data], batch_first=True, padding_value=0)
    padded = nn.functional.pad(padded, (0, block_size - padded.shape[1]), value=0)
    return padded, torch.stack(labels)


@torch.no_grad()
def compute_classifier_accuracy(classifier, data_loader):
    was_training = classifier.training
    classifier.eval()
    device = next(classifier.parameters()).device
    correct = total = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        correct += (classifier(inputs).argmax(dim=-1) == labels).sum().item()
        total += labels.numel()
    classifier.train(was_training)
    return 100.0 * correct / total


def run_part1(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = SimpleTokenizer(" ".join(load_texts(args.data_dir)))
    train_data = SpeechesClassificationDataset(tokenizer, args.data_dir / "train_CLS.tsv")
    # The handout calls this .txt; the distributed file is actually .tsv.
    test_path = args.data_dir / "test_CLS.tsv"
    if not test_path.exists():
        test_path = args.data_dir / "test_CLS.txt"
    test_data = SpeechesClassificationDataset(tokenizer, test_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_batch,
                              generator=torch.Generator().manual_seed(args.seed))
    train_eval_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_batch)
    encoder = TransformerEncoder(tokenizer.vocab_size, block_size, n_embd, n_head, n_layer)
    model = SpeakerClassifier(encoder, n_input, n_hidden, n_output).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    encoder_parameters = sum(p.numel() for p in encoder.parameters())
    classifier_parameters = sum(p.numel() for p in model.classifier.parameters())
    print(f"Device: {device} | PyTorch: {torch.__version__}", flush=True)
    print(f"Vocabulary: {tokenizer.vocab_size} | Train: {len(train_data)} | Test: {len(test_data)}", flush=True)
    print(f"Encoder parameters: {encoder_parameters:,} | Classifier: {classifier_parameters:,}", flush=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
        row = {"epoch": epoch, "train_loss": loss_sum / len(train_data),
               "train_accuracy": compute_classifier_accuracy(model, train_eval_loader),
               "test_accuracy": compute_classifier_accuracy(model, test_loader)}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} | loss {row['train_loss']:.4f} | "
              f"train {row['train_accuracy']:.2f}% | test {row['test_accuracy']:.2f}%", flush=True)
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

    config = {"seed": args.seed, "batch_size": batch_size, "block_size": block_size,
              "learning_rate": learning_rate, "n_embd": n_embd, "n_head": n_head,
              "n_layer": n_layer, "encoder_ffn_hidden": 4 * n_embd,
              "classifier_hidden": n_hidden, "epochs": args.epochs,
              "dropout": 0.0, "optimizer": "Adam", "device": str(device)}
    torch.save({"model_state_dict": model.state_dict(), "config": config,
                "tokenizer_stoi": tokenizer.stoi}, output_dir / "model.pt")
    (output_dir / "vocabulary.json").write_text(json.dumps(tokenizer.stoi, indent=2), encoding="utf-8")
    utility = Utilities(tokenizer, encoder)
    # Fixed examples selected before inspecting attention or predictions.
    sentences = ["They have not served a Red America or a Blue America",
                 "We must never give up on the American people."]
    attention_checks = []
    for i, sentence in enumerate(sentences, start=1):
        attention_checks.append(utility.sanity_check(
            sentence, block_size, output_dir / "figures", f"sentence{i}",
            selections=((0, 0),),
        ))
    metrics = {"config": config, "torch_version": torch.__version__,
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "vocab_size": tokenizer.vocab_size, "train_examples": len(train_data),
               "test_examples": len(test_data), "test_file": test_path.name,
               "encoder_parameters": encoder_parameters,
               "classifier_parameters": classifier_parameters,
               "history": history, "final_test_accuracy": history[-1]["test_accuracy"],
               "attention_checks": attention_checks}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved final-epoch model, metrics and plots to {output_dir}", flush=True)


@torch.no_grad()
def compute_perplexity(model, data_loader, eval_iters=None):
    """Exponentiate token-weighted mean NLL; optionally limit evaluation batches."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    try:
        for i, (inputs, targets) in enumerate(data_loader):
            if eval_iters is not None and i >= eval_iters:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            loss = model(inputs, targets)
            count = targets.numel()
            total_nll += loss.item() * count
            total_tokens += count
    finally:
        model.train(was_training)
    if not total_tokens:
        raise ValueError("Perplexity needs at least one target token.")
    return math.exp(total_nll / total_tokens)


def run_part2(args):
    # Imported here so the classification entry point stays independent.
    from .transformer import DecoderLanguageModel

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = SimpleTokenizer(" ".join(load_texts(args.data_dir)))
    filenames = {"train": "train_LM.txt", "obama": "test_LM_obama.txt",
                 "wbush": "test_LM_wbush.txt", "hbush": "test_LM_hbush.txt"}
    datasets = {name: LanguageModelingDataset(
        tokenizer, (args.data_dir / filename).read_text(encoding="utf-8"), block_size
    ) for name, filename in filenames.items()}
    if any(len(data) < 1 for data in datasets.values()):
        raise ValueError("Each language-model corpus must contain more than 32 tokens.")
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    # Fixed random subset for the starter's 200-batch TRAIN diagnostic.
    # Separate RNGs keep evaluation from affecting the training sample order.
    train_eval_indices = torch.randperm(
        len(datasets["train"]), generator=torch.Generator().manual_seed(args.seed + 1)
    )[:eval_iters * batch_size].tolist()
    eval_loaders = {"train": DataLoader(Subset(datasets["train"], train_eval_indices),
                                       batch_size=batch_size)}
    # Test PPL covers every sliding window, including the final smaller batch.
    eval_loaders.update({name: DataLoader(data, batch_size=batch_size)
                         for name, data in datasets.items() if name != "train"})
    model = DecoderLanguageModel(tokenizer.vocab_size, block_size, n_embd,
                                 n_head, n_layer, n_hidden=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    decoder_parameters = sum(p.numel() for p in model.decoder.parameters())
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | PyTorch: {torch.__version__} | Vocabulary: {tokenizer.vocab_size}", flush=True)
    print(f"Decoder parameters: {decoder_parameters:,} | Including LM head: {total_parameters:,}", flush=True)
    print("Evaluation windows: " + str({name: len(loader.dataset)
                                        for name, loader in eval_loaders.items()}), flush=True)

    model.train()
    history = []
    iterator = iter(train_loader)
    interval_nll = 0.0
    interval_tokens = 0
    tokens_processed = 0
    for step in range(1, args.max_iters + 1):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, targets = next(iterator)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}")
        loss.backward()
        optimizer.step()
        count = targets.numel()
        interval_nll += loss.item() * count
        interval_tokens += count
        tokens_processed += count
        if step % eval_interval == 0 or step == args.max_iters:
            row = {"step": step, "interval_train_loss": interval_nll / interval_tokens}
            row.update({f"{name}_perplexity": compute_perplexity(model, loader)
                        for name, loader in eval_loaders.items()})
            history.append(row)
            print(f"Step {step:03d}/{args.max_iters} | " + " | ".join(
                f"{name} PPL {row[f'{name}_perplexity']:.2f}" for name in eval_loaders
            ), flush=True)
            with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            interval_nll = 0.0
            interval_tokens = 0

    config = {"seed": args.seed, "batch_size": batch_size, "block_size": block_size,
              "learning_rate": learning_rate, "n_embd": n_embd, "n_head": n_head,
              "n_layer": n_layer, "decoder_ffn_hidden": 100,
              "max_iters": args.max_iters, "eval_interval": eval_interval,
              "train_eval_max_batches": eval_iters, "test_eval": "all stride-1 windows",
              "dropout": 0.0, "optimizer": "Adam", "device": str(device),
              "weight_tying": False}
    torch.save({"model_state_dict": model.state_dict(), "config": config,
                "tokenizer_stoi": tokenizer.stoi}, output_dir / "model.pt")
    (output_dir / "vocabulary.json").write_text(json.dumps(tokenizer.stoi, indent=2), encoding="utf-8")
    (output_dir / "train_eval_indices.json").write_text(json.dumps(train_eval_indices), encoding="utf-8")
    utility = Utilities(tokenizer, model.decoder)
    sentences = ["They have not served a Red America or a Blue America",
                 "We must never give up on the American people."]
    attention_checks = [utility.sanity_check(
        sentence, block_size, output_dir / "figures", f"sentence{i}",
        selections=((0, 0),), model_name="Decoder", causal=True,
    ) for i, sentence in enumerate(sentences, start=1)]
    metrics = {"config": config, "torch_version": torch.__version__,
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "vocab_size": tokenizer.vocab_size, "decoder_parameters": decoder_parameters,
               "lm_head_parameters": total_parameters - decoder_parameters,
               "total_parameters": total_parameters, "training_tokens_processed": tokens_processed,
               "corpus_tokens": {name: len(data.data) for name, data in datasets.items()},
               "evaluation_windows": {name: len(loader.dataset) for name, loader in eval_loaders.items()},
               "evaluation_target_tokens": {name: len(loader.dataset) * block_size
                                             for name, loader in eval_loaders.items()},
               "history": history, "attention_checks": attention_checks}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved final-step model, metrics and plots to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", nargs="?", choices=["part1", "part2"], default="part1")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "speechesdataset")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--epochs", type=int, default=epochs_CLS)
    parser.add_argument("--max-iters", type=int, default=max_iters)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.max_iters < 1:
        parser.error("--max-iters must be positive")
    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / args.part
    (run_part1 if args.part == "part1" else run_part2)(args)


if __name__ == "__main__":
    main()
