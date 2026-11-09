"""Check future-token invariance of the saved final Part 2 checkpoint."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if (ROOT / ".local_packages").is_dir():
    sys.path.insert(0, str(ROOT / ".local_packages"))

import torch
from nltk.tokenize import word_tokenize

from cse256_pa2.transformer import DecoderLanguageModel


def main():
    checkpoint_path = ROOT / "outputs" / "part2" / "model.pt"
    output_path = ROOT / "outputs" / "part2" / "causality_verification.json"
    examples_path = ROOT / "outputs" / "part2" / "attention_examples.json"
    saved = torch.load(checkpoint_path, map_location="cpu")
    config, stoi = saved["config"], saved["tokenizer_stoi"]
    itos = {index: token for token, index in stoi.items()}
    assert stoi["<pad>"] == 0 and stoi["<unk>"] == 1
    assert sorted(itos) == list(range(len(stoi)))
    assert config["max_iters"] == 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = DecoderLanguageModel(
        len(stoi), block_size=config["block_size"], n_embd=config["n_embd"],
        n_head=config["n_head"], n_layer=config["n_layer"],
        n_hidden=config["decoder_ffn_hidden"],
    ).to(device).eval()
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.requires_grad_(False)
    assert not model.training

    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    assert len(examples) == 3
    atol, rtol = 1e-6, 1e-5
    cases = []
    sentence_records = []
    with torch.inference_mode():
        for sentence_index, example in enumerate(examples, start=1):
            tokens = word_tokenize(example["sentence"])
            assert tokens == example["tokens"]
            ids = [stoi[token] for token in tokens]
            length = len(ids)
            assert all(index >= 2 for index in ids)
            assert 1 < length <= config["block_size"]
            original = torch.tensor(
                [ids + [0] * (config["block_size"] - length)], device=device
            )
            original_logits, _ = model(original)
            assert torch.isfinite(original_logits).all()
            sentence_records.append({
                "sentence_index": sentence_index,
                "sentence": example["sentence"], "tokens": tokens,
                "real_token_count": length, "original_input_ids": original[0].tolist(),
                "case_count": length - 1,
            })
            # A prefix of k words predicts at positions 0..k-1; change only k..n-1.
            for prefix_length in range(1, length):
                changed = original.clone()
                future = changed[:, prefix_length:length]
                # Cycle through real vocabulary IDs, excluding padding and unknown.
                changed[:, prefix_length:length] = (future - 2 + 1) % (len(stoi) - 2) + 2
                prefix_unchanged = torch.equal(
                    original[:, :prefix_length], changed[:, :prefix_length]
                )
                padding_unchanged = torch.equal(original[:, length:], changed[:, length:])
                every_future_token_changed = bool(
                    (original[:, prefix_length:length] != changed[:, prefix_length:length]).all()
                )
                valid_replacements = bool(
                    ((changed[:, prefix_length:length] >= 2)
                     & (changed[:, prefix_length:length] < len(stoi))).all()
                )
                assert prefix_unchanged and padding_unchanged
                assert every_future_token_changed and valid_replacements
                changed_logits, _ = model(changed)
                reference = original_logits[:, :prefix_length]
                actual = changed_logits[:, :prefix_length]
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
                abs_error = (actual - reference).abs()
                allowed_error = atol + rtol * reference.abs()
                cases.append({
                    "sentence_index": sentence_index,
                    "prefix_length": prefix_length,
                    "changed_future_positions_zero_based": list(range(prefix_length, length)),
                    "replacement_tokens": [itos[index] for index in changed[0, prefix_length:length].tolist()],
                    "modified_input_ids": changed[0].tolist(),
                    "compared_logit_values": reference.numel(),
                    "max_absolute_logit_error": float(abs_error.max()),
                    "max_error_over_allowed_tolerance": float((abs_error / allowed_error).max()),
                    "prefix_unchanged": prefix_unchanged,
                    "padding_unchanged": padding_unchanged,
                    "every_future_token_changed": every_future_token_changed,
                    "valid_replacement_ids": valid_replacements,
                    "assert_close_passed": True,
                })

    assert [record["real_token_count"] for record in sentence_records] == [11, 10, 10]
    assert len(cases) == 28
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "training_updates_in_checkpoint": config["max_iters"],
        "torch_version": torch.__version__, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "dtype": str(next(model.parameters()).dtype),
        "eval_mode": not model.training, "gradient_tracking": False,
        "tf32_enabled": False, "absolute_tolerance": atol, "relative_tolerance": rtol,
        "replacement_rule": "For each future real token ID v: ((v - 2 + 1) mod (vocab_size - 2)) + 2.",
        "comparison_rule": "Compare all vocabulary logits at all positions strictly before prefix_length.",
        "sentence_count": len(sentence_records), "case_count": len(cases),
        "max_absolute_logit_error": max(case["max_absolute_logit_error"] for case in cases),
        "all_cases_passed": all(case["assert_close_passed"] for case in cases),
        "sentences": sentence_records, "cases": cases,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Final checkpoint causal perturbation checks: {len(cases)}/{len(cases)} passed")
    print(f"Maximum absolute prefix-logit error: {result['max_absolute_logit_error']:.12g}")
    print(f"Tolerance: atol={atol}, rtol={rtol}")
    print(f"Checkpoint SHA256: {result['checkpoint_sha256']}")
    print(f"Device: {result['device_name']}")
    print(f"Saved details: {output_path}")


if __name__ == "__main__":
    main()
