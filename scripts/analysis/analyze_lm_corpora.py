"""Reproduce lexical corpus statistics used to discuss Part 2 perplexity."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
from nltk.tokenize import word_tokenize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cse256_pa2.tokenizer import SimpleTokenizer


def analyze(data_dir):
    training_text = " ".join(
        path.read_text(encoding="utf-8")
        for path in sorted(data_dir.iterdir())
        if path.is_file() and path.name.startswith("train_")
    )
    tokenizer = SimpleTokenizer(training_text)
    names = ["train_LM.txt", "test_LM_obama.txt", "test_LM_wbush.txt", "test_LM_hbush.txt"]
    words = {name: word_tokenize((data_dir / name).read_text(encoding="utf-8"))
             for name in names}
    ids = {name: [tokenizer.stoi.get(word, 1) for word in tokens]
           for name, tokens in words.items()}
    train_counts = np.bincount(ids["train_LM.txt"], minlength=tokenizer.vocab_size).astype(float)
    train_probs = train_counts / train_counts.sum()
    # Exclude padding from the smoothed distribution; retain the unknown token.
    smoothed = (train_counts[1:] + 1) / (train_counts.sum() + tokenizer.vocab_size - 1)
    train_words = set(words["train_LM.txt"])
    result = {
        "method": {
            "tokenizer": "Provided NLTK word_tokenize, case sensitive and punctuation retained. "
                         "Vocabulary built from sorted train_* files only, IDs assigned in sorted token order.",
            "vocab_size": tokenizer.vocab_size,
            "js_divergence": "Natural logarithms; unsmoothed normalized unigram histograms over encoded "
                             "token IDs; equal mixture of training and test distributions.",
            "unigram_cross_entropy": "Natural logarithms; add-one smoothed train_LM token counts over all "
                                     "non-padding vocabulary IDs (including <unk>), evaluated on each raw corpus token once.",
            "window_protocol": "Starter dataset: N-32 overlapping windows per corpus, 32 next-token targets "
                               "per window. Token counts below count each raw corpus token once; "
                               "corresponding sliding-window target counts also reported.",
        },
        "corpora": {},
    }
    for name in names:
        tokens, encoded = words[name], ids[name]
        counts = Counter(tokens)
        n = len(tokens)
        if n <= 32:
            raise ValueError(f"{name} needs at least 33 tokens for a complete LM window.")
        corpus_probs = np.bincount(encoded, minlength=tokenizer.vocab_size).astype(float)
        corpus_probs /= corpus_probs.sum()
        mixture = (train_probs + corpus_probs) / 2
        js = 0.5 * sum(float(np.sum(probs[probs > 0] * np.log(probs[probs > 0] / mixture[probs > 0])))
                       for probs in (train_probs, corpus_probs))
        cross_entropy = -float(np.log(smoothed[np.array(encoded) - 1]).mean())
        unknown = [word for word in tokens if word not in tokenizer.stoi]
        absent_lm = [word for word in tokens if word not in train_words]
        known_but_absent = [word for word in tokens if word in tokenizer.stoi and word not in train_words]
        # Target at position j occurs in starts i satisfying i+1 <= j <= i+32.
        positions = np.arange(n)
        multiplicities = np.maximum(0, np.minimum(positions - 1, n - 33)
                                   - np.maximum(0, positions - 32) + 1)
        window_targets = 32 * (n - 32)
        assert int(multiplicities.sum()) == window_targets
        result["corpora"][name] = {
            "tokens": n,
            "unique_surface_tokens": len(counts),
            "oov_tokens": len(unknown),
            "oov_rate": len(unknown) / n,
            "oov_unique_surface_tokens": len(set(unknown)),
            "absent_from_lm_train_tokens": len(absent_lm),
            "absent_from_lm_train_rate": len(absent_lm) / n,
            "known_vocab_but_absent_from_lm_train_tokens": len(known_but_absent),
            "known_vocab_but_absent_from_lm_train_rate": len(known_but_absent) / n,
            "windows_32": n - 32,
            "window_target_occurrences": window_targets,
            "window_oov_target_occurrences": int(multiplicities[np.array(encoded) == 1].sum()),
            "js_divergence_from_lm_train_nats": js,
            "add_one_unigram_cross_entropy_nats": cross_entropy,
            "add_one_unigram_perplexity": math.exp(cross_entropy),
            "top_20_surface_tokens": [{"token": word, "count": count, "rate": count / n}
                                      for word, count in counts.most_common(20)],
            "top_10_oov_surface_tokens": [{"token": word, "count": count}
                                          for word, count in Counter(unknown).most_common(10)],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "speechesdataset")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "part2" / "corpus_analysis.json")
    args = parser.parse_args()
    result = analyze(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved corpus analysis to {args.output}")


if __name__ == "__main__":
    main()
