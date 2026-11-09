"""Attention normalization checks and non-interactive, labelled heatmaps."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    @torch.no_grad()
    def sanity_check(self, sentence, block_size, output_dir="outputs/attention",
                     prefix="attention", selections=((0, 0),),
                     model_name="Encoder", causal=False):
        wordids = self.tokenizer.encode(sentence)[:block_size]
        if not wordids:
            raise ValueError("An attention example cannot be empty.")
        device = next(self.model.parameters()).device
        inputs = torch.tensor([wordids + [0] * (block_size - len(wordids))], device=device)
        was_training = self.model.training
        self.model.eval()
        _, maps = self.model(inputs)
        self.model.train(was_training)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        errors = []
        future_max = 0.0
        for layer, attention in enumerate(maps):
            sums = attention.sum(dim=-1)
            if not torch.isfinite(attention).all() or not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0):
                raise AssertionError(f"Attention normalization failed in layer {layer + 1}")
            errors.append((sums - 1).abs().max().item())
            if causal:
                future = attention.triu(diagonal=1).abs().max().item()
                future_max = max(future_max, future)
                if future != 0.0:
                    raise AssertionError("Future tokens received causal attention probability.")
            if len(wordids) < block_size and attention[..., len(wordids):].abs().max().item() != 0:
                raise AssertionError("Padded keys received attention probability.")
        tokens = [self.tokenizer.itos[index] for index in wordids]
        for layer, head in selections:
            array = maps[layer][0, head, :len(wordids), :len(wordids)].cpu().numpy()
            np.save(output_dir / f"{prefix}_layer{layer+1}_head{head+1}.npy", array)
            fig, ax = plt.subplots(figsize=(6.6, 5.6), constrained_layout=True)
            plot = ax.imshow(array, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(tokens)), tokens, rotation=55, ha="left", fontsize=9)
            ax.set_yticks(range(len(tokens)), tokens, fontsize=9)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_xlabel("Key (attended-to token)")
            ax.set_ylabel("Query token")
            ax.set_title(f"{model_name} layer {layer + 1}, head {head + 1}", pad=14)
            fig.colorbar(plot, ax=ax, label="Attention probability", shrink=0.8)
            fig.savefig(output_dir / f"{prefix}_layer{layer+1}_head{head+1}.png", dpi=180)
            plt.close(fig)
        print(f"Attention checks passed: {len(maps)} layers, {maps[0].shape[1]} heads; "
              f"max row-sum error {max(errors):.2e}", flush=True)
        return {"sentence": sentence, "tokens": tokens, "max_row_sum_error": max(errors),
                "layers_checked": len(maps), "heads_per_layer": maps[0].shape[1],
                "padded_key_attention": 0.0,
                "max_future_attention": future_max if causal else None}
