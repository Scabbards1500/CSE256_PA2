"""Behavioral checks for the scratch causal language model."""

import math
import unittest

import torch
from torch.nn import functional as F

from transformer import DecoderLanguageModel
from main import compute_perplexity


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = DecoderLanguageModel(40, block_size=8, n_embd=12,
                                          n_head=3, n_layer=2, n_hidden=20)
        self.ids = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])

    def test_future_tokens_cannot_change_prefix_predictions(self):
        modified = self.ids.clone()
        modified[:, 3:] = torch.tensor([[12, 13], [14, 15]])
        with torch.no_grad():
            original_logits, _ = self.model(self.ids)
            changed_logits, _ = self.model(modified)
        torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3],
                                   atol=1e-6, rtol=1e-5)
        self.assertGreater((original_logits[:, 3:] - changed_logits[:, 3:]).abs().max().item(), 1e-5)

    def test_attention_is_causal_and_normalized_with_right_padding(self):
        ids = torch.tensor([[2, 3, 4, 0, 0], [7, 8, 9, 10, 0]])
        logits, maps = self.model(ids)
        self.assertEqual(logits.shape, (2, 5, 40))
        self.assertEqual(len(maps), 2)
        for attention in maps:
            torch.testing.assert_close(attention.sum(-1), torch.ones(2, 3, 5))
            self.assertEqual(attention.triu(1).count_nonzero().item(), 0)
            padded_keys = ids.eq(0)[:, None, None, :].expand_as(attention)
            self.assertEqual(attention[padded_keys].count_nonzero().item(), 0)
            self.assertTrue(torch.isfinite(attention).all().item())
            torch.testing.assert_close(attention[:, :, 0, 0], torch.ones(2, 3))

    def test_full_prefix_matches_short_prefix_and_right_padding(self):
        with torch.no_grad():
            full, _ = self.model(self.ids)
            short, _ = self.model(self.ids[:, :3])
            padded, _ = self.model(F.pad(self.ids[:, :3], (0, 5)))
        torch.testing.assert_close(full[:, :3], short, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(padded[:, :3], short, atol=1e-6, rtol=1e-5)

    def test_cross_entropy_aligns_every_shifted_target(self):
        targets = torch.tensor([[3, 4, 5, 6, 12], [8, 9, 10, 11, 13]])
        logits, _ = self.model(self.ids)
        expected = torch.stack([
            -F.log_softmax(logits[b, t], dim=-1)[targets[b, t]]
            for b in range(2) for t in range(5)
        ]).mean()
        actual = self.model(self.ids, targets)
        self.assertEqual(actual.ndim, 0)
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            self.model(self.ids, targets[:, :-1])

    def test_loss_backpropagates_through_decoder_and_output_head(self):
        targets = torch.tensor([[3, 4, 5, 6, 12], [8, 9, 10, 11, 13]])
        self.model(self.ids, targets).backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertNotEqual(self.model.lm_head.weight.data_ptr(),
                            self.model.decoder.token_embedding.weight.data_ptr())

    def test_invalid_or_non_right_padded_inputs_fail_clearly(self):
        bad_inputs = [torch.zeros(1, 3, dtype=torch.long),
                      torch.ones(1, 9, dtype=torch.long),
                      torch.ones(3, dtype=torch.long),
                      torch.tensor([[0, 2, 3]]), torch.tensor([[2, 0, 3]])]
        for ids in bad_inputs:
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    self.model(ids)


class KnownTokenLossModel(torch.nn.Module):
    """A controlled per-token NLL source for testing evaluator aggregation."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.observed_modes = []
        self.observed_grad_modes = []

    def forward(self, inputs, targets):
        self.observed_modes.append(self.training)
        self.observed_grad_modes.append(torch.is_grad_enabled())
        return targets.float().mean() + self.anchor


class PerplexityTests(unittest.TestCase):
    def setUp(self):
        self.model = KnownTokenLossModel()
        # A full batch has six target tokens; the final batch has three.
        self.batches = [
            (torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3)),
            (torch.ones(1, 3, dtype=torch.long), torch.full((1, 3), 3.0)),
        ]

    def test_perplexity_uses_token_weighted_nll_for_unequal_batches(self):
        actual = compute_perplexity(self.model, self.batches)
        expected = math.exp((6 * 1.0 + 3 * 3.0) / 9)
        self.assertAlmostEqual(actual, expected, places=12)
        self.assertNotAlmostEqual(actual, math.exp((1.0 + 3.0) / 2), places=4)

    def test_perplexity_restores_mode_and_disables_gradients(self):
        for training in (False, True):
            with self.subTest(training=training):
                self.model.train(training)
                compute_perplexity(self.model, self.batches)
                self.assertEqual(self.model.training, training)
        self.assertFalse(any(self.model.observed_modes))
        self.assertFalse(any(self.model.observed_grad_modes))

    def test_batch_limit_evaluates_only_requested_batches(self):
        actual = compute_perplexity(self.model, self.batches, eval_iters=1)
        self.assertAlmostEqual(actual, math.e, places=12)
        self.assertEqual(len(self.model.observed_modes), 1)

    def test_empty_evaluation_raises_and_restores_mode(self):
        for training in (False, True):
            for batches, limit in (([], None), (self.batches, 0)):
                with self.subTest(training=training, limit=limit):
                    self.model.train(training)
                    with self.assertRaisesRegex(ValueError, "at least one target token"):
                        compute_perplexity(self.model, batches, eval_iters=limit)
                    self.assertEqual(self.model.training, training)

    def test_failing_evaluation_restores_mode(self):
        class FailingModel(KnownTokenLossModel):
            def forward(self, inputs, targets):
                raise RuntimeError("intentional test failure")

        model = FailingModel()
        for training in (False, True):
            with self.subTest(training=training):
                model.train(training)
                with self.assertRaisesRegex(RuntimeError, "intentional test failure"):
                    compute_perplexity(model, self.batches)
                self.assertEqual(model.training, training)


if __name__ == "__main__":
    unittest.main()
