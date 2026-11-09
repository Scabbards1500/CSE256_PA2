"""Behavioral checks for the scratch encoder (no training run required)."""

import unittest

import torch
from torch.nn import functional as F

from cse256_pa2.train import collate_batch, compute_classifier_accuracy
from cse256_pa2.transformer import MultiHeadSelfAttention, SpeakerClassifier, TransformerEncoder


class EncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.encoder = TransformerEncoder(40, block_size=8, n_embd=12, n_head=3, n_layer=2)
        self.model = SpeakerClassifier(self.encoder, n_embd=12, n_hidden=10)
        self.ids = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 7, 8, 0]])

    def test_attention_normalization_and_padding(self):
        hidden, maps = self.encoder(self.ids)
        self.assertEqual(hidden.shape, (2, 5, 12))
        self.assertEqual(len(maps), 2)
        for attention in maps:
            self.assertEqual(attention.shape, (2, 3, 5, 5))
            torch.testing.assert_close(attention.sum(-1), torch.ones(2, 3, 5))
            padded_keys = self.ids.eq(0)[:, None, None, :].expand_as(attention)
            self.assertEqual(torch.count_nonzero(attention[padded_keys]).item(), 0)
            self.assertTrue(torch.isfinite(attention).all().item())

    def test_right_padding_does_not_change_classification(self):
        short = torch.tensor([[2, 3, 4]])
        padded = F.pad(short, (0, 5))
        with torch.no_grad():
            torch.testing.assert_close(self.model(short), self.model(padded), atol=1e-6, rtol=1e-5)

    def test_encoder_can_use_future_words(self):
        first = torch.tensor([[2, 3, 4]])
        changed = torch.tensor([[2, 3, 9]])
        with torch.no_grad():
            hidden_a, _ = self.encoder(first)
            hidden_b, _ = self.encoder(changed)
        self.assertGreater((hidden_a[:, 0] - hidden_b[:, 0]).abs().max().item(), 1e-5)

    def test_joint_gradient_flow(self):
        loss = F.cross_entropy(self.model(self.ids), torch.tensor([0, 2]))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertEqual(self.encoder.token_embedding.weight.grad[0].abs().sum().item(), 0)

    def test_invalid_inputs_fail_clearly(self):
        for ids in (torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 9, dtype=torch.long),
                    torch.ones(3, dtype=torch.long)):
            with self.subTest(shape=tuple(ids.shape)):
                with self.assertRaises(ValueError):
                    self.encoder(ids)

    def test_attention_matches_pytorch_reference(self):
        # PyTorch's fused primitive is a test oracle only, never a model component.
        attention = MultiHeadSelfAttention(12, 3)
        x = torch.randn(2, 5, 12)
        mask = self.ids.eq(0)
        actual, _ = attention(x, mask)
        q, k, v = attention.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(2, 5, 3, 4).transpose(1, 2) for t in (q, k, v)]
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask[:, None, None, :])
        reference = attention.projection(reference.transpose(1, 2).reshape(2, 5, 12))
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)

    def test_accuracy_preserves_model_mode(self):
        batch = [(self.ids, torch.tensor([0, 2]))]
        for training in (False, True):
            self.model.train(training)
            accuracy = compute_classifier_accuracy(self.model, batch)
            self.assertEqual(self.model.training, training)
            self.assertGreaterEqual(accuracy, 0)
            self.assertLessEqual(accuracy, 100)

    def test_collation_truncates_and_right_pads(self):
        inputs, labels = collate_batch([(torch.tensor([2, 3]), torch.tensor(1)),
                                       (torch.arange(2, 42), torch.tensor(2))])
        self.assertEqual(inputs.shape, (2, 32))
        torch.testing.assert_close(inputs[0, :2], torch.tensor([2, 3]))
        self.assertEqual(inputs[0, 2:].count_nonzero().item(), 0)
        torch.testing.assert_close(inputs[1], torch.arange(2, 34))
        torch.testing.assert_close(labels, torch.tensor([1, 2]))


if __name__ == "__main__":
    unittest.main()
