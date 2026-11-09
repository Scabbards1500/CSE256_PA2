"""Inspect next-token errors in the final decoder using the report's scoring protocol."""

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if (ROOT / '.local_packages').is_dir():
    sys.path.insert(0, str(ROOT / '.local_packages'))

import torch
from torch.utils.data import DataLoader
from cse256_pa2.dataset import LanguageModelingDataset
from cse256_pa2.train import load_texts
from cse256_pa2.tokenizer import SimpleTokenizer
from cse256_pa2.transformer import DecoderLanguageModel


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = ROOT / 'outputs/part2'
    checkpoint = run / 'model.pt'
    protected = [checkpoint, run / 'metrics.json', run / 'vocabulary.json']
    original_hashes = {str(p): sha256(p) for p in protected}
    saved = torch.load(checkpoint, map_location='cpu')
    config, stoi = saved['config'], saved['tokenizer_stoi']
    assert config['max_iters'] == 500
    tokenizer = SimpleTokenizer(' '.join(load_texts(ROOT / 'speechesdataset')))
    assert tokenizer.stoi == stoi
    train_path = ROOT / 'speechesdataset/train_LM.txt'
    train_ids = set(tokenizer.encode(train_path.read_text(encoding='utf-8')))
    assert 0 not in train_ids and 1 not in train_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = DecoderLanguageModel(
        len(stoi), config['block_size'], config['n_embd'], config['n_head'],
        config['n_layer'], config['decoder_ffn_hidden'],
    ).to(device).eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.requires_grad_(False)
    group_for_id = torch.ones(len(stoi), dtype=torch.long, device=device)
    group_for_id[list(train_ids)] = 0
    group_for_id[stoi['<unk>']] = 2
    group_for_id[stoi['<pad>']] = -1
    group_names = ['seen_in_lm_training', 'in_vocabulary_unseen_in_lm_training', 'oov_encoded_as_unk']
    metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    corpora = {}
    for name in ['obama', 'wbush', 'hbush']:
        path = ROOT / f'speechesdataset/test_LM_{name}.txt'
        dataset = LanguageModelingDataset(tokenizer, path.read_text(encoding='utf-8'), config['block_size'])
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
        counts = torch.zeros(3, dtype=torch.long, device=device)
        correct = torch.zeros(3, dtype=torch.float64, device=device)
        losses = torch.zeros(3, dtype=torch.float64, device=device)
        with torch.inference_mode():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                flat_targets = targets.reshape(-1)
                flat_logits = logits.reshape(-1, len(stoi))
                group = group_for_id[flat_targets]
                assert (group >= 0).all()
                nll = torch.nn.functional.cross_entropy(flat_logits, flat_targets, reduction='none')
                hits = flat_logits.argmax(-1).eq(flat_targets)
                counts += torch.bincount(group, minlength=3)
                correct += torch.bincount(group, weights=hits.double(), minlength=3)
                losses += torch.bincount(group, weights=nll.double(), minlength=3)
        counts, correct, losses = counts.cpu().tolist(), correct.cpu().tolist(), losses.cpu().tolist()
        total = sum(counts)
        assert total == metrics['evaluation_target_tokens'][name]
        mean_nll = sum(losses) / total
        perplexity = math.exp(mean_nll)
        original_ppl = metrics['history'][-1][f'{name}_perplexity']
        assert math.isclose(perplexity, original_ppl, rel_tol=1e-5), (name, perplexity, original_ppl)
        groups = {}
        for i, group_name in enumerate(group_names):
            groups[group_name] = {
                'target_occurrences': counts[i], 'target_share_percent': 100 * counts[i] / total,
                'correct_top1': int(correct[i]), 'errors_top1': counts[i] - int(correct[i]),
                'top1_accuracy_percent': 100 * correct[i] / counts[i] if counts[i] else None,
                'mean_nll_nats': losses[i] / counts[i] if counts[i] else None,
                'share_of_total_nll_percent': 100 * losses[i] / sum(losses),
            }
        corpora[name] = {
            'source_sha256': sha256(path), 'windows': len(dataset), 'target_occurrences': total,
            'correct_top1': int(sum(correct)), 'errors_top1': total - int(sum(correct)),
            'top1_accuracy_percent': 100 * sum(correct) / total,
            'mean_nll_nats': mean_nll, 'perplexity': perplexity,
            'original_report_perplexity': original_ppl, 'perplexity_matches_report': True,
            'groups': groups,
        }
        print(name, json.dumps(corpora[name]), flush=True)
    assert all(sha256(Path(p)) == h for p, h in original_hashes.items())
    result = {
        'checkpoint_sha256': original_hashes[str(checkpoint)], 'training_corpus_sha256': sha256(train_path),
        'device': str(device), 'torch_version': torch.__version__,
        'protocol': 'All stride-one windows; every next-token target in each 32-token window, conditioned on the observed prefix.',
        'interpretation': 'Top-1 mismatch is against the observed corpus target, not a judgment that another continuation is linguistically invalid. OOV targets are the shared unknown ID, not their original spellings.',
        'group_weighting': 'Groups partition target occurrences, not raw corpus tokens; aggregate NLL is target-weighted, not an average of group perplexities.',
        'source_checkpoint_metrics_and_vocabulary_unchanged': True, 'corpora': corpora,
    }
    (run / 'error_analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved outputs/part2/error_analysis.json; no model or report changes.', flush=True)


if __name__ == '__main__':
    main()
