"""Cache only the six checkpoints and tokenizer files used in Part 3."""
import os
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[2]
for revision in ['step143000', 'step512', 'step1000', 'step8000', 'step0', 'step32000']:
    print(f'Downloading {revision}', flush=True)
    path = snapshot_download(
        'EleutherAI/pythia-160m-deduped', revision=revision,
        cache_dir=str(root / '.cache/huggingface'),
        allow_patterns=['config.json', 'pytorch_model.bin', 'tokenizer.json',
                        'tokenizer_config.json', 'special_tokens_map.json'],
        max_workers=2,
    )
    print(f'Cached {revision}: {path}', flush=True)
