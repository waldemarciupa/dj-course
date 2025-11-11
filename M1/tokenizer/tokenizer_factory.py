"""Dynamic tokenizer builder

Creates BPE tokenizers from different corpora using the local `corpora.get_corpus_file` helper.

Produces these JSON files under ./tokenizers/ by default:
- tokenizer-pan-tadeusz.json       (only Pan Tadeusz files from WOLNELEKTURY)
- tokenizer-wolnelektury.json     (all files from WOLNELEKTURY)
- tokenizer-nkjp.json             (all files from NKJP)
- tokenizer-all-corpora.json      (union of WOLNELEKTURY and NKJP)

Usage examples:
  python tokenizer_factory.py --dry-run
  python tokenizer_factory.py

Notes:
- This script expects to be run from the `M1/tokenizer` directory or that
  Python's import path resolves the local `corpora` module (there is a
  `corpora.py` alongside this file in the repo).
- If you don't have `tokenizers` installed, use `--dry-run` to only list files.
"""
from pathlib import Path
import argparse
import sys
from typing import Iterable, List

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    HAVE_TOKENIZERS = True
except Exception:
    HAVE_TOKENIZERS = False

from corpora import get_corpus_file


DEFAULT_VOCAB_SIZE = 32000
DEFAULT_MIN_FREQUENCY = 2
DEFAULT_SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]


def build_tokenizer(output_path: Path, files: Iterable[Path], *,
                    vocab_size: int = DEFAULT_VOCAB_SIZE,
                    min_frequency: int = DEFAULT_MIN_FREQUENCY,
                    special_tokens: List[str] = None,
                    dry_run: bool = False):
    special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
    file_list = [str(p) for p in files]
    if not file_list:
        raise ValueError(f"No files provided for tokenizer {output_path}")

    print(f"\nPreparing tokenizer '{output_path.name}' with {len(file_list)} files")
    if dry_run or not HAVE_TOKENIZERS:
        msg = "(dry-run)" if dry_run else "(tokenizers package not available - dry run)"
        print(f"  {msg} Will not train tokenizer, listing first 10 files:")
        for f in file_list[:10]:
            print("   -", f)
        return

    # Initialize tokenizer and trainer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size, min_frequency=min_frequency)

    # Train and save
    print("  Training tokenizer (this may take a while)...")
    tokenizer.train(file_list, trainer=trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"  Saved tokenizer to {output_path}")


def resolve_files_for_target(target: str) -> List[Path]:
    """Return a list of Path objects for a named target.

    target may be one of:
      - 'pan-tadeusz' -> pan-tadeusz files under WOLNELEKTURY
      - 'wolnelektury' -> all files under WOLNELEKTURY
      - 'nkjp' -> all files under NKJP
      - 'all' -> union of WOLNELEKTURY and NKJP
    """
    target = target.lower()
    if target == 'pan-tadeusz':
        return get_corpus_file("WOLNELEKTURY", "pan-tadeusz-ksiega-*.txt")
    if target == 'wolnelektury':
        return get_corpus_file("WOLNELEKTURY", "*.txt")
    if target == 'nkjp':
        return get_corpus_file("NKJP", "*.txt")
    if target == 'all':
        files = []
        files.extend(get_corpus_file("WOLNELEKTURY", "*.txt"))
        files.extend(get_corpus_file("NKJP", "*.txt"))
        # deduplicate while preserving order
        seen = set()
        unique = []
        for p in files:
            if str(p) not in seen:
                seen.add(str(p))
                unique.append(p)
        return unique
    raise ValueError(f"Unknown target: {target}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build multiple BPE tokenizers from repo corpora")
    parser.add_argument('--out-dir', default='tokenizers', help='Output directory for tokenizer JSON files')
    parser.add_argument('--vocab-size', type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument('--min-frequency', type=int, default=DEFAULT_MIN_FREQUENCY)
    parser.add_argument('--dry-run', action='store_true', help='Resolve files and show summary without training')
    parser.add_argument('--targets', nargs='*', choices=['pan-tadeusz','wolnelektury','nkjp','all'],
                        default=['pan-tadeusz','wolnelektury','nkjp','all'],
                        help='Which tokenizers to build')
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)

    mapping = {
        'pan-tadeusz': out_dir / 'tokenizer-pan-tadeusz.json',
        'wolnelektury': out_dir / 'tokenizer-wolnelektury.json',
        'nkjp': out_dir / 'tokenizer-nkjp.json',
        'all': out_dir / 'tokenizer-all-corpora.json',
    }

    for t in args.targets:
        try:
            files = resolve_files_for_target(t)
        except Exception as e:
            print(f"Skipping target {t}: error resolving files: {e}")
            continue

        if not files:
            print(f"Skipping target {t}: no files found")
            continue

        try:
            build_tokenizer(mapping[t], files, vocab_size=args.vocab_size,
                            min_frequency=args.min_frequency,
                            dry_run=args.dry_run)
        except Exception as e:
            print(f"Error building tokenizer {t}: {e}")

    print('\nDone')


if __name__ == '__main__':
    main()
