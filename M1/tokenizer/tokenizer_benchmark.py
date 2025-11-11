#!/usr/bin/env python3
"""Tokenization benchmark across many tokenizers.

Usage: run from repo root or from M1/tokenizer. Examples in README-BENCH.md

What it does:
- Loads tokenizers (tokenizers JSON files from `M1/tokenizer/tokenizers` by default)
- Optionally loads HuggingFace/tokenizers-based tokenizers (if available)
- Tokenizes provided texts and records token counts
- Writes CSV and prints per-text ranking (which tokenizer produced fewest tokens)

This script is defensive: if optional packages are missing it will run in dry-run
mode and print what it would do.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import List, Tuple

try:
    from tokenizers import Tokenizer as TokenizersTokenizer
    HAVE_TOKENIZERS = True
except Exception:
    TokenizersTokenizer = None
    HAVE_TOKENIZERS = False

try:
    from transformers import AutoTokenizer as HFTokenizer
    HAVE_TRANSFORMERS = True
except Exception:
    HFTokenizer = None
    HAVE_TRANSFORMERS = False


def find_default_tokenizers(dirpath: Path) -> List[Path]:
    p = dirpath
    return sorted([Path(p_) for p_ in glob.glob(str(p / "*.json"))])


def load_tokenizers(tokenizer_paths: List[Path]):
    loaded = []
    for p in tokenizer_paths:
        name = p.stem
        if p.suffix == '.json' and HAVE_TOKENIZERS:
            try:
                tok = TokenizersTokenizer.from_file(str(p))
                loaded.append((name, 'tokenizers_json', p, tok))
            except Exception as e:
                print(f"Warning: failed to load tokenizers JSON {p}: {e}")
        else:
            # try HF if available
            if HAVE_TRANSFORMERS:
                try:
                    tok = HFTokenizer.from_pretrained(str(p))
                    loaded.append((name, 'hf', p, tok))
                except Exception:
                    # fallback: maybe the path is a name on HF hub
                    try:
                        tok = HFTokenizer.from_pretrained(name)
                        loaded.append((name, 'hf', p, tok))
                    except Exception:
                        print(f"Skipping {p}: cannot load as JSON or HF tokenizer")
            else:
                print(f"Skipping {p}: neither tokenizers nor transformers are available")

    return loaded


def tokenize_with_tokenizers_lib(tok: 'TokenizersTokenizer', text: str) -> int:
    enc = tok.encode(text)
    return len(enc.ids)


def tokenize_with_hf(tok, text: str) -> int:
    # return number of tokens (ids) when encoding without adding special tokens
    ids = tok.encode(text, add_special_tokens=False)
    return len(ids)


def read_text(path: Path, max_chars: int = None) -> Tuple[str, int]:
    txt = path.read_text(encoding='utf-8')
    if max_chars:
        txt = txt[:max_chars]
    return txt, len(txt)


def run_benchmark(tokenizer_paths: List[str], text_paths: List[str], out_csv: str, max_chars: int = None):
    tpaths = [Path(p) for p in tokenizer_paths]
    texts = [Path(p) for p in text_paths]

    # load candidates (best-effort)
    loaded = load_tokenizers(tpaths)

    if not loaded:
        print("No tokenizers loaded. Install 'tokenizers' and/or 'transformers' or pass valid paths.")
        return

    rows = []
    for text_path in texts:
        text_name = text_path.name
        txt, chars = read_text(text_path, max_chars=max_chars)
        for name, kind, p, tok in loaded:
            try:
                if kind == 'tokenizers_json' and HAVE_TOKENIZERS:
                    count = tokenize_with_tokenizers_lib(tok, txt)
                elif kind == 'hf' and HAVE_TRANSFORMERS:
                    count = tokenize_with_hf(tok, txt)
                else:
                    count = -1
            except Exception as e:
                print(f"Error tokenizing {text_name} with {name}: {e}")
                count = -1

            rows.append({'text': text_name, 'text_path': str(text_path), 'chars': chars,
                         'tokenizer_name': name, 'tokenizer_path': str(p), 'token_count': count})

    # write CSV
    fieldnames = ['text', 'text_path', 'chars', 'tokenizer_name', 'tokenizer_path', 'token_count']
    outp = Path(out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # print ranking per text
    print('\nPer-text ranking (fewest tokens => most efficient)')
    by_text = {}
    for r in rows:
        by_text.setdefault(r['text'], []).append(r)

    for text_name, list_rows in by_text.items():
        sorted_rows = sorted(list_rows, key=lambda x: x['token_count'] if x['token_count']>=0 else 10**9)
        print(f"\nText: {text_name} (chars: {sorted_rows[0]['chars']})")
        for r in sorted_rows[:10]:
            tc = r['token_count']
            ratio = None
            if tc and tc > 0:
                ratio = tc / (r['chars'] / 1000)
                ratio_str = f"{ratio:.1f} tokens/1k_chars"
            else:
                ratio_str = 'N/A'
            print(f"  - {r['tokenizer_name']:<30} tokens: {tc:>7} | {ratio_str} | path: {Path(r['tokenizer_path']).name}")

    print(f"\nWrote CSV to {outp}")


def main(argv=None):
    p = argparse.ArgumentParser(description='Tokenizers cross-benchmark: count tokens for given texts')
    p.add_argument('--tokenizers-dir', default='M1/tokenizer/tokenizers', help='Directory with tokenizer JSONs or HF tokenizer dirs')
    p.add_argument('--tokenizers', nargs='*', help='Explicit tokenizer paths (overrides tokenizers-dir)')
    p.add_argument('--texts', nargs='*', help='Text files to tokenize (paths)',
                   default=['M1/korpus-wolnelektury/pan-tadeusz-ksiega-1.txt',
                            'M1/korpus-mini/the-pickwick-papers-gutenberg.txt',
                            'M1/korpus-mini/fryderyk-chopin-wikipedia.txt'])
    p.add_argument('--out', default='M1/tokenizer/tokenization_benchmark.csv', help='CSV output')
    p.add_argument('--max-chars', type=int, default=None, help='If set, limit each text to this many chars')
    args = p.parse_args(argv)

    if args.tokenizers:
        tokenizers = args.tokenizers
    else:
        tokenizers = [str(p) for p in find_default_tokenizers(Path(args.tokenizers_dir))]

    if not tokenizers:
        print('No tokenizer files found in', args.tokenizers_dir)
        return

    print('Tokenizers to try:')
    for t in tokenizers:
        print(' -', t)

    print('\nTexts to process:')
    for t in args.texts:
        print(' -', t)

    run_benchmark(tokenizers, args.texts, args.out, max_chars=args.max_chars)


if __name__ == '__main__':
    main()
