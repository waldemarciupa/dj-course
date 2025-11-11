# Tokenizer cross-benchmark

This small utility tokenizes a set of texts using all tokenizer JSONs in `M1/tokenizer/tokenizers` (and optionally HuggingFace tokenizers) and writes token counts to CSV.

Quick usage (from repo root):

```bash
# run with default tokenizers dir and default 3 texts
python M1/tokenizer/tokenizer_benchmark.py

# limit texts to first 20000 chars (faster)
python M1/tokenizer/tokenizer_benchmark.py --max-chars 20000

# specify explicit tokenizer files
python M1/tokenizer/tokenizer_benchmark.py --tokenizers M1/tokenizer/tokenizers/tokenizer-pan-tadeusz.json M1/tokenizer/tokenizers/bielik-v3-tokenizer.json
```

Output:

- `M1/tokenizer/tokenization_benchmark.csv` — token counts per (text, tokenizer)

If you want to include HF tokenizers (e.g., downloaded via `transformers`), pass their paths as `--tokenizers` or put them in `--tokenizers-dir` and have `transformers` installed.

Dependencies are optional; see `requirements-benchmark.txt`.
