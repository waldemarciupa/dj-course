from tokenizers import Tokenizer
import os
import json
import csv
import math
import traceback
from collections import defaultdict

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception:
    pd = None
    plt = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_DIR = os.path.join(SCRIPT_DIR, 'tokenizers')
SAMPLES_DIR = os.path.join(SCRIPT_DIR, 'samples')

def discover_tokenizers(tokenizer_dir):
    tokenizers = {}
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(f"Tokenizer dir not found: {tokenizer_dir}")

    for fn in os.listdir(tokenizer_dir):
        if fn.endswith('.json'):
            key = fn[:-5]
            path = os.path.join(tokenizer_dir, fn)
            try:
                tokenizers[key] = Tokenizer.from_file(path)
            except Exception:
                print(f"⚠️ Failed to load tokenizer '{key}' from {path}")
                traceback.print_exc()
    return tokenizers


def discover_samples(samples_dir):
    files = os.listdir(samples_dir)
    stems = set()
    for fn in files:
        if fn.endswith('-nows.json'):
            stems.add(fn[:-10])
        elif fn.endswith('.json'):
            stems.add(fn[:-5])
        elif fn.endswith('.toon'):
            stems.add(fn[:-5])
        elif fn.endswith('.yaml') or fn.endswith('.yml'):
            stems.add(fn[:-5])
    return sorted(stems)


def read_sample(samples_dir, stem):
    data = {}
    paths = {
        'json': os.path.join(samples_dir, f"{stem}.json"),
        'nows-json': os.path.join(samples_dir, f"{stem}-nows.json"),
        'toon': os.path.join(samples_dir, f"{stem}.toon"),
        'yaml': os.path.join(samples_dir, f"{stem}.yaml"),
    }
    for k, p in paths.items():
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data[k] = f.read()
        except FileNotFoundError:
            data[k] = ""
    return data


def encode_text(tokenizer, text, is_json=False):
    to_encode = text
    if is_json:
        # Keep previous behaviour: wrap in json.dumps so we tokenize the JSON string form
        to_encode = json.dumps(text)
    try:
        enc = tokenizer.encode(to_encode)
        return len(enc.ids)
    except Exception:
        # fallback: try encoding raw
        try:
            enc = tokenizer.encode(text)
            return len(enc.ids)
        except Exception:
            traceback.print_exc()
            return None


def main():
    tokenizers = discover_tokenizers(TOKENIZER_DIR)
    if not tokenizers:
        print('No tokenizers found. Exiting.')
        return

    samples = discover_samples(SAMPLES_DIR)
    if not samples:
        print('No samples found. Exiting.')
        return

    print(f"Found tokenizers: {list(tokenizers.keys())}")
    print(f"Found samples: {samples}")

    rows = []

    for tk_name, tk in tokenizers.items():
        for stem in samples:
            data = read_sample(SAMPLES_DIR, stem)
            # json and nows-json will be wrapped with json.dumps like previous scripts
            counts = {}
            counts['json'] = encode_text(tk, data.get('json', ''), is_json=True)
            counts['nows-json'] = encode_text(tk, data.get('nows-json', ''), is_json=True)
            counts['yaml'] = encode_text(tk, data.get('yaml', ''), is_json=False)
            counts['toon'] = encode_text(tk, data.get('toon', ''), is_json=False)

            for fmt, cnt in counts.items():
                rows.append({
                    'tokenizer': tk_name,
                    'sample': stem,
                    'format': fmt,
                    'tokens': cnt if cnt is not None else -1,
                })

    out_csv = os.path.join(SCRIPT_DIR, 'tokenization_results.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['tokenizer', 'sample', 'format', 'tokens'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote results to {out_csv}")

    # Generate summary plots if pandas & matplotlib available
    if pd is None or plt is None:
        print('pandas or matplotlib not installed; skipping plots. Install with: pip install pandas matplotlib')
        return

    df = pd.DataFrame(rows)
    # Replace -1 (errors) with NaN for plotting
    df['tokens'] = df['tokens'].replace(-1, pd.NA).astype('Float64')

    # Average tokens per sample & format across tokenizers
    avg = df.groupby(['sample', 'format'])['tokens'].mean().reset_index()
    pivot = avg.pivot(index='sample', columns='format', values='tokens').fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(14,6))

    # Left: grouped bars of average tokens per format per sample
    pivot.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Średnia liczba tokenów (po wszystkich tokenizerach)')
    axes[0].set_ylabel('Średnia liczba tokenów')
    axes[0].legend(title='Format')

    # Right: percent savings of toon vs json (avg across tokenizers)
    # compute per-sample: (json - toon) / json *100
    savings = []
    for sample in pivot.index:
        json_t = pivot.loc[sample].get('json', pd.NA)
        toon_t = pivot.loc[sample].get('toon', pd.NA)
        if pd.isna(json_t) or json_t == 0:
            pct = pd.NA
        else:
            pct = (json_t - toon_t) / json_t * 100.0
        savings.append({'sample': sample, 'save_pct': pct})

    save_df = pd.DataFrame(savings).set_index('sample')
    save_df.plot(kind='bar', ax=axes[1], legend=False, color='tab:green')
    axes[1].set_title('Oszczędność tokenów: toon vs json (średnio)')
    axes[1].set_ylabel('Procent oszczędności (%)')

    plt.tight_layout()
    out_png = os.path.join(SCRIPT_DIR, 'tokenization_summary.png')
    fig.savefig(out_png)
    print(f"Wygenerowano wykres: {out_png}")

    # Also compute detailed savings per tokenizer and sample and write small report
    detail = []
    for tk_name in df['tokenizer'].unique():
        sub = df[df['tokenizer'] == tk_name]
        for sample in sub['sample'].unique():
            row_json = sub[(sub['sample']==sample) & (sub['format']=='json')]
            row_toon = sub[(sub['sample']==sample) & (sub['format']=='toon')]
            if row_json.empty or row_toon.empty:
                continue
            j = row_json['tokens'].values[0]
            t = row_toon['tokens'].values[0]
            if j is None or j == 0 or pd.isna(j) or pd.isna(t):
                continue
            pct = (j - t) / j * 100.0
            detail.append({'tokenizer': tk_name, 'sample': sample, 'json': j, 'toon': t, 'save_pct': pct})

    detail_csv = os.path.join(SCRIPT_DIR, 'tokenization_savings_detail.csv')
    pd.DataFrame(detail).to_csv(detail_csv, index=False)
    print(f"Wygenerowano szczegóły oszczędności: {detail_csv}")


if __name__ == '__main__':
    main()
