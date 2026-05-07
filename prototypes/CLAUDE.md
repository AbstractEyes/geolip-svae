# prototypes/CLAUDE.md

For Claudes (and humans) working in `prototypes/` — the sister-package
sandbox where new experiments live before graduating into `geolip_svae/`
proper. Read this before authoring a new experiment, *especially* if
the experiment touches text/byte data or any other workload where dataset
construction does heavy lifting before training begins.

This is the prototype-side companion to the repo-level `../CLAUDE.md`.
The main package's invariants still apply (sphere-norm, fp64 SVD,
no AdamW, etc.) — this file just adds prototypes-specific gotchas.

## What this directory is

`svae-proto` is a separate installable package (`pyproject.toml` here at
`prototypes/`, importable as `svae_proto`). It depends on `geolip-svae`;
the core never imports from prototypes. That contract is load-bearing —
violating it puts you back in the bloat hole earlier projects fell into.

Each experiment is one subdirectory: `prototypes/svae_proto/exp_NNN_<slug>/`.

## Authoring a new experiment — the 6-step procedure

1. **Pick the next number and a short slug.** `exp_002_2gram_battery`,
   `exp_003_seq_strict_loss`, etc. The `exp_` prefix is REQUIRED — Python
   module names cannot start with a digit, so `002_2gram_battery` would
   not be importable. Three-digit zero-padded so 99→100 sorts correctly.

2. **Copy the exp_001 directory as a template.**
   ```bash
   cp -r prototypes/svae_proto/exp_001_vocab_trigram_recall \
         prototypes/svae_proto/exp_NNN_<slug>/
   ```

3. **Fill the four code files** (existing exp_001 versions are the
   canonical shapes):

   | File | Contains |
   |---|---|
   | `NOTES.md`  | hypothesis, success criteria, what would convince you to graduate this, results once known |
   | `dataset.py`| any new Dataset class + a `<name>_factory(cfg, channels)` returning `DatasetBundle` |
   | `eval.py`   | post-train metrics specific to this experiment (run separately or via `run.py`) |
   | `cfg.py`    | `_BASE` for shared keys + per-variant `CFG_*` dicts; `hf_version` always `'exp_NNN_*'` |
   | `run.py`    | thin CLI that registers the dataset transiently and calls `geolip_svae.train.train(cfg)` |

4. **Add experiment-specific deps** to `prototypes/pyproject.toml` under
   `[project.optional-dependencies]` as `exp_NNN = [...]`. Never add to
   the core `dependencies` list — that defeats the slim-core contract.
   Update the `all` extras to include the new key.

5. **Register the dataset transiently in `run.py`.** Pattern from
   exp_001:
   ```python
   from geolip_svae.dataset_presets import DATASET_FACTORIES
   from .dataset import my_factory
   DATASET_FACTORIES['<my_name>'] = my_factory   # transient — not source-edited
   ```
   The cfg's `dataset='<my_name>'` then resolves through `get_dataset_bundle`.

6. **Update the registry tables** in `prototypes/README.md` and
   `../README.md`'s Prototypes section — add the new experiment row with
   ID / slug / status / extras-key. One-line update; helps anyone landing
   on the repo find the experiment.

When the experiment proves out, graduate the dataset/metric helpers into
the appropriate `geolip_svae` module and demote the prototype to a
historical record. Don't graduate code that hasn't actually shipped a
result.

## The "looks like a free choice but isn't" list — traps observed in exp_001

These are the things that bit us hard during the vocab_trigram_recall
experiment. Future experiments touching text/byte data or any large-corpus
preprocessing should assume these traps will fire again unless explicitly
worked around.

### Tokenizer traps

**1. HF "fast" tokenizers silently fall back.** Calling
`AutoTokenizer.from_pretrained(name, use_fast=True)` returns the Python
slow tokenizer if the installed `tokenizers` Rust binding can't parse
the saved `tokenizer.json`. The class name reports as `T5Tokenizer`, but
`is_fast=True` (lying — class attribute mismatch in HF's fallback path).
Throughput drops to ~1 MB/s on 500 MB of text = ~10 minutes that looks
indistinguishable from slow setup work.

**Workaround:** for SentencePiece-based models (T5/mT5/ALBERT/XLNet),
bypass HF entirely and use `sentencepiece.SentencePieceProcessor` directly:
```python
import sentencepiece as spm
hf = AutoTokenizer.from_pretrained(name)             # only for spiece.model path
sp = spm.SentencePieceProcessor()
sp.Load(hf.vocab_file)
ids = sp.EncodeAsIds(chunk_text)                     # pure C++, 30-100 MB/s
```
See `exp_001_vocab_trigram_recall/dataset.py:_TokenizerWrapper` for the
production version. It dispatches by tokenizer family.

**2. `return_offsets_mapping=True` is a slow path on T5.** Even the
genuine `T5TokenizerFast` walks character-by-character to compute char
offsets, dropping throughput by 5-10×. Use vocab-id-to-bytes lookup
instead (build a `[vocab_id -> utf-8 bytes]` table once, ~32K decode
calls, then concatenate per-chunk).

### Memory traps

**3. Python lists of ints are 5-7× the size of the int64 numpy
equivalent.** A 150M-token Python list (one per token in wikitext-103)
is ~5.4 GB of pure Python overhead before the actual int data. The HF
tokenizer call returns this list, and if you let it stay alive while
also building the numpy array + downstream buffers, peak RSS hits 60+ GB.

**Workaround:** as soon as you've copied a Python list to numpy via
`np.asarray`, `del` the original list and `gc.collect()`. The
list_obj → np.array → del pattern is the difference between 6 GB and
65 GB peak in exp_001's dataset construction.

**4. Single-shot tokenize on a 500 MB string is the worst case.** It
combines trap #3 with a giant BatchEncoding holding extra metadata
(attention_mask, token_type_ids) even when you ask it not to. Always
chunk corpora ≥10 MB into ~2 MB pieces and tokenize per-chunk:
```python
for chunk_start in range(0, len(text), chunk_chars):
    chunk = text[chunk_start : chunk_start + chunk_chars]
    enc = tokenizer(chunk, ...)
    chunk_ids.append(np.asarray(enc['input_ids'], dtype=np.int64))
    del enc                                              # release per-chunk
```
Concatenate small numpy arrays at the end. Per-chunk peak is ~25 MB
regardless of corpus size.

### Caching traps

**5. Pay tokenize cost ONCE per unique (corpus, tokenizer, max_chars,
split).** Once you've built a `TokenizedStream` (or whatever the
experiment's preprocessed-data shape is), save it to `~/.cache/svae_proto/<exp_NNN>/`
as `.npz`. Hash the cfg tuple for the file name. Cache hit = ~3 seconds
to load 3 GB; cache miss = whatever the slow build cost was.

Override the cache root via env var so Drive-mounted setups can persist
across runtime restarts:
```python
SVAE_PROTO_CACHE_DIR = os.environ.get(
    'SVAE_PROTO_CACHE_DIR',
    os.path.expanduser('~/.cache/svae_proto/<exp_NNN>'),
)
```

### Diagnostic discipline

**6. Always instrument heavy preprocessing with per-step timing + RSS
prints to stderr.** When something hangs, you need to know which step
it's hung in. Pattern:
```python
class _Step:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t0 = time.time(); self.rss0 = _rss_gb()
        sys.stderr.write(f'  [{exp_NNN}] {self.label}... ')
        sys.stderr.flush()
        return self
    def __exit__(self, *exc):
        dt = time.time() - self.t0; rss = _rss_gb()
        sys.stderr.write(
            f'done in {dt:.1f}s (RSS {rss:.2f} GB, '
            f'{rss-self.rss0:+.2f})\n')
        sys.stderr.flush()
```
For long loops, print every chunk for the first 5-10 iterations so you
spot anomalies immediately, then drop to every-N for the long tail. Use
`stderr` and `flush()` so prints survive past tqdm bars and pipe redirects.

**7. Trust the empirical RSS over the theoretical one.** When peak RSS
exceeds your back-of-envelope estimate by 5×, there's a hidden
copy/intermediate somewhere. Don't tune chunk sizes hoping it goes away
— add `del + gc.collect()` between every meaningful stage and watch the
RSS deltas to find which stage is actually leaking.

## Standard `cfg.py` shape

Three-tier structure (from exp_001):

```python
_BASE: Dict[str, Any] = dict(
    # Dataset, loss/band, output/IO — keys ALL variants share
    dataset='<name>',
    upload=True,
    hf_repo='AbstractPhil/<repo-for-this-experiment-series>',
    build_codebook=True,
    build_topology=True,
    # ...
)

# Per-variant keys live HERE, not in _BASE:
#   V, D, patch_size, hidden, depth, n_cross, n_heads, smooth_mid
#   batch_size, lr, epochs, ds_size, val_size, save_every, report_every
#   linear_readout, svd_mode, svd_method, etc.
# Reason: param-count differences (e.g. ~57K vs ~17M) make shared
# batch_size / ds_size either OOM the big variant or starve the small.

CFG_<VARIANT_A>: Dict[str, Any] = dict(_BASE, ..., hf_version='exp_NNN_...')
CFG_<VARIANT_B>: Dict[str, Any] = dict(_BASE, ..., hf_version='exp_NNN_...')
```

Always prefix `hf_version` with `exp_NNN_` so HuggingFace artifacts sort
under the experiment that produced them.

## Standard `run.py` shape

```python
from geolip_svae.dataset_presets import DATASET_FACTORIES
from .dataset import <name>_factory
DATASET_FACTORIES['<name>'] = <name>_factory   # transient registration

import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=list(_VARIANTS), default='<default>')
parser.add_argument('--hf-token', default=None)
parser.add_argument('--hf-repo', default=None)
parser.add_argument('--hf-version', default=None)
parser.add_argument('--no-upload', action='store_true')
parser.add_argument('--cfg-override', action='append', default=[])
parser.add_argument('--skip-train', action='store_true')
args = parser.parse_args()

# Set HF_TOKEN BEFORE importing geolip_svae.train (its module-level auth runs at import)
if args.hf_token:
    os.environ['HF_TOKEN'] = args.hf_token

from geolip_svae.train import train
from . import cfg as cfg_mod
cfg = dict(getattr(cfg_mod, _VARIANTS[args.variant]))
# Apply overrides...
if not args.skip_train:
    train(cfg)
# Then run experiment-specific eval against best.pt under save_dir/
```

The CLI flags should match what `geolip_svae.train`'s main CLI exposes
for consistency.

## Mount truncation gotcha

If you're working in Cowork mode (Windows VFS mounted into a Linux
sandbox), expect long Edit/Write operations to occasionally truncate at
~30-40KB. Symptoms: file `.py` ends mid-statement, `ast.parse` raises
`SyntaxError: '(' was never closed` or `unterminated triple-quoted
string`. Recovery: re-splice the tail from `git show HEAD:<path>`. The
file API tools (Read/Write/Edit) are more reliable than bash heredocs
for large writes; bash views the mount inconsistently with the
file-API view in degraded states.

## When in doubt

- **Found a function in `geolip_svae` that does what you want?** Use it.
  Don't re-implement; the prototype's job is the *new* piece, not
  rebuilding what the core already provides.
- **Need a heavy dep that the core doesn't have?** Put it in
  `pyproject.toml`'s `[project.optional-dependencies] exp_NNN = [...]`,
  never in core deps.
- **Tempted to edit `geolip_svae/` to make the prototype work?** Stop.
  Either the prototype should call core APIs as-is, or the change to
  the core is a graduation move that deserves its own PR with tests
  and docs.
- **About to write a Python loop that scales with corpus size?** It's
  the wrong loop. Numpy fancy indexing or vectorized ops can almost
  always do the same work in a tight C path. The vocab_id_to_bytes
  vs per-token decode story (exp_001) is the canonical example —
  150M iterations vs 32K iterations for the same outcome.
