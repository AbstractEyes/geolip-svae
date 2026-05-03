# svae-proto

Experimental scaffolding sister package for [geolip-svae](https://github.com/AbstractEyes/geolip-svae).

`svae-proto` is a **separate installable package**, not an optional extra
of `geolip-svae`. The main package stays slim — production environments
that just want to run the trained models pay zero cost for experiment
deps. When you want to run an experiment, you install `svae-proto`
(which pulls `geolip-svae` along) and optionally one of the per-experiment
extras for that experiment's dependencies.

## Install

`geolip-svae` and `svae-proto` are not on PyPI — both install directly
from GitHub. The svae-proto distribution lives in the `svae_proto/`
subdirectory of the geolip-svae repository, so pip needs the
`#subdirectory=svae_proto` URL fragment to find its `pyproject.toml`.

```bash
# ─── From GitHub (recommended) ──────────────────────────────────────

# Core only — every experiment can run, but per-experiment deps must be
# installed separately if needed.
pip install "git+https://github.com/AbstractEyes/geolip-svae.git#subdirectory=svae_proto"

# With one experiment's heavy deps:
pip install "svae-proto[exp_001] @ git+https://github.com/AbstractEyes/geolip-svae.git#subdirectory=svae_proto"

# Everything (every experiment's deps):
pip install "svae-proto[all] @ git+https://github.com/AbstractEyes/geolip-svae.git#subdirectory=svae_proto"

# Pin to a specific tag / branch / commit:
pip install "svae-proto @ git+https://github.com/AbstractEyes/geolip-svae.git@v0.9.0#subdirectory=svae_proto"

# ─── From a local clone (active development) ────────────────────────

# After cloning the repo:
git clone https://github.com/AbstractEyes/geolip-svae.git
cd geolip-svae

pip install ./svae_proto                    # core
pip install -e ./svae_proto                 # editable for live edits
pip install "./svae_proto[exp_001]"         # one experiment's deps
pip install "./svae_proto[all]"             # everything
```

In all cases the install pulls `geolip-svae` along automatically (also
from GitHub) — you do not need to install the main package first.

## Contract

**Prototypes depend on `geolip_svae`. The package never imports prototypes.**

This is the rule that prevents the bloat seen in earlier projects (where
research scaffolding leaked into core modules and never got cleaned up).
A prototype directory can be deleted at any time without breaking
anything in `geolip_svae/`. Keep that property load-bearing.

A prototype may register itself into `geolip_svae` registries
(`DATASET_FACTORIES`, `ACTIVATIONS`, `VERSIONS`, etc.) **at runtime**, but
must not modify the source files. A typical pattern:

```python
# svae_proto/exp_NNN_xxx/run.py
from geolip_svae.dataset_presets import DATASET_FACTORIES
from .dataset import my_factory
DATASET_FACTORIES['my_proto'] = my_factory   # transient registration

from geolip_svae.train import train
from .cfg import CFG
train(CFG)
```

When an experiment proves out and is ready for production use, its dataset
class / metric helper / etc. graduates into the appropriate `geolip_svae`
module and the prototype becomes a historical record of how it got there.

## Directory naming

```
svae_proto/
    pyproject.toml                  # this package
    README.md                       # this file
    __init__.py
    exp_NNN_short_slug/             # NNN = monotonic 3-digit experiment number
        __init__.py
        NOTES.md                    # hypothesis, success criteria, results
        dataset.py                  # any new dataset class + factory
        eval.py                     # any new metric / probe
        cfg.py                      # the cfg dict(s) under test
        run.py                      # `python -m svae_proto.exp_NNN_x.run` entry point
```

The `exp_` prefix is required — Python module names cannot start with a
digit, so `001_vocab_trigram_recall` would not be importable as a module.
The `exp_` prefix makes the experiment number render naturally in import
paths (`svae_proto.exp_001_vocab_trigram_recall`) while keeping the
visual sort order intact.

`NOTES.md` is the most important file in each experiment — it captures
*what we're testing and what would convince us to graduate this into the
package*. Keep it honest. If an experiment falls through, leave NOTES.md
updated with what went wrong so future-you doesn't repeat it.

## Per-experiment dependencies

Heavy or unusual deps go in `pyproject.toml`'s
`[project.optional-dependencies]` table under a per-experiment key. Users
opt in with `pip install "svae-proto[exp_NNN]"`. Never bake an
experiment's deps into the core `dependencies` list — that defeats the
whole point of keeping the core slim.

Example: `exp_001` needs `transformers`, `datasets`, `sentencepiece` for
its tokenizer-aware dataset. These go under:

```toml
[project.optional-dependencies]
exp_001 = ["transformers>=4.40.0", "datasets>=2.16.0", "sentencepiece>=0.2.0"]
```

so a user can `pip install "svae-proto[exp_001]"` to grab exactly what
they need.

## Current experiments

| ID  | Slug                       | Status     | Extras key |
|-----|----------------------------|------------|------------|
| 001 | `vocab_trigram_recall`     | scaffolded | `exp_001` |

```bash
python -m svae_proto.exp_001_vocab_trigram_recall.run --variant proto_64
python -m svae_proto.exp_001_vocab_trigram_recall.run --variant freckles_64
python -m svae_proto.exp_001_vocab_trigram_recall.run --variant fresnel_128
```

## Importing across experiments

Experiments may import from each other (e.g. `exp_002` reuses `exp_001`'s
dataset) *only* by direct relative path:
`from svae_proto.exp_001_vocab_trigram_recall.dataset import ...`. There
is no shared `svae_proto/lib/` — graduate to `geolip_svae` first if two
experiments both need the same code.
