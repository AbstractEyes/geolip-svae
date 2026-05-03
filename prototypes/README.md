# prototypes/

Self-contained experimental scaffolds. Each subdirectory is one experiment.

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
# prototypes/NNN_xxx/run.py
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
prototypes/
    NNN_short_slug/          # NNN = monotonic 3-digit experiment number
        __init__.py
        NOTES.md             # hypothesis, success criteria, results
        dataset.py           # any new dataset class + factory
        eval.py              # any new metric / probe
        cfg.py               # the cfg dict(s) under test
        run.py               # `python -m prototypes.NNN_x.run` entry point
```

`NOTES.md` is the most important file — it captures *what we're testing
and what would convince us to graduate this into the package*. Keep it
honest. If an experiment falls through, leave NOTES.md updated with what
went wrong so future-you doesn't repeat it.

## Current experiments

| ID  | Slug                       | Status   |
|-----|----------------------------|----------|
| 001 | `vocab_trigram_recall`     | scaffolded |

## Importing across prototypes

Prototypes may import from each other (e.g. 002 reuses 001's dataset)
*only* by direct relative path: `from prototypes.001_vocab_trigram_recall.dataset import ...`.
There is no shared `prototypes/lib/` — graduate to `geolip_svae` first if
two prototypes both need the same code.
