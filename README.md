# test_torch_hub

Proof of concept: can `torch.hub` serve as the entry point for the H.IAAC model zoo?

It exposes the 36 SSL + encoder models published on
[Zenodo](https://zenodo.org/records/19301058), each with a `pretrained` and a
`finetuned` checkpoint. The architectures live in
[Minerva](https://github.com/discovery-unicamp/Minerva), not in this repository.

**All 36 models were validated against Table III of the paper: mean deviation
0.030 pp, largest 0.10 pp, none beyond 0.5 pp.**

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install minerva==0.3.10b0
```

```python
import torch

# architecture only, randomly initialised
model = torch.hub.load(".", "cnnpff", source="local")

# backbone with the SSL pretrained weights, downloaded from Zenodo and cached
backbone = torch.hub.load(".", "lfr_ts2vec_ms", role="pretrained", source="local")

# full classifier: finetuned backbone + MLP head, 6 activity classes
clf = torch.hub.load(".", "lfr_ts2vec_ms", role="finetuned", head=True, source="local")

clf.eval()
with torch.no_grad():
    logits = clf(torch.randn(1, 6, 60))   # 6 IMU channels, 60 samples
print(logits.argmax(1))                   # 0 sit, 1 stand, 2 walk, 3 up, 4 down, 5 run
```

To see what is available:

```python
import hubconf

hubconf.list_models()                 # all 36
hubconf.list_models(dataset="ms")     # filter by dataset, ssl or encoder
```

Model keys follow `<ssl>_<encoder>_<dataset>`:

- ssl: `lfr`, `tfc`, `diet`
- encoder: `ts2vec`, `cnnpff`, `resnetse5`, `rnn`, `tstcc`, `imutransformer`
- dataset: `ms`, `uci`, `kh`, `rwthigh`, `rwwaist`, `wisdm`

## Files

| File | Role |
|---|---|
| `hubconf.py` | torch.hub entrypoints. No model is hardcoded, it only reads the catalog |
| `models.yaml` | Catalog: 6 encoder architectures, 36 models, Zenodo URLs for both roles |
| `evaluate_daghar.py` | Evaluate one model on a DAGHAR test split |
| `validate_all.py` | Sweep the whole catalog against the paper |
| `test_torch_hub.ipynb` | Walkthrough of every test, with the findings |

Only 5 functions are written by hand in `hubconf.py`. The other 42 entrypoints
are generated at import time from `models.yaml`, so adding a weight to the zoo
means adding a YAML block, not editing code.

## Reproducing the paper numbers

The evaluation scripts need the DAGHAR data. From the parent repository:

```bash
./download_data.sh
```

Use that script, not `prepare_data.py --only_daghar`: the latter downloads every
file in the Zenodo record, including `baseline_view.zip`, and extracts both views
into the same folder. The top-level directories then hold 150-sample windows and
`Reshape((6, 60))` fails.

```bash
python evaluate_daghar.py                    # lfr_ts2vec_ms on MotionSense
python evaluate_daghar.py tfc_ts2vec_kh      # another model
python evaluate_daghar.py lfr_ts2vec_ms uci  # cross-dataset transfer

python validate_all.py                       # all 36, about 11 min on one GPU
```

`validate_all.py` writes `validation_results.csv` with accuracy, macro F1 and the
deviation from the paper for every model.

## Findings

- `dependencies = ["torch", "minerva", "yaml"]` is **only a check**, not an
  install. A missing entry raises `RuntimeError: Missing dependencies: minerva`.
  torch.hub never runs pip and has no notion of version ranges, so an
  incompatible Minerva passes the check and fails later at `load_state_dict`.
- Entry names are **module names**, not PyPI names: `yaml`, not `PyYAML`.
- `torch.hub.list()` has no `source` parameter, it only works against GitHub.
  To list locally, import `hubconf` directly.
- Loading a class that lives outside this repository works: `hubconf.py` is a
  normal Python module and can import from `minerva.models.nets`.
- Dynamically created entrypoints show up in the listing, since torch.hub just
  scans the module for public callables. Watch out for imported names leaking
  in as entrypoints (`from pathlib import Path` would be listed as a model).
- `minerva.__version__` reports `0.3.8-beta` while pip installed `0.3.10b0`.
  Use `importlib.metadata.version` for any compatibility check.
- Zenodo checkpoints load through `torch.hub.load_state_dict_from_url`, cached
  under `~/.cache/torch/hub/checkpoints`. Lightning checkpoints want
  `weights_only=False` on torch >= 2.6.

The conclusion for the model zoo design: `hubconf.py` solves the ergonomics but
not version compatibility, so it works best as a thin shortcut over a pip package
that carries the Minerva pin.

## Running the tests

The notebook needs a second environment without Minerva for test 1:

```bash
python -m venv .venv-hub-nominerva
.venv-hub-nominerva/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-hub-nominerva/bin/pip install pyyaml
```

Then run `test_torch_hub.ipynb`.
