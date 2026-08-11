"""torch.hub entrypoints for the H.IAAC model zoo (proof of concept).

Exposes the same 36 models published at https://zenodo.org/records/19301058,
each one with a `pretrained` and a `finetuned` checkpoint.

What this file is validating:

1. torch.hub can load a model whose class does NOT live in this repository,
   but in Minerva (`pip install minerva==0.3.10b0`).
2. The `dependencies` list only checks that a module is importable. It never
   installs anything: if one is missing, torch.hub raises RuntimeError.
3. The catalog lives in models.yaml and is read at run time. Adding a new
   weight does not require touching any code.
4. Entrypoints created dynamically (one per model) show up in torch.hub.list.

Usage:

    import torch
    repo = "gustavo-luz/test_torch_hub"

    torch.hub.list(repo, trust_repo=True)

    # backbone with the weights from Zenodo
    bb = torch.hub.load(repo, "lfr_ts2vec_ms", role="pretrained",
                        trust_repo=True)

    # full classifier (backbone + MLP head)
    clf = torch.hub.load(repo, "lfr_ts2vec_ms", role="finetuned",
                         head=True, trust_repo=True)

    # architecture only, no weights
    bb = torch.hub.load(repo, "cnnpff", trust_repo=True)

Locally, without cloning from GitHub:

    torch.hub.load(".", "cnnpff", source="local")
"""

import importlib

# Imported with a leading underscore so it does not become a torch.hub
# entrypoint: the listing picks up every public callable attribute of the
# module, so a plain `from pathlib import Path` would show up as a model.
from pathlib import Path as _Path

# Checked by torch.hub before the entrypoint runs. It is a MODULE name, not a
# PyPI name ("yaml", not "PyYAML"). Nothing is installed: a missing entry only
# turns into a readable RuntimeError.
dependencies = ["torch", "minerva", "yaml"]

_CONFIG = _Path(__file__).parent / "models.yaml"


# ─────────────────────────────────────────────────────────────────────────────
#  Catalog
# ─────────────────────────────────────────────────────────────────────────────
def _catalog():
    """Read models.yaml, which sits next to this file in the repository."""
    import yaml

    with open(_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(module_path, cls_name):
    return getattr(importlib.import_module(module_path), cls_name)


# ─────────────────────────────────────────────────────────────────────────────
#  Architecture (mirrors build_backbone from ssl_har_model_zoo.ipynb)
# ─────────────────────────────────────────────────────────────────────────────
def _build_encoder(encoder, catalog):
    spec = catalog["encoders"][encoder]
    kwargs = dict(spec.get("kwargs") or {})

    # the only kwarg that is a class rather than a value
    if encoder == "resnetse5":
        kwargs["residual_block_cls"] = _resolve(
            "minerva.models.nets.time_series.resnet", kwargs["residual_block_cls"]
        )

    return _resolve(spec["module"], spec["cls"])(**kwargs)


def _build_backbone(ssl, encoder, catalog):
    if ssl == "tfc":
        from minerva.models.ssl.tfc import TFC_Backbone

        adapter = None
        if encoder == "ts2vec":
            # TSEncoder returns (batch, time, features), TFC needs 2D
            from minerva.models.adapters import (
                MaxPoolingTransposingSqueezingAdapter,
            )

            adapter = MaxPoolingTransposingSqueezingAdapter(kernel_size=60)

        return TFC_Backbone(
            input_channels=6,
            TS_length=60,
            single_encoding_size=128,
            time_encoder=_build_encoder(encoder, catalog),
            frequency_encoder=_build_encoder(encoder, catalog),
            adapter=adapter,
        )

    if ssl in ("lfr", "tnc", "diet"):
        return _build_encoder(encoder, catalog)

    raise ValueError(f"unknown SSL technique: {ssl}")


def _build_classifier(backbone, spec):
    """Backbone + MLP head, same as SimpleSupervisedModel in the notebook."""
    import torch
    from minerva.models.nets.base import SimpleSupervisedModel
    from minerva.models.nets.mlp import MLP

    adapter = None
    if spec["ssl"] in ("lfr", "tnc", "diet") and spec["encoder"] == "ts2vec":
        from minerva.models.adapters import MaxPoolingTransposingSqueezingAdapter

        adapter = MaxPoolingTransposingSqueezingAdapter(kernel_size=60)

    head = MLP([spec["head_input_dim"], 128, spec["num_classes"]])
    return SimpleSupervisedModel(
        backbone=backbone,
        fc=head,
        loss_fn=torch.nn.CrossEntropyLoss(),
        adapter=adapter,
        flatten=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Weights
# ─────────────────────────────────────────────────────────────────────────────
def _download_state_dict(url):
    """Fetch the .ckpt, cached under ~/.cache/torch/hub/checkpoints."""
    import torch

    try:
        # torch >= 2.6 defaults to weights_only=True, which can reject
        # Lightning checkpoints (they hold more than tensors)
        ckpt = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", weights_only=False
        )
    except TypeError:
        ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu")

    return ckpt.get("state_dict", ckpt)


def _load_weights(model, url, backbone_only):
    """Same logic as the FromPretrained used in the notebook, without the class."""
    state = _download_state_dict(url)

    if backbone_only:
        prefix = "backbone."
        state = {
            k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
        }
        if not state:
            raise RuntimeError(
                "no key with the 'backbone.' prefix in the checkpoint, "
                "did the format change?"
            )

    model.load_state_dict(state, strict=True)
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Entrypoints
# ─────────────────────────────────────────────────────────────────────────────
def list_models(dataset=None, ssl=None, encoder=None):
    """List the catalog, with optional filters.

    >>> torch.hub.load(repo, "list_models", dataset="ms", trust_repo=True)
    """
    models = _catalog()["models"]
    out = {}
    for key, spec in models.items():
        if dataset and spec["dataset"] != dataset:
            continue
        if ssl and spec["ssl"] != ssl:
            continue
        if encoder and spec["encoder"] != encoder:
            continue
        out[key] = {
            "dataset": spec["dataset_name"],
            "accuracy": spec.get("accuracy"),
            "roles": sorted(spec["weights"]),
        }
    return out


def get_model(key, role=None, head=False):
    """Generic entrypoint: build the architecture and optionally load weights.

    Args:
        key: catalog key, e.g. "lfr_ts2vec_ms", or a bare encoder name
             ("cnnpff") for the architecture alone.
        role: "pretrained" or "finetuned". None leaves the weights random.
        head: True returns backbone + MLP head (6-class classifier).
    """
    catalog = _catalog()

    # bare encoder key, no weights published: useful as a smoke test
    if key in catalog["encoders"]:
        if role:
            raise ValueError(f"{key} is a bare encoder, it has no published weights")
        return _build_encoder(key, catalog)

    if key not in catalog["models"]:
        raise KeyError(f"{key} is not in the catalog. Use list_models() to see options.")

    spec = catalog["models"][key]
    model = _build_backbone(spec["ssl"], spec["encoder"], catalog)

    if head:
        model = _build_classifier(model, spec)

    if role:
        if role not in spec["weights"]:
            raise ValueError(
                f"invalid role: {role}. Available: {sorted(spec['weights'])}"
            )
        _load_weights(model, spec["weights"][role], backbone_only=not head)

    return model


def _make_entrypoint(key, spec):
    def entrypoint(role=None, head=False):
        return get_model(key, role=role, head=head)

    acc = spec.get("accuracy")
    entrypoint.__name__ = key
    entrypoint.__doc__ = (
        f"{spec['ssl'].upper()} + {spec['encoder']} trained on "
        f"{spec['dataset_name']}"
        + (f" (paper accuracy: {acc}%)." if acc else ".")
        + " role='pretrained'|'finetuned', head=True for the classifier."
    )
    return entrypoint


def _register_entrypoints():
    """Create one entrypoint per model and per bare encoder.

    Functions created here show up in torch.hub.list like any other, since it
    only looks at the module's public callable attributes.
    """
    catalog = _catalog()

    for key, spec in catalog["models"].items():
        globals()[key] = _make_entrypoint(key, spec)

    for name in catalog["encoders"]:
        def encoder_entrypoint(_name=name):
            """Encoder architecture, no weights."""
            return get_model(_name)

        encoder_entrypoint.__name__ = name
        encoder_entrypoint.__doc__ = (
            f"Minerva {name} encoder, randomly initialised (no pretraining)."
        )
        globals()[name] = encoder_entrypoint


_register_entrypoints()


# Entrypoint written by hand during the first test, kept as a reference for the
# simplest possible shape: one function, one Minerva class.
def harsccencoder(pretrained=False):
    print(pretrained)
    return HARSCnnEncoder(dim=125, input_channel=6, inner_conv_output_dim=128*10)