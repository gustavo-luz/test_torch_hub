"""Validate every model in the zoo against the numbers reported in the paper.

Accuracy is reported to one decimal, the precision of Table III. A deviation of
0.1 is one step of the last printed digit, so anything within 0.1 pp counts as
reproducing the published number.

For each model: load the finetuned checkpoint through torch.hub, build the full
classifier and evaluate it on the test split of the dataset it was trained on.
The reference accuracy comes from Table III and lives in models.yaml.

Each test split is read once, decoded into tensors and kept in memory, so the
36 models share 6 dataset reads instead of re-reading a CSV per model.

Usage:
    python validate_all.py                  # all 36
    python validate_all.py --ssl tfc        # one SSL technique
    python validate_all.py --dataset ms     # one dataset

Writes validation_results.csv and prints a markdown table.
"""

import argparse
import csv
import time
import traceback
from pathlib import Path

import torch
import yaml
from minerva.data.datasets.base import SimpleDataset
from minerva.data.readers.csv_reader import CSVReader
from minerva.transforms import CastTo, Reshape, TransformPipeline

REPO = Path(__file__).parent
DATA_ROOT = REPO.parent / "shared_data" / "daghar" / "standardized_view"

FOLDER = {
    "ms": "MotionSense",
    "uci": "UCI",
    "kh": "KuHar",
    "rwthigh": "RealWorld_thigh",
    "rwwaist": "RealWorld_waist",
    "wisdm": "WISDM",
}
COLS = ["accel-x-*", "accel-y-*", "accel-z-*", "gyro-x-*", "gyro-y-*", "gyro-z-*"]
TRANSFORM = [TransformPipeline([Reshape((6, 60)), CastTo("float32")]), CastTo("int64")]

_SPLITS = {}


def test_split(dataset, device):
    """Whole test split as two tensors, read once and cached in memory.

    The Minerva readers go through the CSV row by row, which is fine for a
    single evaluation but costs minutes when 36 models each walk the same file.
    """
    key = (dataset, str(device))
    if key not in _SPLITS:
        root = DATA_ROOT / FOLDER[dataset]
        ds = SimpleDataset(
            readers=[
                CSVReader(path=root / "test.csv", columns_to_select=COLS),
                CSVReader(
                    path=root / "test.csv",
                    columns_to_select="standard activity code",
                    cast_to="int64",
                ),
            ],
            transforms=TRANSFORM,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=512, num_workers=4)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x)
            ys.append(y)
        _SPLITS[key] = (torch.cat(xs).to(device), torch.cat(ys).to(device))
    return _SPLITS[key]


def evaluate(key, spec, device):
    model = torch.hub.load(str(REPO), key, role="finetuned", head=True, source="local")
    model.eval().to(device)

    x, y = test_split(spec["dataset"], device)

    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), 256):
            pred = model(x[i : i + 256]).argmax(1)
            correct += (pred == y[i : i + 256]).sum().item()

    return {
        "accuracy": 100 * correct / len(y),
        "params": sum(p.numel() for p in model.parameters()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl", help="filter by technique: lfr, tfc, diet")
    ap.add_argument("--dataset", help="filter by dataset: ms, uci, kh, ...")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    catalog = yaml.safe_load(open(REPO / "models.yaml", encoding="utf-8"))["models"]
    targets = {
        k: v
        for k, v in catalog.items()
        if (not args.ssl or v["ssl"] == args.ssl)
        and (not args.dataset or v["dataset"] == args.dataset)
    }

    print(f"validating {len(targets)} models on {args.device}\n", flush=True)
    rows, failures = [], []
    started = time.time()

    for i, (key, spec) in enumerate(sorted(targets.items()), 1):
        t0 = time.time()
        try:
            r = evaluate(key, spec, args.device)
            paper = spec.get("accuracy")
            # the paper reports one decimal, so compare at that precision
            measured = round(r["accuracy"], 1)
            delta = round(measured - paper, 1) if paper else None
            rows.append(
                {
                    "model": key,
                    "ssl": spec["ssl"],
                    "encoder": spec["encoder"],
                    "dataset": spec["dataset_name"],
                    "paper": paper,
                    "measured": measured,
                    "delta": delta,
                    "params": r["params"],
                    "seconds": round(time.time() - t0, 1),
                }
            )
            flag = "==" if delta == 0 else "  "
            print(
                f"[{i:>2}/{len(targets)}] {flag} {key:<28} "
                f"paper={paper:<5} measured={measured:<5} delta={delta:+.1f}",
                flush=True,
            )
        except Exception as exc:
            failures.append((key, f"{type(exc).__name__}: {exc}"))
            print(f"[{i:>2}/{len(targets)}] FAILED {key}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    out = REPO / "validation_results.csv"
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"\n{'=' * 70}")
    print(f"validated: {len(rows)}/{len(targets)} in {(time.time() - started) / 60:.1f} min")
    if rows:
        exact = [r for r in rows if r["delta"] == 0]
        off = [r for r in rows if r["delta"] != 0]
        within = [r for r in rows if abs(r["delta"]) <= 0.1]
        print(f"within 0.1 pp of the paper : {len(within)}/{len(rows)}")
        print(f"exact match                : {len(exact)}/{len(rows)}")
        print(f"largest deviation          : {max(abs(r['delta']) for r in rows):.1f} pp")
        for r in off:
            print(
                f"   {r['model']:<28} paper={r['paper']} "
                f"measured={r['measured']} ({r['delta']:+.1f})"
            )
    if failures:
        print(f"\nfailed: {len(failures)}")
        for key, err in failures:
            print(f"   {key}: {err}")
    print(f"\ncsv: {out}")

    print("\n| Model | Dataset | Paper | Measured | Delta |")
    print("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -(x["paper"] or 0)):
        print(
            f"| `{r['model']}` | {r['dataset']} | {r['paper']}% | "
            f"{r['measured']}% | {r['delta']:+.1f} |"
        )


if __name__ == "__main__":
    main()
