"""Evaluate one zoo model on DAGHAR and compare it with the paper.

Loads the model through torch.hub (the hubconf.py in this repository) and
rebuilds the data pipeline from ssl_har_model_zoo.ipynb: a CSVReader over the
6 IMU columns, reshape to (6, 60), label in "standard activity code".

Usage:
    python evaluate_daghar.py                      # lfr_ts2vec_ms on MotionSense
    python evaluate_daghar.py tfc_ts2vec_kh        # another model, its own dataset
    python evaluate_daghar.py lfr_ts2vec_ms uci    # cross-dataset transfer
"""

import sys
from pathlib import Path

import torch
import torchmetrics
import yaml
from minerva.data.datasets.base import SimpleDataset
from minerva.data.readers.csv_reader import CSVReader
from minerva.transforms import CastTo, Reshape, TransformPipeline

REPO = Path(__file__).parent

# Layout produced by ./download_data.sh, which fetches standardized_view.zip
# only. If you use prepare_data.py --only_daghar instead, it also pulls
# baseline_view.zip into the same folder: the top-level directories then hold
# 150-sample windows and the 60-sample standardized view ends up one level
# deeper. Use download_data.sh.
DATA_ROOT = REPO.parent / "shared_data" / "daghar" / "standardized_view"

# same mapping as the official notebook
FOLDER = {
    "ms": "MotionSense",
    "uci": "UCI",
    "kh": "KuHar",
    "rwthigh": "RealWorld_thigh",
    "rwwaist": "RealWorld_waist",
    "wisdm": "WISDM",
}
ACTIVITIES = {0: "sit", 1: "stand", 2: "walk", 3: "stair up", 4: "stair down", 5: "run"}

COLS = ["accel-x-*", "accel-y-*", "accel-z-*", "gyro-x-*", "gyro-y-*", "gyro-z-*"]
TRANSFORM = [TransformPipeline([Reshape((6, 60)), CastTo("float32")]), CastTo("int64")]


def test_dataset(dataset):
    root = DATA_ROOT / FOLDER[dataset]
    return SimpleDataset(
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


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "lfr_ts2vec_ms"
    catalog = yaml.safe_load(open(REPO / "models.yaml", encoding="utf-8"))["models"]
    spec = catalog[key]
    dataset = sys.argv[2] if len(sys.argv) > 2 else spec["dataset"]

    print(f"model     : {key}  ({spec['ssl'].upper()} + {spec['encoder']})")
    print(f"evaluating: {FOLDER[dataset]}  (test.csv)")
    print(f"paper     : {spec.get('accuracy')}% on its training dataset\n")

    # the full classifier: finetuned backbone + MLP head
    model = torch.hub.load(str(REPO), key, role="finetuned", head=True, source="local")
    model.eval()

    loader = torch.utils.data.DataLoader(
        test_dataset(dataset), batch_size=64, num_workers=4
    )
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=6)
    confusion = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=6)

    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1)
            acc.update(pred, y)
            confusion.update(pred, y)

    # one decimal, the precision the paper reports
    measured = round(acc.compute().item() * 100, 1)
    print(f"accuracy  : {measured}%")

    if dataset == spec["dataset"] and spec.get("accuracy"):
        delta = round(measured - spec["accuracy"], 1)
        verdict = "identical to the paper" if delta == 0 else f"{delta:+.1f} pp"
        print(f"vs paper  : {verdict}")

    print("\nconfusion matrix (row = true, column = predicted)")
    cm = confusion.compute()
    print("            " + "".join(f"{ACTIVITIES[i]:>12}" for i in range(6)))
    for i in range(6):
        print(f"{ACTIVITIES[i]:>12}" + "".join(f"{int(v):>12}" for v in cm[i]))


if __name__ == "__main__":
    main()
