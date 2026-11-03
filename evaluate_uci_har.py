from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from fedcfc.checkpoint import load_checkpoint, restore_model
from fedcfc.data import SIGNALS, load_uci_har_test
from fedcfc.evaluation import evaluate_model
from train_uci_har import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--client-id", type=int, help="zero-based client ID; default: evaluate every client")
    parser.add_argument("--output", help="also write the JSON evaluation report to this path")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    checkpoint = load_checkpoint(args.checkpoint)
    client_ids = list(range(len(checkpoint["client_states"])))
    if args.client_id is not None:
        if args.client_id not in client_ids:
            parser.error(f"--client-id must be in [0, {len(client_ids) - 1}]")
        client_ids = [args.client_id]
    preprocessing = checkpoint["preprocessing"]
    if preprocessing["signals"] != list(SIGNALS):
        raise ValueError("checkpoint signal order does not match the UCI HAR loader")
    test_x, test_y = load_uci_har_test(
        args.data_dir, preprocessing["mean"], preprocessing["std"], download=args.download
    )
    if test_x.shape[1:] != (preprocessing["sequence_length"], checkpoint["model_config"]["input_size"]):
        raise ValueError("test window shape does not match the checkpoint")
    timespans = torch.full(test_x.shape[:2], preprocessing["timespan_seconds"])
    loader = DataLoader(TensorDataset(test_x, test_y, timespans), batch_size=args.batch_size)
    device = choose_device(args.device)
    results = []
    for client_id in client_ids:
        model = restore_model(checkpoint, client_id)
        result = evaluate_model(model, loader, device=device,
                                num_classes=checkpoint["model_config"]["num_classes"])
        results.append({"client_id": client_id, **result})
        del model
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "round": checkpoint["round"],
        "device": str(device), "training_mode": checkpoint["run_config"]["training_mode"],
        "partition": checkpoint["run_config"]["partition"],
        "bounded": checkpoint["run_config"]["bounded"],
        "test_samples": len(test_y), "class_names": checkpoint["class_names"],
        "confusion_matrix_axes": {"rows": "true class", "columns": "predicted class"},
        "clients": results,
        "mean_accuracy": sum(item["accuracy"] for item in results) / len(results),
        "mean_loss": sum(item["loss"] for item in results) / len(results),
    }
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        path = Path(args.output).expanduser()
        if path.resolve() == Path(args.checkpoint).resolve():
            raise ValueError("evaluation output must not overwrite the checkpoint")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
