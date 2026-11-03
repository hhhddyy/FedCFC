from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from fedcfc.checkpoint import load_checkpoint, restore_model
from fedcfc.evaluation import evaluate_model
from fedcfc.mnist import CLASS_NAMES, load_mnist, mnist_preprocessing
from fedcfc.training import choose_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--client-id", type=int, help="zero-based client ID; default: all clients")
    parser.add_argument("--output", help="also save the JSON evaluation report")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output and Path(args.output).expanduser().resolve() == Path(args.checkpoint).expanduser().resolve():
        parser.error("evaluation output must not overwrite the checkpoint")
    checkpoint = load_checkpoint(args.checkpoint)
    preprocessing = checkpoint["preprocessing"]
    if preprocessing.get("dataset") != "mnist":
        raise ValueError("checkpoint is not a sequential MNIST model")
    expected = mnist_preprocessing(preprocessing.get("sequence_mode"))
    if preprocessing != expected or checkpoint["model_config"]["input_size"] != expected["input_size"]:
        raise ValueError("checkpoint has incompatible MNIST preprocessing")
    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("checkpoint has incompatible MNIST classes")
    client_ids = list(range(len(checkpoint["client_states"])))
    if args.client_id is not None:
        if args.client_id not in client_ids:
            parser.error(f"--client-id must be in [0, {len(client_ids) - 1}]")
        client_ids = [args.client_id]
    test = load_mnist(args.data_dir, train=False, download=args.download,
                      sequence_mode=preprocessing["sequence_mode"])
    loader = DataLoader(test, batch_size=args.batch_size)
    device = choose_device(args.device)
    results = []
    for client_id in client_ids:
        model = restore_model(checkpoint, client_id)
        result = evaluate_model(model, loader, device=device, num_classes=len(CLASS_NAMES))
        results.append({"client_id": client_id, **result})
        del model
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "round": checkpoint["round"],
        "dataset": "mnist", "sequence_mode": preprocessing["sequence_mode"],
        "device": str(device), "training_mode": checkpoint["run_config"]["training_mode"],
        "partition": checkpoint["run_config"]["partition"], "test_samples": len(test),
        "class_names": CLASS_NAMES, "confusion_matrix_axes": {"rows": "true class", "columns": "predicted class"},
        "clients": results, "mean_accuracy": sum(r["accuracy"] for r in results) / len(results),
        "mean_loss": sum(r["loss"] for r in results) / len(results),
    }
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
