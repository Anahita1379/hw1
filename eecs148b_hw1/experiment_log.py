import json
import csv
import time
from pathlib import Path
import matplotlib.pyplot as plt

class ExperimentLogger:
    def __init__(self, run_dir: str, config: dict):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.config = dict(config)
        self.start_time = time.time()

        self.metrics_path = self.run_dir / "metrics.csv"
        self.config_path = self.run_dir / "config.json"
        self.summary_path = self.run_dir / "summary.json"
        self.curve_path = self.run_dir / "loss_curves.png"

        self.rows = []

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        with open(self.metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "tokens_processed",
                    "split",
                    "loss",
                    "perplexity",
                    "wallclock_sec",
                ],
            )
            writer.writeheader()

    def log(self, step: int, tokens_processed: int, split: str, loss: float, perplexity: float):
        row = {
            "step": int(step),
            "tokens_processed": int(tokens_processed),
            "split": str(split),
            "loss": float(loss),
            "perplexity": float(perplexity),
            "wallclock_sec": float(time.time() - self.start_time),
        }
        self.rows.append(row)

        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "tokens_processed",
                    "split",
                    "loss",
                    "perplexity",
                    "wallclock_sec",
                ],
            )
            writer.writerow(row)

    def save_curves(self):
        train_steps = [r["step"] for r in self.rows if r["split"] == "train"]
        train_losses = [r["loss"] for r in self.rows if r["split"] == "train"]

        valid_steps = [r["step"] for r in self.rows if r["split"] == "valid"]
        valid_losses = [r["loss"] for r in self.rows if r["split"] == "valid"]

        plt.figure(figsize=(8, 5))
        if train_steps:
            plt.plot(train_steps, train_losses, label="train loss")
        if valid_steps:
            plt.plot(valid_steps, valid_losses, label="valid loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training / Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.curve_path)
        plt.close()

    def write_summary(self):
        valid_rows = [r for r in self.rows if r["split"] == "valid"]
        train_rows = [r for r in self.rows if r["split"] == "train"]

        best_valid = min(valid_rows, key=lambda r: r["loss"]) if valid_rows else None
        last_train = train_rows[-1] if train_rows else None

        summary = {
            "total_wallclock_sec": time.time() - self.start_time,
            "num_logged_points": len(self.rows),
            "best_valid": best_valid,
            "last_train": last_train,
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)