"""Measure Section II-B filter responses and save the figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.filters import (  # noqa: E402
    filter_ecg,
    filter_ppg,
    filter_tonometry,
    specification_check,
    stage_response,
)


def main() -> None:
    measured = specification_check()
    for name, value in measured.items():
        print(f"{name}: {value:.2f}")

    fs = 500.0
    seconds = np.arange(0, 10, 1 / fs)
    ecg = np.sin(2 * np.pi * 1.2 * seconds) + 0.2 * np.sin(2 * np.pi * 60 * seconds)
    ppg = np.sin(2 * np.pi * 1.2 * seconds) + 0.05 * np.sin(2 * np.pi * 20 * seconds)
    pressure = 2900 + 20 * np.sin(2 * np.pi * 1.2 * seconds)
    filtered = {
        "ECG": filter_ecg(ecg),
        "PPG": filter_ppg(ppg),
        "Tonometry": filter_tonometry(pressure),
    }
    for name, values in filtered.items():
        print(f"{name} finite={np.isfinite(values).all()} std={np.std(values):.4f}")

    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)
    panels = [
        ("ECG", [("ecg", "dc"), ("ecg", "lowpass"), ("ecg", "notch")], [0.1, 40, 45, 60]),
        ("PPG", [("ppg", "highpass"), ("ppg", "lowpass")], [0.25, 10, 12]),
        ("Tonometry", [("tonometry", "highpass"), ("tonometry", "lowpass")], [0.2, 0.3, 22, 26]),
    ]
    for axis, (title, stages, marks) in zip(axes, panels):
        for kind, stage in stages:
            response = stage_response(kind, stage)
            axis.plot(response["hz"], response["db"], label=stage)
        for mark in marks:
            axis.axvline(mark, color="0.6", linewidth=0.6, linestyle="--")
        axis.set_xlim(0, 80 if title == "ECG" else 30)
        axis.set_ylim(-80, 5)
        axis.set_ylabel("Gain (dB)")
        axis.set_title(title + " single-pass response")
        axis.legend(loc="lower left", fontsize=8)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frequency (Hz)")
    figure.tight_layout()
    destination = ROOT / "outputs" / "figures" / "filter_responses.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
