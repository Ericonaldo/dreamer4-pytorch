"""Plot env return vs eval.action_horizon (mean line, min–max shade).

Usage: python -m dreamer4.plot_action_horizon_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "action_horizon_eval_step142000.json"
OUT = ROOT / "analysis" / "action_horizon_eval_step142000.png"


def main() -> None:
    payload = json.loads(DATA.read_text())
    rows = sorted(payload["results"], key=lambda r: r["action_horizon"])
    h = np.array([r["action_horizon"] for r in rows], dtype=np.float64)
    mean = np.array([r["return_mean"] for r in rows])
    lo = np.array([r["return_min"] for r in rows])
    hi = np.array([r["return_max"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(h, lo, hi, alpha=0.25, color="#4C72B0", label="min–max (8 ep)")
    ax.plot(h, mean, "o-", color="#4C72B0", linewidth=2, markersize=8, label="mean return")
    ax.set_xlabel("eval.action_horizon (open-loop steps per forward)")
    ax.set_ylabel("DMC Walker Walk return")
    ax.set_title("BC policy env return vs open-loop horizon (step 142000)")
    ax.set_xticks(h)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
