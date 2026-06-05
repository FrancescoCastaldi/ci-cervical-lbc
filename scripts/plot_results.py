"""
Confronto finale: carica le metriche di tutti i metodi e genera grafici.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.data.dataset import load_config
from src.plots.visualize import plot_metrics

config = load_config()
results_dir = Path(config["eval"]["results_dir"])

plot_metrics(results_dir, save_path=results_dir / "comparison.png")
