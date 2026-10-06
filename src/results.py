"""Saving each run's report to disk."""
import os
from datetime import datetime


def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """
    Writes the run's settings and full report to `results_dir/run_<timestamp>.txt` (the
    folder is created if needed) and returns the path.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    cv = config["cv"]
    tuning = config.get("tuning", {})
    tuning_line = (f"Optuna, {tuning['n_trials']} trials x {tuning['n_splits']} folds, nested CV  "
                   f"random_state={tuning['random_state']}" if tuning.get("enabled") else "off")
    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}  "
        f"scaler={config['preprocessing']['scaler']}\n"
        f"CV: {cv['n_splits']} stratified folds  shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state')}  metric={cv.get('scoring', 'accuracy')}\n"
        f"Tuning: {tuning_line}\n"
        f"Locked test set: size={config['test_set']['size']}  "
        f"random_state={config['test_set']['random_state']}  (not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)
    return path