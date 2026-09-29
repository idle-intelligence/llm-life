"""Python port of the scoring subset of crates/llm-life/src/score.rs used by
the PyTorch training/eval loops: accuracy, IoU, alive/dead recall. Takes the
truth `Grid` and a `pred` array (0/1 per cell), matching `score::score`'s
conventions on ties (both-empty IoU = 1.0, etc.)."""

from __future__ import annotations

import numpy as np


def score(truth, pred: np.ndarray, p_alive: np.ndarray) -> dict:
    truth_cells = truth.cells
    n = len(truth_cells)
    wrong = int((truth_cells != pred).sum())
    true_live = int(truth_cells.sum())
    model_live = int(pred.sum())

    hit = int(((truth_cells != 0) & (pred != 0)).sum())
    dead = n - true_live
    union = true_live + model_live - hit
    recall = 1.0 if true_live == 0 else hit / true_live
    fp = model_live - hit
    tn = dead - fp
    dead_recall = 1.0 if dead == 0 else tn / dead
    iou = 1.0 if union == 0 else hit / union
    accuracy = 1.0 - wrong / n
    return {
        "accuracy": accuracy,
        "iou": iou,
        "alive_recall": recall,
        "dead_recall": dead_recall,
        "wrong_cells": wrong,
        "true_live": true_live,
        "model_live": model_live,
    }
