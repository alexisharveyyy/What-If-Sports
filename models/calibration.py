"""Calibration and uncertainty quantification for the NIL multi-task models.

Provides:
- CalibratedNILModel: wraps a trained NILLSTMModel/NILTransformerModel and
  applies a learned scalar temperature to the tier logits at inference time.
- expected_calibration_error: scalar miscalibration metric.
- reliability_diagram: matplotlib plot of confidence vs. accuracy.
- mc_dropout_predict: predictive mean/std/95% CI for both tier probabilities
  and valuation regression via Monte Carlo Dropout.
- regression_interval_coverage: coverage of 95% predictive intervals against
  ground truth — a calibration metric for the regression head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


# ---------------------------------------------------------------------------
# Calibrated model wrapper (temperature scaling on the tier head only)
# ---------------------------------------------------------------------------

class CalibratedNILModel(nn.Module):
    """Wraps a trained NIL multi-task model and applies temperature scaling to
    the tier logits. The valuation regression head is passed through unchanged.

    Usage:
        cal = CalibratedNILModel(model).to(device)
        cal.fit_temperature(val_loader, device)
        tier_logits_cal, value_pred = cal(X_seq)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Single learnable scalar, init to 1.0 (no change).
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tier_logits, value_pred = self.model(x)
        return tier_logits / self.temperature, value_pred

    @torch.no_grad()
    def _collect(self, val_loader, device):
        """Gather logits and tier targets from the underlying (uncalibrated) model."""
        self.model.eval()
        all_logits, all_targets = [], []
        for X_seq, y_tier, _y_value in val_loader:
            X_seq = X_seq.to(device)
            y_tier = y_tier.to(device)
            tier_logits, _ = self.model(X_seq)
            all_logits.append(tier_logits)
            all_targets.append(y_tier)
        return torch.cat(all_logits), torch.cat(all_targets)

    def fit_temperature(self, val_loader, device, max_iter: int = 50) -> float:
        """Learn T by minimizing NLL of the tier head on the validation set."""
        logits, targets = self._collect(val_loader, device)
        nll = nn.CrossEntropyLoss()
        # LBFGS is the standard choice for this 1-D, smooth, well-behaved problem.
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll(logits / self.temperature, targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        T = float(self.temperature.detach().cpu().item())
        print(f"Learned temperature: T = {T:.4f}")
        if T > 1.05:
            print("  (T > 1 -> original model was over-confident; T softens probs.)")
        elif T < 0.95:
            print("  (T < 1 -> original model was under-confident; T sharpens probs.)")
        else:
            print("  (T ~= 1 -> original model was already well-calibrated.)")
        return T


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE for a multi-class classifier. Lower is better.

    Args:
        y_true: integer ground-truth labels, shape [N].
        y_probs: probabilities, shape [N, num_classes].
        n_bins: number of equal-width confidence bins on [0, 1].
    """
    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece / len(y_true))


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def reliability_diagram(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: str | None = None,
    equal_mass: bool = True,
):
    """Plot per-bin accuracy vs. confidence. Perfect calibration = diagonal.

    With equal_mass=True, bin edges are placed at data quantiles so each bin
    holds roughly the same number of samples. Edges are NOT forced to span
    [0, 1] -- bars are drawn in the actual data region, while the axes still
    show the full [0, 1] range so the perfect-calibration diagonal is visible.
    """
    import matplotlib.pyplot as plt

    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    if equal_mass:
        edges = np.unique(np.quantile(confidences, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            # Degenerate case: every prediction has the exact same confidence.
            c = float(confidences[0])
            edges = np.array([max(0.0, c - 0.01), min(1.0, c + 0.01)])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    actual_n_bins = len(edges) - 1

    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    bin_acc = np.zeros(actual_n_bins)
    bin_conf = np.zeros(actual_n_bins)
    for i in range(actual_n_bins):
        mask = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if i == 0:
            mask = mask | (confidences == edges[0])
        if mask.sum() > 0:
            bin_acc[i] = accuracies[mask].mean()
            bin_conf[i] = confidences[mask].mean()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(
        centers, bin_acc, width=widths,
        edgecolor="black", alpha=0.75, label="Accuracy",
    )
    gap = bin_conf - bin_acc
    ax.bar(
        centers, gap, width=widths,
        bottom=bin_acc, edgecolor="black", alpha=0.35,
        color="red", label="Gap (over/under-confidence)",
    )
    if equal_mass:
        # Confidence values are clustered; zoom the axes to where the data lives
        # so bars are actually visible. Add a small margin and keep aspect square.
        margin = max(0.02, 0.1 * (edges.max() - edges.min()))
        x_lo = max(0.0, edges.min() - margin)
        x_hi = min(1.0, edges.max() + margin)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(x_lo, x_hi)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved reliability diagram -> {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# MC Dropout predictive uncertainty for both heads
# ---------------------------------------------------------------------------

def _enable_dropout(model: nn.Module):
    """Force every Dropout layer back into train mode while leaving everything
    else (BatchNorm, LSTM stateful behavior, etc.) in eval mode."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 30,
) -> dict:
    """Run `n_samples` stochastic forward passes with dropout active.

    Works with any model returning (tier_logits, value_pred). Returns a dict:
      tier_mean    [N, C]  mean tier-class probabilities (softmax averaged)
      tier_std     [N, C]  per-class standard deviation across samples
      tier_ci_low  [N, C]  2.5th percentile (lower 95% CI bound)
      tier_ci_high [N, C]  97.5th percentile (upper 95% CI bound)
      value_mean   [N]     mean valuation prediction
      value_std    [N]     std of valuation across samples
      value_ci_low [N]     2.5th percentile of valuation
      value_ci_high[N]     97.5th percentile of valuation
    """
    model.eval()
    _enable_dropout(model)

    tier_samples, value_samples = [], []
    for _ in range(n_samples):
        tier_logits, value_pred = model(x)
        tier_samples.append(F.softmax(tier_logits, dim=-1))
        value_samples.append(value_pred)
    tier_stack = torch.stack(tier_samples, dim=0)    # [n_samples, N, C]
    value_stack = torch.stack(value_samples, dim=0)  # [n_samples, N]

    return {
        "tier_mean": tier_stack.mean(dim=0),
        "tier_std": tier_stack.std(dim=0),
        "tier_ci_low": torch.quantile(tier_stack, 0.025, dim=0),
        "tier_ci_high": torch.quantile(tier_stack, 0.975, dim=0),
        "value_mean": value_stack.mean(dim=0),
        "value_std": value_stack.std(dim=0),
        "value_ci_low": torch.quantile(value_stack, 0.025, dim=0),
        "value_ci_high": torch.quantile(value_stack, 0.975, dim=0),
    }


# ---------------------------------------------------------------------------
# Regression calibration: coverage of 95% intervals
# ---------------------------------------------------------------------------

def regression_interval_coverage(
    y_true: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
) -> float:
    """Fraction of ground-truth values that fall inside the predicted interval.

    For a well-calibrated 95% interval this should be approximately 0.95.
    """
    inside = (y_true >= ci_low) & (y_true <= ci_high)
    return float(inside.mean())