import torch
import torch.nn as nn
import torch.nn.functional as F


def get_pos_weight(train_graphs, max_weight=12.0):
    """
    Compute per-task positive-class weights.

    This is important because realistic grid-risk labels are highly imbalanced:
    faults and congestion are rare compared with normal operation.
    """

    y = torch.cat([g.y for g in train_graphs], dim=0)

    pos = y.sum(dim=0)
    neg = y.shape[0] - pos

    raw_weight = neg / pos.clamp(min=1.0)

    pos_weight = torch.clamp(
        raw_weight,
        min=1.0,
        max=max_weight,
    )

    return pos_weight


class WeightedFocalLoss(nn.Module):
    """
    Weighted focal loss for imbalanced multi-label grid-risk prediction.

    Combines:
    - BCEWithLogitsLoss positive weighting
    - focal modulation for hard rare-event examples

    This is reviewer-safe for rare-event prediction.
    """

    def __init__(
        self,
        pos_weight,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()

        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        probs = torch.sigmoid(logits)

        pt = torch.where(
            targets == 1,
            probs,
            1.0 - probs,
        )

        focal_factor = (1.0 - pt).pow(self.gamma)

        loss = focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


def make_loss(train_graphs, device):
    pos_weight = get_pos_weight(
        train_graphs,
        max_weight=12.0,
    ).to(device)

    return WeightedFocalLoss(
        pos_weight=pos_weight,
        gamma=2.0,
        reduction="mean",
    )