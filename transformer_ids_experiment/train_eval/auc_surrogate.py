import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseAUCLoss(nn.Module):
    """
    AUC surrogate loss with sampled positive/negative pairs.
    Works for imbalanced binary classification.
    """
    def __init__(self, num_pairs: int = 2048, margin: float = 0.0):
        super().__init__()
        self.num_pairs = num_pairs
        self.margin = margin

    @torch.no_grad()
    def _sample_pairs(self, y):
        # y: [B] 0/1
        pos_idx = (y == 1).nonzero(as_tuple=False).squeeze(-1)
        neg_idx = (y == 0).nonzero(as_tuple=False).squeeze(-1)
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return None, None

        # sample pairs
        p = pos_idx[torch.randint(0, pos_idx.numel(), (self.num_pairs,), device=y.device)]
        n = neg_idx[torch.randint(0, neg_idx.numel(), (self.num_pairs,), device=y.device)]
        return p, n

    def forward(self, logits, y):
        # logits: [B], y: [B]
        p, n = self._sample_pairs(y)
        if p is None:
            return logits.new_tensor(0.0)

        # want logits[p] > logits[n]
        diff = logits[n] - logits[p] + self.margin
        return F.softplus(diff).mean()   # smooth hinge