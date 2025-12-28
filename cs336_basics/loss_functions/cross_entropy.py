import torch

def cross_entropy_loss(logits_BV: torch.Tensor, y_gt_B: torch.Tensor) -> torch.Tensor:
    # Compute the log-sum-exp trick for numerical stability
    max_logits_B = torch.max(logits_BV, dim=1, keepdim=True).values
    logits_shifted_BV = logits_BV - max_logits_B
    log_sum_exp_B = torch.log(torch.sum(torch.exp(logits_shifted_BV), dim=1))

    # Compute the log probabilities for the ground truth labels
    log_probs_gt = logits_shifted_BV[torch.arange(logits_shifted_BV.shape[0]), y_gt_B] - log_sum_exp_B

    # Return the mean negative log likelihood
    return -log_probs_gt.mean()