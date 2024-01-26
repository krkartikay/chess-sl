import torch


def calculate_precision(move_probs: torch.Tensor, valid_moves: torch.Tensor, threshold: float = 0.5):
    """
    Calculates Precision given predicted move_probs and actual valid_moves.

    Precision = True positives / Predicted Positives
              = 1 - False Positives / Predicted Positives
              = 1 - False Discovery Rate (FDR)
              = True Positive Rate
    """
    predicted_positives = move_probs >= threshold
    true_positives = predicted_positives & valid_moves.bool()
    return (true_positives.sum() / predicted_positives.sum()).item()

def calculate_recall(move_probs: torch.Tensor, valid_moves: torch.Tensor, threshold: float = 0.5):
    """
    Calculates Recall given predicted move_probs and actual valid_moves.

    Recall = True positives / Actual Positives
           = True Positive Rate (TPR)
           = 1 - False Negatives / Actual Positives
           = 1 - False Negative Rate (FNR, Miss Rate)
    """
    predicted_positives = move_probs >= threshold
    true_positives = predicted_positives & valid_moves.bool()
    return (true_positives.sum() / valid_moves.sum()).item()
