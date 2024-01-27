import argparse
import torch
import torch.nn.functional as F

from config import *
from experiment import Experiment
from train import load_data, train_model
from evaluate import evaluate_model_vs_random, calculate_precision, calculate_recall

device = 'cuda' if torch.cuda.is_available() else 'cpu'
positions, valid_moves = load_data()
positions = positions.to(device)
valid_moves = valid_moves.to(device)

def experiment_main():
    # Train model and evaluate it
    results_dict, model = train_model(positions, valid_moves)
    avg_moves, score, all_moves = evaluate_model_vs_random(model)
    results_dict['avg_moves'] = avg_moves
    results_dict['score'] = score
    all_moves_hist = [0]*10
    with torch.no_grad():
        predicted_logits = model(positions)
        predicted_moves = F.softmax(predicted_logits, dim=1)
    precision = calculate_precision(predicted_moves, valid_moves, 0.01)
    recall = calculate_recall(predicted_moves, valid_moves, 0.01)
    results_dict['precision'] = precision
    results_dict['recall'] = recall
    for m in all_moves:
        all_moves_hist[m // 10] += 1
    results_dict['all_moves_hist'] = all_moves_hist
    return results_dict, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_mode', action='store_true')
    args = parser.parse_args()

    # Determine mode based on command-line arguments
    dev_mode = args.dev_mode

    # Run the experiment
    experiment = Experiment(
        variables=[],
        dev_mode=dev_mode)

    experiment.run_experiment(
        function=experiment_main,
        time_limit=300)


if __name__ == "__main__":
    main()
