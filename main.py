import argparse

from config import *
from experiment import Experiment
from train import train_model


def run_training():
    # Train model
    return train_model()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_mode', action='store_true')
    args = parser.parse_args()

    # Determine mode based on command-line arguments
    dev_mode = args.dev_mode

    # Run the experiment
    experiment = Experiment(
        variables=[OPTIMIZER, LEARNING_RATE], dev_mode=dev_mode)

    experiment.run_experiment(
        function=run_training,
        time_limit=300)


if __name__ == "__main__":
    main()
